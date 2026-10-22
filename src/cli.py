"""
Points d'entree en ligne de commande.

Ces commandes sont aussi le mode d'emploi du depot : elles reproduisent, a
partir d'un jeu de donnees public, chaque chiffre affiche par le tableau de
bord. C'est ce qui rend les resultats verifiables plutot que declaratifs.

    python -m src.cli screen-athletes --n 200
    python -m src.cli ingest    [--athlete UUID] [--limit N]
    python -m src.cli calibrate
    python -m src.cli state
    python -m src.cli evaluate  [--models chronos2,ttm] [--horizon 28]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from .config import Config, ConfigError, clear_config_cache, load_config
from .evaluation.foundation import SPECS, build_forecast_context, is_available, variant_name
from .evaluation.report import (
    evaluate_by_duration,
    evaluate_results,
    format_durations,
    format_pairs,
    format_report,
    load_ablation_pairs,
    paired_table,
    results_table,
)
from .evaluation.sensitivity import (
    detectability_curve,
    format_detectability,
)
from .evaluation.validation import horizon_splits
from .ingestion.opendata import (
    OpenDataClient,
    ScreeningCriteria,
    screen_athletes,
)
from .layer_a_signal.envelope import EnvelopeSpec, rolling_envelope
from .pipeline.orchestrator import (
    OpenDataSpec,
    load_calibration,
    load_daily_load,
    load_perf_observations,
    load_sessions,
    provenance_of,
    run_pipeline,
)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


# ── screen-athletes ───────────────────────────────────────────────────────

def cmd_screen(args, cfg: Config) -> int:
    client = OpenDataClient(cache_dir=cfg.paths.cache, node=cfg.data_source.osf_node)
    criteria = ScreeningCriteria(
        min_runs=args.min_runs,
        min_span_days=args.min_span_days,
        min_hr_coverage=args.min_hr_coverage,
    )
    seen = 0

    def show(result):
        nonlocal seen
        seen += 1
        if result.passed or len(result.summary.runs) >= 20:
            print(f"{seen:4d}. {result.describe()}", flush=True)

    kept = screen_athletes(client, max_candidates=args.n, stop_after=args.keep,
                           criteria=criteria, on_result=show)
    print(f"\n{len(kept)} athlete(s) retenu(s) sur {seen} examine(s).")
    for r in kept:
        print(f"  {r.summary.uuid}  file_id={r.remote.file_id}  md5={r.remote.md5}")
    if kept:
        print(f"\nAjouter l'elu a {cfg.data_source.manifest}, puis `cli ingest`.")
    return 0 if kept else 1


# ── ingest ────────────────────────────────────────────────────────────────

def cmd_ingest(args, cfg: Config) -> int:
    uuid = args.athlete or cfg.data_source.athlete_ref
    if not uuid:
        print("Aucun athlete : passer --athlete ou renseigner data_source.athlete_ref.",
              file=sys.stderr)
        return 2

    print(f"Ingestion de {uuid}" + (f" (limite {args.limit})" if args.limit else ""))

    def progress(n, ref):
        if n % 25 == 0:
            print(f"  {n} seances traitees ({ref})", flush=True)

    report = run_pipeline(OpenDataSpec(uuid), cfg, limit=args.limit,
                          resume=not args.rebuild, progress=progress)
    print(f"\n{report}")

    sessions = load_sessions(cfg)
    if sessions.empty:
        print("Aucune seance enregistree.", file=sys.stderr)
        return 1

    daily = load_daily_load(cfg)
    obs = load_perf_observations(cfg)
    print(f"\n  sessions.parquet          {len(sessions):5d} seances")
    print(f"  daily_load.parquet        {len(daily):5d} jours de calendrier")
    print(f"  perf_observations.parquet {len(obs):5d} observations")
    print(f"\n  {provenance_of(sessions).banner}")
    return 0


# ── calibrate ─────────────────────────────────────────────────────────────

def cmd_calibrate(args, cfg: Config) -> int:
    """
    Reporte dans params.yaml ce que l'historique a permis d'estimer.

    FC max et FC repos restaient a 205 et 65, annotees « a calibrer », alors
    que tout le TRIMP en depend. Le chrono de reference remplace les deux
    constantes qui encodaient un objectif personnel.
    """
    calib = load_calibration(cfg)
    if calib is None:
        print("Aucune calibration : lancer `cli ingest` d'abord.", file=sys.stderr)
        return 1

    from .layer_a_signal.performance import ReferenceCurve

    curve = ReferenceCurve.from_dict(calib["reference_curve"])
    reference_time = curve.time_for_distance(cfg.goal.distance_m)

    print(f"Athlete {calib['athlete_id']} — {calib['n_sessions']} seances")
    print(f"  FC max estimee  : {_fmt(calib['hr_max_estimated'])} bpm")
    print(f"  FC repos estimee: {_fmt(calib['hr_rest_estimated'])} bpm")
    print(f"  courbe          : exposant {curve.exponent:.4f}, R2 {curve.r_squared:.4f}")
    print(f"  {cfg.goal.event} predit : {_hms(reference_time)}")
    print(f"  objectif a -{cfg.goal.improvement_pct:g}% : "
          f"{_hms(reference_time * (1 - cfg.goal.improvement_pct / 100))}")

    if args.dry_run:
        print("\n(--dry-run : params.yaml inchange)")
        return 0

    _patch_params_yaml(cfg, calib, reference_time)
    clear_config_cache()
    print("\nparams.yaml mis a jour.")
    return 0


def _patch_params_yaml(cfg: Config, calib: dict, reference_time_s: float) -> None:
    """
    Edition ciblee du YAML, ligne par ligne.

    On ne le reserialise pas : les commentaires expliquent l'origine de chaque
    seuil et valent mieux que la commodite d'un yaml.dump.
    """
    path = cfg.source_path
    lines = path.read_text(encoding="utf-8").splitlines()
    updates = {
        "  hr_max:": f"  hr_max: {calib['hr_max_estimated']:.0f}",
        "  hr_rest:": f"  hr_rest: {calib['hr_rest_estimated']:.0f}",
        "  reference_time_s:": f"  reference_time_s: {reference_time_s:.0f}",
    }
    seen: set[str] = set()
    out = []
    for line in lines:
        key = next((k for k in updates if line.startswith(k) and k not in seen), None)
        if key:
            comment = line.split("#", 1)
            suffix = f"  # {comment[1].strip()}" if len(comment) > 1 else ""
            out.append(updates[key] + (suffix if "estime" in suffix or "null" in suffix else ""))
            seen.add(key)
        else:
            out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ── state ─────────────────────────────────────────────────────────────────

def cmd_state(args, cfg: Config) -> int:
    sessions = load_sessions(cfg)
    if sessions.empty:
        print("Aucune donnee. Lancer `cli ingest`.", file=sys.stderr)
        return 1

    daily = load_daily_load(cfg)
    obs = load_perf_observations(cfg)
    prov = provenance_of(sessions)

    print(f"Provenance : {prov.banner}")
    print(f"Periode    : {daily['date'].min():%Y-%m-%d} -> {daily['date'].max():%Y-%m-%d} "
          f"({len(daily)} jours)")
    print(f"Charge     : TRIMP median {daily['trimp'].median():.1f}, "
          f"max {daily['trimp'].max():.1f}, "
          f"{(daily['trimp'] == 0).mean():.0%} de jours sans seance")

    by_sport = sessions["sport"].value_counts()
    print(f"Disciplines: {', '.join(f'{k} {v}' for k, v in by_sport.head(5).items())}")

    if not obs.empty:
        clean = obs[obs["quality_flags"] == ""]
        print(f"\nObservations de performance : {len(obs)} dont {len(clean)} sans reserve")
        print(f"  index median {obs['perf_index'].median():.3f}, "
              f"max {obs['perf_index'].max():.3f}")
    else:
        print("\nAucune observation de performance — le modele ne peut pas etre ajuste.")

    ref = pd.to_numeric(sessions.get("trimp_reference"), errors="coerce")
    ours = pd.to_numeric(sessions.get("trimp"), errors="coerce")
    both = pd.DataFrame({"a": ours, "b": ref}).dropna()
    if len(both) > 20:
        rho = both["a"].corr(both["b"], method="spearman")
        print(f"\nTRIMP recoupe contre GoldenCheetah sur {len(both)} seances : rho = {rho:.3f}")
    return 0


# ── evaluate ──────────────────────────────────────────────────────────────

RESULTS_FILE = "evaluation_results.parquet"
ABLATION_FILE = "evaluation_ablation.parquet"
PROVENANCE_FILE = "evaluation_provenance.json"
DURATIONS_FILE = "evaluation_durations.parquet"
FLOORS_FILE = "evaluation_detectability.parquet"


def _package_versions(models: tuple[str, ...]) -> dict[str, str]:
    """
    Versions exactes des paquets ayant produit les chiffres.

    Un score de modele de fondation n'est reproductible qu'a version de
    paquet et revision de poids donnees : les deux bougent plus vite que ce
    depot, et un chiffre sans sa version n'est pas verifiable.
    """
    wanted = {"chronos2": "chronos-forecasting", "tabpfn-ts": "tabpfn-time-series",
              "ttm": "granite-tsfm"}
    out = {}
    for name in ["torch", *[wanted[m] for m in models if m in wanted]]:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "absent"
    return out



#: Colonne de puissance critique retenue comme cible velo. 20 minutes est le
#: compromis usuel : assez long pour etre limite par l'aerobie, assez court
#: pour etre atteint souvent (417 observations contre 273 a soixante minutes).
BIKE_TARGET_COLUMN = "cp20_w"

#: Un watt est un watt, sur route comme sur home-trainer : les deux
#: disciplines se regroupent pour une PUISSANCE, ce qui serait faux pour une
#: vitesse.
BIKE_SPORTS = ("bike", "virtualride")


def _evaluate_bike_target(sessions, daily, tau, horizon_days, min_train_days, spec):
    """
    Confronte la charge velo a une performance velo.

    C'est le seul appariement du depot ou la discipline qui porte la charge
    est celle dont on mesure la performance. Il sert aussi de controle de
    validite discriminante : la charge de COURSE ne doit PAS predire une
    puissance en velo, et si elle le faisait, ce serait le signe d'une fuite
    dans l'alignement des dates, pas d'un transfert physiologique.
    """
    if sessions.empty or BIKE_TARGET_COLUMN not in sessions.columns:
        return None, None
    lowered = sessions["sport"].astype(str).str.lower()
    bike = sessions[lowered.str.startswith(BIKE_SPORTS)]
    source = (bike[["date", BIKE_TARGET_COLUMN]].dropna()
              .rename(columns={BIKE_TARGET_COLUMN: "pi"}))
    if len(source) < 40:
        return None, None

    context = build_forecast_context(source, daily, "pi",
                                     window_days=spec.window_days,
                                     min_periods=spec.min_periods)
    dense = (context.frame[[context.target_column]].dropna()
             .rename(columns={context.target_column: "perf"}).reset_index())
    splits = list(horizon_splits(dense["date"], horizon_days=horizon_days,
                                 min_train_days=min_train_days,
                                 step_days=horizon_days))
    if not splits:
        return None, None
    results = evaluate_results(dense, daily, target="perf", taus=[tau], splits=splits)
    floors = detectability_curve(dense, daily, target="perf", tau1=tau[0],
                                 tau2=tau[1], horizon_days=horizon_days,
                                 min_train_days=min_train_days,
                                 column="trimp_bike")
    return results, floors

def cmd_evaluate(args, cfg: Config) -> int:
    daily = load_daily_load(cfg)
    obs = load_perf_observations(cfg)
    if daily.empty or obs.empty:
        print("Aucune donnee a evaluer. Lancer `cli ingest`.", file=sys.stderr)
        return 1

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    for model in models:
        if model not in SPECS:
            print(f"Modele inconnu : {model}. Connus : {', '.join(SPECS)}", file=sys.stderr)
            return 2
        if not is_available(model):
            print(f"{model} : paquet absent, variante non evaluee. "
                  f"Installer avec `pip install -e \".[tsfm]\"`.", file=sys.stderr)

    taus = [(42.0, 12.0), (60.0, 15.0)]
    source = obs[["date", "perf_index"]].rename(columns={"perf_index": "pi"})
    spec = EnvelopeSpec("pi", 42, min_periods=3)
    timings: dict = {}
    tables = []

    # Protocole 1 : celui du depot, inchange. Les references n'ont pas besoin
    # d'une grille journaliere, et ces lignes doivent rester reproductibles au
    # chiffre pres — toute derive ici signale une regression, pas un resultat.
    blocked = rolling_envelope(source, spec).rename(columns={spec.label: "perf"})
    t1 = results_table(evaluate_results(blocked, daily, target="perf", taus=taus))
    t1.insert(0, "protocol", "blocs")
    tables.append(t1)
    print("Protocole « blocs » — 4 plis positionnels sur "
          f"{len(blocked)} points d'enveloppe\n")
    print(format_report(t1))

    # Protocole 2 : horizon calendaire fixe sur la grille journaliere, le seul
    # ou « prevoir » veut dire quelque chose. C'est aussi le seul que les
    # modeles de fondation peuvent consommer.
    context = build_forecast_context(source, daily, "pi",
                                     window_days=spec.window_days,
                                     min_periods=spec.min_periods)
    dense = (context.frame[[context.target_column]]
             .dropna()
             .rename(columns={context.target_column: "perf"})
             .reset_index())
    splits = list(horizon_splits(dense["date"], horizon_days=args.horizon,
                                 min_train_days=args.min_train, step_days=args.horizon))
    results = evaluate_results(dense, daily, target="perf", taus=taus, context=context,
                               models=models, splits=splits, device=args.device,
                               timings=timings)
    t2 = results_table(results)
    t2.insert(0, "protocol", f"horizon {args.horizon}j")
    tables.append(t2)
    print(f"\n\nProtocole « horizon {args.horizon}j » — {len(splits)} plis, "
          f"{len(dense)} jours d'enveloppe definie\n")
    print(format_report(t2))

    # L'ablation est l'experience : chaque modele est compare a lui-meme prive
    # de la charge, puis a la persistance. Un ecart de R2 agrege ne suffit pas
    # a affirmer un gain, d'ou l'intervalle apparie.
    pairs = load_ablation_pairs(*taus[0])
    for model in models:
        if model in SPECS and is_available(model):
            with_cov, without = variant_name(model, True), variant_name(model, False)
            pairs += [(with_cov, without), (with_cov, "persistance")]
    ablation = paired_table(results, pairs)
    if not ablation.empty:
        print("\n\nComparaisons appariees — bootstrap par blocs sur les plis\n")
        print(format_pairs(ablation))
        ablation.to_parquet(cfg.paths.processed / ABLATION_FILE, index=False)

    # Protocole 3 : une enveloppe PAR DUREE d'effort. `perf_index` prend un
    # maximum sur les durees avant que l'enveloppe en reprenne un sur 42
    # jours ; separer les durees est la seule facon de voir si une
    # dose-reponse existe sur l'une d'elles sans exister sur le scalaire.
    durations, duration_pairs = evaluate_by_duration(
        obs, daily, taus=[taus[0]], horizon_days=args.horizon,
        min_train_days=args.min_train, window_days=spec.window_days,
        min_periods=spec.min_periods)
    if not durations.empty:
        print("\n\nProtocole « par duree » — enveloppe de chaque plafond, "
              f"horizon {args.horizon}j\n")
        print(format_durations(durations))
        durations.to_parquet(cfg.paths.processed / DURATIONS_FILE, index=False)
        if not duration_pairs.empty:
            print("\n\nCharge specifique contre persistance, duree par duree\n")
            print(format_pairs(duration_pairs))

    # Protocole 4 : le plancher de detectabilite. Il ne mesure pas l'athlete,
    # il mesure le protocole — quel effet celui-ci saurait voir. Sans ce
    # nombre, « on n'a rien trouve » et « il n'y a rien » sont
    # indistinguables, et c'est le premier reproche a faire au resultat nul.
    floors = detectability_curve(dense, daily, target="perf", tau1=taus[0][0],
                                 tau2=taus[0][1], horizon_days=args.horizon,
                                 min_train_days=args.min_train)
    if not floors.empty:
        print("\n\nPlancher de detectabilite — effet de charge SYNTHETIQUE "
              "ajoute a l'enveloppe reelle\n")
        print(format_detectability(floors))
        floors.to_parquet(cfg.paths.processed / FLOORS_FILE, index=False)

    # Protocole 5 : la CIBLE VELO. C'est l'appariement que le depot n'avait
    # jamais fait — 68,5 % de la charge est cycliste, et la puissance critique
    # du velo est mesuree depuis 2016 (417 observations de `cp20_w`, espacees
    # de deux jours medians) sans jamais avoir servi de cible, parce que
    # `build_perf_observations` filtre sur `is_run`.
    bike_results, bike_floors = _evaluate_bike_target(
        load_sessions(cfg), daily, taus[0], args.horizon, args.min_train, spec)
    if bike_results is not None:
        t5 = results_table(bike_results)
        t5.insert(0, "protocol", "cible velo cp20")
        tables.append(t5)
        print("\n\nProtocole « cible velo » — enveloppe de cp20_w, "
              f"horizon {args.horizon}j\n")
        print(format_report(t5))
        bike_pairs = paired_table(bike_results, [
            (f"banister velo seul t{taus[0][0]:.0f}/{taus[0][1]:.0f}", other)
            for other in (f"banister t{taus[0][0]:.0f}/{taus[0][1]:.0f}",
                          f"banister course seule t{taus[0][0]:.0f}/{taus[0][1]:.0f}",
                          "moyenne", "persistance")
        ])
        if not bike_pairs.empty:
            print("\n\nCharge velo contre les autres, sur la cible velo\n")
            print(format_pairs(bike_pairs))
        if bike_floors is not None and not bike_floors.empty:
            print("\n\nPlancher de detectabilite sur la cible velo\n")
            print(format_detectability(bike_floors))

    table = pd.concat(tables, ignore_index=True)

    # `cross_validate` avale les exceptions d'un prediteur : une variante
    # cassee ressort en `folds = 0` avec des NaN partout, ce qui ressemble a un
    # resultat mesure. Le dire fort, et sortir en erreur, est la seule facon de
    # ne pas publier le silence d'un bug comme un constat.
    broken = table[table["folds"] == 0]["name"].tolist()
    if broken:
        print(f"\n\nAucun pli n'a abouti pour : {', '.join(broken)}. "
              "Ce n'est pas un resultat, c'est une panne.", file=sys.stderr)

    # Deux variantes qui n'ont pas couvert le meme nombre de plis ne sont pas
    # comparables ligne a ligne, meme si le tableau les range ensemble. Le dire
    # plutot que de laisser la colonne « plis » au lecteur attentif.
    for protocol, group in table[table["folds"] > 0].groupby("protocol"):
        counts = group.groupby("folds")["name"].apply(list)
        if len(counts) > 1:
            detail = " ; ".join(f"{n} pli(s) : {', '.join(names)}"
                                for n, names in counts.items())
            print(f"\nProtocole « {protocol} » — plis inegaux ({detail}). "
                  "Les scores ne portent pas sur les memes points.", file=sys.stderr)

    out = cfg.paths.processed / RESULTS_FILE
    table.to_parquet(out, index=False)

    provenance = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "packages": _package_versions(models),
        "models": {m: vars(SPECS[m]) for m in models if m in SPECS},
        "target": {"column": "pi", "window_days": spec.window_days,
                   "min_periods": spec.min_periods, **context.describe()},
        "protocols": {
            "blocs": {"n_splits": 5, "n_points": len(blocked)},
            f"horizon {args.horizon}j": {
                "horizon_days": args.horizon, "step_days": args.horizon,
                "min_train_days": args.min_train, "n_folds": len(splits),
                "n_points": len(dense),
            },
        },
        "latency_seconds": {
            name: {"calls": t.calls, "mean": round(t.mean_seconds, 3),
                   "horizons": t.horizons}
            for name, t in timings.items() if t.calls
        },
    }
    (cfg.paths.processed / PROVENANCE_FILE).write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n\nEcrit : {out}")
    print(f"Ecrit : {cfg.paths.processed / PROVENANCE_FILE}")
    return 1 if broken else 0


def _fmt(value) -> str:
    return f"{value:.0f}" if value is not None and np.isfinite(value) else "?"


def _hms(seconds: float) -> str:
    s = round(seconds)
    return f"{s // 3600}h{s % 3600 // 60:02d}:{s % 60:02d}"


# ── point d'entree ────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("screen-athletes", help="chercher un athlete exploitable")
    s.add_argument("--n", type=int, default=200, help="candidats a examiner")
    s.add_argument("--keep", type=int, default=3, help="s'arreter apres N retenus")
    s.add_argument("--min-runs", type=int, default=100)
    s.add_argument("--min-span-days", type=int, default=300)
    s.add_argument("--min-hr-coverage", type=float, default=0.85)
    s.set_defaults(func=cmd_screen)

    i = sub.add_parser("ingest", help="construire les artefacts depuis une source")
    i.add_argument("--athlete", help="UUID (defaut : data_source.athlete_ref)")
    i.add_argument("--limit", type=int, help="s'arreter apres N seances")
    i.add_argument("--rebuild", action="store_true", help="ignorer l'existant")
    i.set_defaults(func=cmd_ingest)

    c = sub.add_parser("calibrate", help="reporter les estimations dans params.yaml")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(func=cmd_calibrate)

    t = sub.add_parser("state", help="resumer l'etat des donnees")
    t.set_defaults(func=cmd_state)

    e = sub.add_parser("evaluate", help="confronter chaque variante aux memes references")
    e.add_argument("--models", default="", help="modeles de fondation, separes par une virgule")
    e.add_argument("--horizon", type=int, default=28, help="horizon de prevision, en jours")
    e.add_argument("--min-train", type=int, default=180,
                   help="historique minimal avant le premier pli, en jours")
    e.add_argument("--device", default="cpu", help="cpu ou mps")
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args, load_config())
    except ConfigError as exc:
        print(f"Configuration : {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrompu.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
