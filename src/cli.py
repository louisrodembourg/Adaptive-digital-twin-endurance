"""
Points d'entree en ligne de commande.

Ces commandes sont aussi le mode d'emploi du depot : elles reproduisent, a
partir d'un jeu de donnees public, chaque chiffre affiche par le tableau de
bord. C'est ce qui rend les resultats verifiables plutot que declaratifs.

    python -m src.cli screen-athletes --n 200
    python -m src.cli ingest    [--athlete UUID] [--limit N]
    python -m src.cli calibrate
    python -m src.cli state
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from .config import Config, ConfigError, clear_config_cache, load_config
from .ingestion.opendata import (
    OpenDataClient,
    ScreeningCriteria,
    screen_athletes,
)
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
