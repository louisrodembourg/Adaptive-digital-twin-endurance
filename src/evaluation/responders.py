"""
Qui repond ? La question ouverte du projet, posee hors echantillon.

L'effet de charge existe chez 23 athletes sur 61, est indetectable chez 33, et
degrade la prevision chez 5. C'est le resultat le plus utile du depot et le
plus inexploitable : savoir qu'un athlete sur trois repond ne dit pas LEQUEL,
et un planificateur a besoin du second.

Pourquoi des correlations ne suffisent pas
------------------------------------------
Chercher le predicteur dont la correlation avec le gain est la plus forte sur
61 athletes, c'est tester une dizaine d'hypotheses puis publier la meilleure.
Avec douze predicteurs, un |rho| de 0,25 apparait par hasard a peu pres a
coup sur. Le depot exige du hors echantillon partout ailleurs ; la meme regle
vaut ici. La question devient donc : **en retirant un athlete, les soixante
autres permettent-ils de predire s'il repond ?** Si l'exactitude obtenue ne
depasse pas celle de la classe majoritaire, rien ne predit la reponse, et le
dire est un resultat.

Le classifieur est volontairement pauvre
----------------------------------------
Un centroide par classe sur des predicteurs centres-reduits, et la classe du
plus proche. Avec 61 sujets et une dizaine de predicteurs, toute methode plus
riche apprendrait le bruit — et un surapprentissage ici produirait exactement
la fausse bonne nouvelle que ce module existe pour eviter. Le centrage, la
reduction ET les centroides sont recalcules DANS chaque pli : les calculer une
fois sur tout l'echantillon ferait fuir la cible dans le pretraitement, ce qui
est la facon la plus discrete de fabriquer une performance.

Deux familles de predicteurs, et la distinction porte l'enjeu
-------------------------------------------------------------
`AT_SIGNUP` ne contient que ce qu'on sait le premier jour : sexe et age. Tout
le reste demande de l'historique, parfois des annees. Un predicteur de la
seconde famille qui marcherait serait interessant scientifiquement et inutile
pour le demarrage a froid — c'est pourquoi les deux jeux sont evalues
separement, et jamais melanges dans une seule ligne de resultat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..layer_b_twin.load_model import LOAD_COLUMNS

# Connus le premier jour, avant toute seance.
AT_SIGNUP: tuple[str, ...] = ("sexe_feminin", "age")

# Derives de l'historique. Un predicteur d'ici ne resout pas le demarrage a
# froid, meme s'il predit.
NEEDS_HISTORY: tuple[str, ...] = (
    "n_observations", "jours", "densite_obs_an", "trou_max_j",
    "part_velo_pct", "charge_hebdo_moy", "cv_charge_hebdo_pct",
    "jours_actifs_pct", "pente_charge_an", "w_median", "cv_w_pct",
)

RESPONDER = "repond"
NEUTRAL = "neutre"
ADVERSE = "degrade"

# Annee de reference pour l'age : le jeu de donnees GoldenCheetah OpenData a
# ete publie en 2020, et un age calcule sur l'annee courante vieillirait d'un
# an a chaque execution — le resultat cesserait d'etre reproductible.
DATASET_YEAR = 2020

# Six athletes du cache declarent une annee de naissance posterieure a la
# publication du jeu de donnees, ce qui donne des ages de -58 a -42 ans. Un age
# negatif entre sans bruit dans un predicteur et y reste : la borne est
# explicite, et les athletes ecartes sont COMPTES, pas effaces.
PLAUSIBLE_AGE = (10.0, 90.0)


@dataclass(frozen=True)
class Separation:
    """Ce qu'un jeu de predicteurs sait dire d'un athlete qu'il n'a pas vu."""

    family: str
    n_athletes: int
    n_features: int
    accuracy: float
    baseline: float
    p_value: float
    floor: float | None

    @property
    def separates(self) -> bool:
        """
        Vrai seulement si l'exactitude hors echantillon bat la classe
        majoritaire ET que l'ecart survit a son test.

        Battre la classe majoritaire de quelques points sur 61 sujets ne veut
        rien dire : un seul athlete reclasse vaut 1,6 point d'exactitude.
        """
        return (np.isfinite(self.p_value) and self.p_value < 0.05
                and self.accuracy > self.baseline)

    def describe(self) -> str:
        if self.n_athletes == 0:
            return f"{self.family} : aucun athlete exploitable."
        plancher = ("deja separe" if self.floor == 0.0
                    else f"plancher {self.floor:.0%}" if self.floor is not None
                    else "plancher non atteint")
        verdict = "SEPARE" if self.separates else "ne separe pas"
        return (f"{self.family} ({self.n_features} predicteur(s), "
                f"{self.n_athletes} athletes) : {verdict} — "
                f"{self.accuracy:.1%} contre {self.baseline:.1%} pour la classe "
                f"majoritaire, p = {self.p_value:.4f}, {plancher}")


def athlete_features(frames, daily_load: pd.DataFrame) -> dict | None:
    """
    Les predicteurs d'un athlete, ou None s'il n'en porte pas assez.

    La charge hebdomadaire passe par un reechantillonnage et non par un
    comptage de lignes : un jour sans seance doit peser zero et non manquer.
    """
    if daily_load.empty or "trimp_bike" not in daily_load.columns:
        return None
    obs = frames.power[["date", "pi"]].dropna().copy()
    if len(obs) < 10:
        return None
    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs.sort_values("date")
    watts = pd.to_numeric(obs["pi"], errors="coerce").to_numpy(dtype=float)
    etendue = int((obs["date"].max() - obs["date"].min()).days)
    if etendue <= 0 or not np.isfinite(watts.std()) or watts.mean() == 0:
        return None

    dl = daily_load.copy()
    dl["date"] = pd.to_datetime(dl["date"])
    dl = dl.sort_values("date")
    bike = pd.to_numeric(dl["trimp_bike"], errors="coerce").fillna(0.0)
    total = sum(pd.to_numeric(dl[c], errors="coerce").fillna(0.0).sum()
                for c in LOAD_COLUMNS if c in dl.columns)
    jours = (dl["date"] - dl["date"].min()).dt.days.to_numpy(dtype=float)
    hebdo = dl.set_index("date")["trimp_bike"].resample("7D").sum()
    gaps = obs["date"].diff().dt.days.dropna()

    age = (DATASET_YEAR - frames.year_of_birth
           if frames.year_of_birth else np.nan)
    if np.isfinite(age) and not PLAUSIBLE_AGE[0] <= age <= PLAUSIBLE_AGE[1]:
        age = np.nan
    sexe = frames.gender
    return {
        "uuid": frames.uuid,
        "sexe_feminin": (np.nan if not sexe
                         else float(str(sexe).strip().upper().startswith("F"))),
        "age": float(age) if np.isfinite(age) else np.nan,
        "n_observations": float(len(obs)),
        "jours": float(etendue),
        "densite_obs_an": 365.0 * len(obs) / etendue,
        "trou_max_j": float(gaps.max()) if len(gaps) else 0.0,
        "part_velo_pct": 100.0 * float(bike.sum()) / total if total else np.nan,
        "charge_hebdo_moy": float(hebdo.mean()) if len(hebdo) else np.nan,
        "cv_charge_hebdo_pct": (100.0 * float(hebdo.std() / hebdo.mean())
                                if len(hebdo) > 1 and hebdo.mean() else np.nan),
        "jours_actifs_pct": 100.0 * float((bike > 0).mean()),
        "pente_charge_an": (float(np.polyfit(jours, bike.to_numpy(dtype=float), 1)[0]
                                  * 365.0) if len(jours) > 2 else np.nan),
        "w_median": float(np.median(watts)),
        "cv_w_pct": 100.0 * float(watts.std() / watts.mean()),
    }


def label_response(cohort: pd.DataFrame, variant: str = "velo seul",
                   reference: str = "moyenne") -> pd.DataFrame:
    """
    Trois groupes, lus dans l'artefact de cohorte et jamais reconstruits.

    Rejouer la classification ici ferait deriver les deux chiffres le jour ou
    le protocole change, et la question « qui repond ? » porterait alors sur
    une reponse que le depot ne publie plus.
    """
    if cohort.empty:
        return pd.DataFrame(columns=["uuid", "skill", "groupe"])
    vue = cohort[(cohort["variante"] == variant)
                 & (cohort["reference"] == reference)].copy()
    vue["groupe"] = np.where(vue["ci_low"] > 0, RESPONDER,
                             np.where(vue["ci_high"] < 0, ADVERSE, NEUTRAL))
    return vue[["uuid", "skill", "groupe"]]


def correlation_table(table: pd.DataFrame, columns: tuple[str, ...],
                      n_permutations: int = 20000,
                      seed: int = 0) -> pd.DataFrame:
    """
    Correlation de rang de chaque predicteur avec le gain, et son p permute.

    Le p est permute et non analytique parce que les predicteurs sont
    fortement correles entre eux et tres loin d'etre normaux. La derniere
    colonne porte le seuil de Bonferroni : regarder douze predicteurs puis
    publier le meilleur demande d'en payer le prix.
    """
    present = [c for c in columns if c in table.columns]
    if not present or "skill" not in table.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows, ecartes = [], []
    for col in present:
        paire = table[[col, "skill"]].dropna()
        if len(paire) < 10:
            ecartes.append((col, f"{len(paire)} valeur(s)"))
            continue
        # Un predicteur constant n'a pas de correlation DEFINIE, et le p
        # permute d'un NaN ressortait a 0,0000 donc « significatif » : le seul
        # predicteur declare retenu de la premiere execution etait un
        # artefact. `sexe_feminin` vaut 0 chez les 61 athletes du cache.
        if paire[col].nunique() < 2:
            ecartes.append((col, f"constant a {paire[col].iloc[0]:g}"))
            continue
        rho = float(paire[col].corr(paire["skill"], method="spearman"))
        x = paire[col].rank().to_numpy(dtype=float)
        y = paire["skill"].rank().to_numpy(dtype=float)
        base = abs(np.corrcoef(x, y)[0, 1])
        tirages = np.array([abs(np.corrcoef(rng.permutation(x), y)[0, 1])
                            for _ in range(n_permutations)])
        rows.append({
            "predicteur": col, "n": len(paire), "rho": rho,
            "p": float(np.mean(tirages >= base)),
            "rho_detectable": float(np.quantile(tirages, 0.95)),
        })
    out = pd.DataFrame(rows)
    for col, raison in ecartes:
        out = pd.concat([out, pd.DataFrame([{
            "predicteur": col, "n": 0, "rho": np.nan, "p": np.nan,
            "rho_detectable": np.nan, "ecarte": raison}])], ignore_index=True)
    if out.empty:
        return out
    if "ecarte" not in out.columns:
        out["ecarte"] = None
    retenus = out["ecarte"].isna()
    # Le prix de Bonferroni se paie sur les predicteurs REELLEMENT testes.
    out["seuil_bonferroni"] = 0.05 / max(1, int(retenus.sum()))
    out["significatif"] = retenus & (out["p"] < out["seuil_bonferroni"])
    return out.sort_values("rho", na_position="last").reset_index(drop=True)


def _nearest_centroid_loo(x: np.ndarray, y: np.ndarray) -> float:
    """
    Exactitude en laissant un athlete dehors, pretraitement inclus.

    Le centrage, la reduction et les centroides sont recalcules sans l'athlete
    teste. Les calculer une fois sur tout l'echantillon ferait fuir la cible
    dans le pretraitement — la facon la plus discrete de fabriquer une
    performance hors echantillon.
    """
    n = len(y)
    if n < 10:
        return float("nan")
    bons = 0
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        xt, yt = x[train], y[train]
        mu, sigma = xt.mean(axis=0), xt.std(axis=0)
        sigma = np.where(sigma > 0, sigma, 1.0)
        zt = (xt - mu) / sigma
        zi = (x[i] - mu) / sigma
        classes = np.unique(yt)
        centroides = np.vstack([zt[yt == c].mean(axis=0) for c in classes])
        distances = np.linalg.norm(centroides - zi, axis=1)
        if classes[int(np.argmin(distances))] == y[i]:
            bons += 1
    return bons / n


def _binomial_p(successes: int, n: int, p0: float) -> float:
    """p unilateral : l'exactitude depasse-t-elle la classe majoritaire ?"""
    from math import comb
    if n == 0 or not np.isfinite(p0):
        return float("nan")
    p0 = min(max(p0, 1e-9), 1 - 1e-9)
    return float(min(1.0, sum(comb(n, k) * p0 ** k * (1 - p0) ** (n - k)
                              for k in range(successes, n + 1))))


def loo_separation(table: pd.DataFrame, columns: tuple[str, ...], family: str,
                   deltas: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.40),
                   ) -> Separation:
    """
    Les autres athletes permettent-ils de classer celui qu'on retire ?

    La cible est binaire — repondeur contre le reste — parce que trois classes
    dont une a cinq membres ne s'apprennent pas sur 61 sujets. Le plancher est
    obtenu en deplacant les predicteurs des repondeurs d'une fraction connue
    d'ecart-type : c'est le meme procede que `sensitivity.py`, et il est
    optimiste pour la meme raison.
    """
    present = [c for c in columns if c in table.columns]
    if not present or "groupe" not in table.columns:
        return Separation(family, 0, 0, float("nan"), float("nan"),
                          float("nan"), None)
    vue = table[[*present, "groupe"]].dropna()
    if len(vue) < 10:
        return Separation(family, len(vue), len(present), float("nan"),
                          float("nan"), float("nan"), None)

    x = vue[present].to_numpy(dtype=float)
    y = (vue["groupe"] == RESPONDER).to_numpy()
    base = float(max(y.mean(), 1.0 - y.mean()))
    accuracy = _nearest_centroid_loo(x, y)
    p = _binomial_p(round(accuracy * len(y)), len(y), base)

    floor = None
    sigma = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    for delta in sorted(deltas):
        decale = x.copy()
        decale[y] = decale[y] + delta * sigma
        acc = _nearest_centroid_loo(decale, y)
        if _binomial_p(round(acc * len(y)), len(y), base) < 0.05 \
                and acc > base:
            floor = float(delta)
            break
    return Separation(family, len(vue), len(present), accuracy, base, p, floor)


def format_separations(separations: list[Separation]) -> str:
    """Une ligne par famille de predicteurs."""
    if not separations:
        return "Aucune famille evaluable."
    return "\n".join(s.describe() for s in separations)


def format_correlations(table: pd.DataFrame) -> str:
    """Les correlations, avec le rho que l'echantillon saurait distinguer."""
    if table.empty:
        return "Aucun predicteur evaluable."
    header = (f"{'predicteur':22s} {'n':>4s} {'rho':>7s} {'p':>7s} "
              f"{'rho detectable':>15s}  retenu")
    lines = [header, "-" * len(header)]
    ecartes = table[table["ecarte"].notna()] if "ecarte" in table else table.iloc[:0]
    retenus = table[table["ecarte"].isna()] if "ecarte" in table else table
    for _, r in retenus.iterrows():
        lines.append(
            f"{r['predicteur'][:22]:22s} {int(r['n']):4d} {r['rho']:+7.3f} "
            f"{r['p']:7.4f} {r['rho_detectable']:15.3f}  "
            f"{'oui' if r['significatif'] else 'non'}")
    if not retenus.empty:
        lines.append("")
        lines.append(f"Seuil de Bonferroni pour {len(retenus)} predicteurs "
                     f"testes : p < {retenus['seuil_bonferroni'].iloc[0]:.4f}")
    # Un predicteur ecarte est rapporte comme ecarte : sans cette ligne, le
    # tableau se lit comme si tous avaient ete testes.
    for _, r in ecartes.iterrows():
        lines.append(f"{r['predicteur']} ecarte — {r['ecarte']}.")
    return "\n".join(lines)
