"""
Qui repond ? Les gardes de la question ouverte du projet.

Deux defauts ont ete trouves par la premiere execution de ce module, et
chacun aurait fait publier un faux :

- `sexe_feminin` vaut 0 chez les soixante-et-un athletes du cache. Un
  predicteur constant n'a pas de correlation definie, le p permute d'un NaN
  ressortait a 0,0000, et ce predicteur degenere etait le SEUL declare retenu.
- six athletes declarent une annee de naissance posterieure a la publication du
  jeu de donnees, donc des ages de -58 a -42 ans. Leur presence portait la
  correlation de l'age a +0,230 ; une fois ecartes, elle tombe a +0,132.

Les deux ont leur garde ici, ainsi que celle qui compte le plus : le
pretraitement du classifieur doit etre recalcule DANS chaque pli, sans quoi la
cible fuit par le centrage et l'exactitude hors echantillon est fabriquee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.responders import (
    ADVERSE,
    AT_SIGNUP,
    NEEDS_HISTORY,
    NEUTRAL,
    PLAUSIBLE_AGE,
    RESPONDER,
    Separation,
    _nearest_centroid_loo,
    athlete_features,
    correlation_table,
    format_correlations,
    format_separations,
    label_response,
    loo_separation,
)


class _Frames:
    """Le minimum qu'`athlete_features` lit d'un athlete."""

    def __init__(self, uuid="a", n=120, gender="M", year_of_birth=1985,
                 watts=200.0):
        self.uuid = uuid
        self.gender = gender
        self.year_of_birth = year_of_birth
        dates = pd.date_range("2018-01-01", periods=n, freq="3D")
        self.power = pd.DataFrame({
            "date": dates,
            "pi": watts + 5.0 * np.sin(np.arange(n, dtype=float)),
        })
        self.sessions = pd.DataFrame()


def _daily(days=400, load=50.0):
    dates = pd.date_range("2018-01-01", periods=days, freq="D")
    values = np.full(days, load, dtype=float)
    return pd.DataFrame({"date": dates, "trimp_bike": values,
                         "trimp_run": np.zeros(days), "trimp_other": np.zeros(days),
                         "trimp": values})


# ── athlete_features ──────────────────────────────────────────────────────

def test_une_charge_absente_rend_none():
    assert athlete_features(_Frames(), pd.DataFrame()) is None


def test_une_colonne_de_charge_velo_manquante_rend_none():
    daily = _daily().drop(columns=["trimp_bike"])
    assert athlete_features(_Frames(), daily) is None


def test_trop_peu_d_observations_rend_none():
    assert athlete_features(_Frames(n=5), _daily()) is None


def test_les_deux_familles_de_predicteurs_sont_produites():
    f = athlete_features(_Frames(), _daily())
    for colonne in AT_SIGNUP + NEEDS_HISTORY:
        assert colonne in f, colonne


def test_l_age_est_calcule_sur_l_annee_du_jeu_de_donnees():
    """
    Un age calcule sur l'annee courante vieillirait d'un an a chaque
    execution, et le resultat cesserait d'etre reproductible.
    """
    f = athlete_features(_Frames(year_of_birth=1985), _daily())
    assert f["age"] == 2020 - 1985


def test_un_age_implausible_devient_absent_et_non_negatif():
    """Six athletes du cache declarent une naissance posterieure a 2020."""
    f = athlete_features(_Frames(year_of_birth=2078), _daily())
    assert np.isnan(f["age"])


def test_les_bornes_d_age_encadrent_les_valeurs_retenues():
    bas = athlete_features(_Frames(year_of_birth=2020 - int(PLAUSIBLE_AGE[0])),
                           _daily())
    assert bas["age"] == PLAUSIBLE_AGE[0]
    hors = athlete_features(_Frames(year_of_birth=2020 - int(PLAUSIBLE_AGE[1]) - 1),
                            _daily())
    assert np.isnan(hors["age"])


def test_un_sexe_absent_devient_absent_et_non_zero():
    f = athlete_features(_Frames(gender=None), _daily())
    assert np.isnan(f["sexe_feminin"])


def test_le_sexe_feminin_est_code_a_un():
    assert athlete_features(_Frames(gender="F"), _daily())["sexe_feminin"] == 1.0


def test_une_part_velo_exclusive_vaut_cent_pour_cent():
    f = athlete_features(_Frames(), _daily())
    assert f["part_velo_pct"] == 100.0


def test_une_puissance_constante_donne_une_variabilite_nulle():
    """
    Contrairement a `plan_rules`, qui divise par l'ecart-type de la puissance
    et doit donc refuser une serie plate, ici la dispersion est un PREDICTEUR :
    zero est une valeur legitime et l'athlete reste dans l'echantillon.
    """
    frames = _Frames()
    frames.power["pi"] = 200.0
    f = athlete_features(frames, _daily())
    assert f is not None
    assert f["cv_w_pct"] == 0.0
    assert f["w_median"] == 200.0


def test_une_puissance_nulle_rend_none():
    """La diviser par sa moyenne serait une division par zero."""
    frames = _Frames()
    frames.power["pi"] = 0.0
    assert athlete_features(frames, _daily()) is None


# ── label_response ────────────────────────────────────────────────────────

def _cohorte():
    return pd.DataFrame([
        {"uuid": "a", "variante": "velo seul", "reference": "moyenne",
         "skill": 0.5, "ci_low": 0.1, "ci_high": 0.9},
        {"uuid": "b", "variante": "velo seul", "reference": "moyenne",
         "skill": 0.0, "ci_low": -0.3, "ci_high": 0.3},
        {"uuid": "c", "variante": "velo seul", "reference": "moyenne",
         "skill": -0.5, "ci_low": -0.9, "ci_high": -0.1},
        {"uuid": "d", "variante": "velo seul", "reference": "persistance",
         "skill": -1.0, "ci_low": -2.0, "ci_high": -0.5},
    ])


def test_les_trois_groupes_sortent_de_l_intervalle():
    labels = label_response(_cohorte()).set_index("uuid")["groupe"]
    assert labels["a"] == RESPONDER
    assert labels["b"] == NEUTRAL
    assert labels["c"] == ADVERSE


def test_seule_la_reference_demandee_est_lue():
    """La paire variante/reference est celle que le depot publie."""
    assert "d" not in set(label_response(_cohorte())["uuid"])


def test_une_cohorte_vide_rend_un_tableau_vide():
    assert label_response(pd.DataFrame()).empty


# ── correlation_table ─────────────────────────────────────────────────────

def _table(n=60, seed=0, lie=0.0):
    rng = np.random.default_rng(seed)
    skill = rng.normal(size=n)
    return pd.DataFrame({
        "uuid": [f"a{i}" for i in range(n)],
        "skill": skill,
        "bruit": rng.normal(size=n),
        "lie": lie * skill + rng.normal(scale=0.3, size=n),
        "constant": np.zeros(n),
        "groupe": np.where(skill > 0, RESPONDER, NEUTRAL),
    })


def test_un_predicteur_constant_est_ecarte_et_non_declare_significatif():
    """
    Le defaut de la premiere execution : le p permute d'un NaN ressortait a
    0,0000, donc le seul predicteur declare retenu etait degenere.
    """
    out = correlation_table(_table(), ("bruit", "constant"), n_permutations=500)
    ligne = out[out["predicteur"] == "constant"].iloc[0]
    assert not bool(ligne["significatif"])
    assert "constant" in str(ligne["ecarte"])


def test_un_predicteur_ecarte_est_rapporte_comme_ecarte():
    out = correlation_table(_table(), ("bruit", "constant"), n_permutations=500)
    assert "ecarte" in format_correlations(out)


def test_le_prix_de_bonferroni_se_paie_sur_les_predicteurs_testes():
    out = correlation_table(_table(), ("bruit", "lie", "constant"),
                            n_permutations=500)
    teste = out[out["ecarte"].isna()]
    assert len(teste) == 2
    assert teste["seuil_bonferroni"].iloc[0] == 0.05 / 2


def test_un_lien_franc_est_detecte():
    out = correlation_table(_table(lie=2.0), ("lie",), n_permutations=2000)
    assert out["p"].iloc[0] < 0.01


def test_du_bruit_ne_l_est_pas():
    out = correlation_table(_table(), ("bruit",), n_permutations=2000)
    assert out["p"].iloc[0] > 0.05


def test_le_rho_detectable_est_publie_avec_le_rho():
    """Un resultat nul sans son plancher confond « rien vu » et « rien »."""
    out = correlation_table(_table(), ("bruit",), n_permutations=2000)
    assert out["rho_detectable"].iloc[0] > 0.0


def test_des_colonnes_absentes_rendent_un_tableau_vide():
    assert correlation_table(_table(), ("inexistant",)).empty


# ── classifieur hors echantillon ──────────────────────────────────────────

def test_un_predicteur_parfait_est_classe_sans_erreur():
    y = np.array([True] * 20 + [False] * 20)
    x = np.where(y, 10.0, -10.0).reshape(-1, 1)
    assert _nearest_centroid_loo(x, y) == 1.0


def test_du_bruit_pur_ne_depasse_pas_le_hasard_de_beaucoup():
    rng = np.random.default_rng(0)
    y = np.array([True] * 30 + [False] * 30)
    x = rng.normal(size=(60, 3))
    assert _nearest_centroid_loo(x, y) < 0.7


def test_un_echantillon_trop_petit_rend_un_nan():
    y = np.array([True, False, True])
    assert np.isnan(_nearest_centroid_loo(np.zeros((3, 1)), y))


def test_le_pretraitement_ne_voit_pas_l_athlete_teste():
    """
    La garde qui compte. Un predicteur constant partout sauf chez un seul
    athlete ne doit pas permettre de le classer : si le centrage etait
    calcule sur tout l'echantillon, sa valeur aberrante le trahirait.
    """
    y = np.array([True] * 15 + [False] * 15)
    x = np.zeros((30, 1))
    x[0, 0] = 1000.0
    # Le pli qui teste l'athlete 0 ne voit qu'une colonne constante : les deux
    # centroides sont confondus et la classe rendue ne depend plus de lui.
    assert _nearest_centroid_loo(x, y) <= 0.6


# ── loo_separation ────────────────────────────────────────────────────────

def test_une_separation_franche_est_declaree():
    t = _table(n=60, lie=4.0)
    s = loo_separation(t, ("lie",), "essai")
    assert s.separates
    assert s.accuracy > s.baseline


def test_du_bruit_ne_separe_pas():
    assert not loo_separation(_table(n=60), ("bruit",), "essai").separates


def test_une_famille_absente_rend_une_separation_vide():
    s = loo_separation(_table(), ("inexistant",), "essai")
    assert s.n_athletes == 0 and not s.separates


def test_le_plancher_est_nul_quand_la_separation_existe_deja():
    assert loo_separation(_table(n=60, lie=4.0), ("lie",), "essai").floor == 0.0


def test_un_plancher_est_atteint_par_decalage_quand_rien_ne_separe():
    s = loo_separation(_table(n=60), ("bruit",), "essai",
                       deltas=(0.0, 0.5, 2.0, 8.0))
    assert s.floor is not None and s.floor > 0.0


def test_battre_la_classe_majoritaire_de_peu_ne_suffit_pas():
    """Un seul athlete reclasse vaut 1,6 point d'exactitude sur 61 sujets."""
    s = Separation("essai", 61, 3, 0.639, 0.623, 0.42, None)
    assert not s.separates


def test_la_description_dit_le_verdict_et_la_reference():
    s = Separation("essai", 61, 3, 0.557, 0.623, 0.88, None)
    rendu = s.describe()
    assert "ne separe pas" in rendu and "62.3%" in rendu


def test_un_format_sans_famille_le_dit():
    assert "Aucune famille" in format_separations([])
