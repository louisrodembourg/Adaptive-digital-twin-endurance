"""
Confrontation des bornes du planificateur aux donnees.

Ces tests verrouillent surtout les trois pieges dans lesquels le protocole est
tombe avant d'arriver a sa forme actuelle, parce que chacun produisait un
resultat qui ressemblait a une mesure :

- la semaine evaluee contaminait les deux bornes de la difference, donc
  l'effort dur de cette semaine entrait mecaniquement dans l'issue ;
- le maximum croissait avec le nombre de tirages, donc une semaine suivie de
  plus de sorties « progressait » sans progresser ;
- une regle ne concluant pas ressortait comme une regle refutee.

Chacun a sa garde ici. Et aucun seuil n'est recopie : la mesure recoit les
bornes du planificateur en parametres, sans quoi elle validerait une regle que
le planificateur n'applique plus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.plan_rules import (
    CONTRADICTED,
    SUPPORTED,
    UNDETERMINED,
    RuleVerdict,
    evaluate_plan_rules,
    evaluate_rule,
    format_plan_rules,
    longest_streak,
    rule_floor,
    rule_masks,
    sign_test,
    weekly_frame,
)


def _daily(start="2018-01-01", days=400, load=50.0, rest_every=0):
    """Charge journaliere reguliere, avec un jour de repos tous les N jours."""
    dates = pd.date_range(start, periods=days, freq="D")
    values = np.full(days, load, dtype=float)
    if rest_every:
        values[::rest_every] = 0.0
    return pd.DataFrame({"date": dates, "trimp_bike": values,
                         "trimp": values})


def _power(start="2018-01-01", days=400, step=3, watts=200.0, slope=0.0):
    """Observations de puissance regulieres, avec une derive optionnelle."""
    dates = pd.date_range(start, periods=days // step, freq=f"{step}D")
    n = len(dates)
    base = watts + slope * np.arange(n, dtype=float)
    # Un bruit deterministe : sans dispersion, l'ecart-type vaut zero et le
    # module refuse l'athlete, ce qui est le comportement voulu mais pas ce
    # qu'on veut tester ici.
    base = base + 5.0 * np.sin(np.arange(n, dtype=float))
    return pd.DataFrame({"date": dates, "pi": base})


# ── longest_streak ────────────────────────────────────────────────────────

def test_une_serie_vide_vaut_zero():
    assert longest_streak(np.array([], dtype=bool)) == 0


def test_une_semaine_entierement_active_vaut_sept():
    assert longest_streak(np.ones(7, dtype=bool)) == 7


def test_un_jour_de_repos_coupe_la_serie():
    actifs = np.array([True, True, False, True, True, True, False])
    assert longest_streak(actifs) == 3


def test_la_serie_la_plus_longue_est_rendue_pas_la_derniere():
    actifs = np.array([True, True, True, True, False, True])
    assert longest_streak(actifs) == 4


# ── weekly_frame ──────────────────────────────────────────────────────────

def test_une_charge_absente_rend_un_tableau_vide():
    assert weekly_frame("a", pd.DataFrame(), _power()).empty


def test_une_colonne_de_charge_manquante_rend_un_tableau_vide():
    daily = _daily().rename(columns={"trimp_bike": "autre"})
    assert weekly_frame("a", daily, _power()).empty


def test_une_puissance_absente_rend_un_tableau_vide():
    assert weekly_frame("a", _daily(), pd.DataFrame()).empty


def test_une_puissance_constante_est_refusee():
    """Sans dispersion, reduire par l'ecart-type diviserait par zero."""
    plate = pd.DataFrame({"date": pd.date_range("2018-01-01", periods=100, freq="3D"),
                          "pi": np.full(100, 200.0)})
    assert weekly_frame("a", _daily(), plate).empty


def test_chaque_semaine_porte_son_acwr_et_son_issue():
    weeks = weekly_frame("a", _daily(), _power())
    assert not weeks.empty
    for colonne in ("acwr", "consecutive_days", "meilleur", "median",
                    "n_before", "n_after"):
        assert colonne in weeks.columns
    assert (weeks["uuid"] == "a").all()


def test_une_charge_constante_donne_un_acwr_de_un():
    weeks = weekly_frame("a", _daily(rest_every=0), _power())
    assert np.allclose(weeks["acwr"], 1.0, atol=1e-9)


def test_les_jours_sans_seance_coupent_la_serie_consecutive():
    """
    Un jour de repos tous les trois jours borne la serie a deux.

    La charge est reindexee sur un calendrier journalier complet avant tout
    comptage : si les lignes a charge nulle etaient absentes, un repos passerait
    inapercu et la serie compterait des seances au lieu de jours.
    """
    weeks = weekly_frame("a", _daily(rest_every=3), _power())
    assert weeks["consecutive_days"].max() <= 2


def test_la_semaine_evaluee_est_exclue_des_deux_fenetres():
    """
    Le piege central : l'effort de la semaine jugee ne doit pas entrer dans
    son issue. Une puissance exceptionnelle placee dans une seule semaine ne
    doit donc rien changer a l'issue de CETTE semaine.
    """
    power = _power()
    cible = pd.Timestamp("2018-06-10")
    weeks_sans = weekly_frame("a", _daily(), power)

    pique = power.copy()
    dans_la_semaine = (pique["date"] >= cible - pd.Timedelta(days=6)) & (pique["date"] <= cible)
    assert dans_la_semaine.any(), "la fixture doit poser une observation dans la semaine"
    pique.loc[dans_la_semaine, "pi"] = 500.0
    weeks_avec = weekly_frame("a", _daily(), pique)

    ligne_sans = weeks_sans[weeks_sans["week_end"] == cible]
    ligne_avec = weeks_avec[weeks_avec["week_end"] == cible]
    if ligne_sans.empty or ligne_avec.empty:
        pytest.skip("la semaine cible n'est pas exploitable dans cette fixture")
    # L'ecart-type de reference change avec le pic, donc l'issue reduite bouge
    # un peu ; ce qui compte est qu'elle ne saute pas de l'ordre du pic.
    ecart = abs(float(ligne_avec["meilleur"].iloc[0])
                - float(ligne_sans["meilleur"].iloc[0]))
    assert ecart < 1.0


def test_le_nombre_d_observations_des_deux_fenetres_est_rendu():
    """Sans ces comptages, le confond d'intensite de mesure est invisible."""
    weeks = weekly_frame("a", _daily(), _power())
    assert (weeks["n_before"] > 0).all()
    assert (weeks["n_after"] > 0).all()


def test_le_tampon_elargi_reduit_le_nombre_de_semaines_exploitables():
    serre = weekly_frame("a", _daily(), _power(), buffer_days=7, reach_days=42)
    large = weekly_frame("a", _daily(), _power(), buffer_days=28, reach_days=42)
    assert len(large) <= len(serre)


# ── sign_test ─────────────────────────────────────────────────────────────

def test_un_test_du_signe_sans_donnee_rend_un_nan():
    favour, n, p = sign_test(pd.Series([], dtype=float))
    assert (favour, n) == (0, 0)
    assert np.isnan(p)


def test_un_accord_unanime_sur_dix_vaut_deux_sur_mille_vingt_quatre():
    favour, n, p = sign_test(pd.Series([1.0] * 10))
    assert (favour, n) == (10, 10)
    assert p == pytest.approx(2 / 1024)


def test_un_partage_exact_ne_conclut_pas():
    _, _, p = sign_test(pd.Series([1.0, 1.0, -1.0, -1.0]))
    assert p == 1.0


def test_les_valeurs_absentes_sont_ecartees_du_compte():
    favour, n, _ = sign_test(pd.Series([1.0, np.nan, 1.0]))
    assert (favour, n) == (2, 2)


# ── RuleVerdict ───────────────────────────────────────────────────────────

def _verdict(**kwargs):
    base = {"rule": "r", "statistic": "median", "n_athletes": 20,
            "mean_gap": 0.1, "median_gap": 0.1, "in_favour": 18,
            "p_value": 0.001, "floor": 0.0}
    return RuleVerdict(**(base | kwargs))


def test_une_majorite_significative_en_faveur_est_soutenue():
    assert _verdict().verdict == SUPPORTED


def test_une_majorite_significative_contre_est_contredite():
    assert _verdict(in_favour=2).verdict == CONTRADICTED


def test_un_p_non_significatif_est_indeterminable_et_non_absent():
    """
    La distinction qui compte : « le test ne conclut pas » n'est pas « la regle
    ne sert a rien ». Le planificateur applique encore la borne, et presenter
    une absence de mesure comme une absence d'effet est l'erreur que ce depot
    s'emploie a ne pas commettre.
    """
    assert _verdict(p_value=0.4, in_favour=12).verdict == UNDETERMINED


def test_aucun_athlete_comparable_est_indeterminable():
    assert _verdict(n_athletes=0, p_value=float("nan")).verdict == UNDETERMINED


def test_la_description_dit_le_plancher_atteint_sans_injection():
    assert "sans injection" in _verdict(floor=0.0).describe()


def test_la_description_dit_un_plancher_non_atteint():
    assert "non atteint" in _verdict(floor=None).describe()


# ── evaluate_rule et rule_floor ───────────────────────────────────────────

def _weeks_synthetiques(n_athletes=12, n_weeks=40, effet=0.0, seed=0):
    """
    Semaines fabriquees : la moitie respecte la regle, et l'effet est injecte.

    Les generateurs de donnees vivent dans les tests, jamais dans `src` : la
    regle du depot vaut aussi pour une mesure de regles.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_athletes):
        for w in range(n_weeks):
            respecte = w % 2 == 0
            rows.append({
                "uuid": f"ath{a}", "week_end": pd.Timestamp("2018-01-07") + pd.Timedelta(weeks=w),
                "acwr": 1.0 if respecte else 2.0,
                "consecutive_days": 2 if respecte else 6,
                "median": float(rng.normal(effet if respecte else 0.0, 0.3)),
                "meilleur": float(rng.normal(effet if respecte else 0.0, 0.3)),
                "n_before": 20, "n_after": 20, "load": 300.0, "active_days": 5,
            })
    return pd.DataFrame(rows)


def test_un_effet_franc_est_soutenu():
    weeks = _weeks_synthetiques(effet=0.5)
    v = evaluate_rule(weeks, weeks["acwr"] <= 1.5, "borne")
    assert v.verdict == SUPPORTED
    assert v.mean_gap > 0


def test_aucun_effet_ne_conclut_pas():
    weeks = _weeks_synthetiques(effet=0.0)
    assert evaluate_rule(weeks, weeks["acwr"] <= 1.5, "borne").verdict == UNDETERMINED


def test_un_athlete_sans_semaines_des_deux_cotes_sort_du_compte():
    weeks = _weeks_synthetiques(n_athletes=3)
    # Un quatrieme athlete qui ne viole jamais la borne n'est pas comparable.
    seul = weeks[weeks["uuid"] == "ath0"].copy()
    seul["uuid"] = "ath_pur"
    seul["acwr"] = 1.0
    weeks = pd.concat([weeks, seul], ignore_index=True)
    v = evaluate_rule(weeks, weeks["acwr"] <= 1.5, "borne")
    assert v.n_athletes == 3


def test_le_plancher_est_nul_quand_l_effet_est_deja_detecte():
    weeks = _weeks_synthetiques(effet=0.5)
    assert rule_floor(weeks, weeks["acwr"] <= 1.5, "median") == 0.0


def test_le_plancher_est_atteint_par_injection_quand_rien_n_est_detecte():
    weeks = _weeks_synthetiques(effet=0.0)
    plancher = rule_floor(weeks, weeks["acwr"] <= 1.5, "median")
    assert plancher is not None and plancher > 0.0


def test_un_plancher_sans_athlete_comparable_est_absent():
    weeks = _weeks_synthetiques(n_athletes=2, n_weeks=4)
    assert rule_floor(weeks, weeks["acwr"] <= 1.5, "median") is None


# ── rule_masks et evaluate_plan_rules ─────────────────────────────────────

def test_les_seuils_viennent_des_parametres_et_ne_sont_pas_recopies():
    """
    Un seuil recopie derive le jour ou la configuration change, et la mesure
    valide alors une regle que le planificateur n'applique plus.
    """
    weeks = _weeks_synthetiques()
    masques = rule_masks(weeks, acwr_upper=1.2, acwr_lower=0.5, max_consecutive=4)
    noms = " ".join(masques)
    assert "1.2" in noms and "0.5" in noms and "4 jours" in noms
    assert masques["ACWR <= 1.2 (borne haute)"].equals(weeks["acwr"] <= 1.2)


def test_chaque_regle_est_jugee_sur_les_deux_statistiques():
    """
    Une regle qui ne tient que sur le meilleur effort tient sur un artefact
    d'echantillonnage : les deux lignes doivent etre publiees ensemble.
    """
    table = evaluate_plan_rules(_weeks_synthetiques(effet=0.4))
    assert set(table["statistique"]) == {"meilleur", "median"}


def test_les_deux_regimes_sont_publies():
    """L'ecart entre les deux porte le confond de reprise apres coupure."""
    table = evaluate_plan_rules(_weeks_synthetiques(effet=0.4))
    assert table["regime"].nunique() == 2


def test_un_tableau_de_semaines_vide_rend_un_tableau_vide():
    assert evaluate_plan_rules(pd.DataFrame()).empty


def test_le_controle_de_reprise_ecarte_les_semaines_mal_mesurees():
    weeks = _weeks_synthetiques(effet=0.4)
    weeks.loc[weeks["acwr"] > 1.5, "n_before"] = 3
    table = evaluate_plan_rules(weeks, min_baseline_obs=15)
    restreint = table[table["regime"].str.contains("reference")]
    assert (restreint["athletes"] == 0).all()


def test_le_format_nomme_le_verdict_de_chaque_ligne():
    table = evaluate_plan_rules(_weeks_synthetiques(effet=0.4))
    rendu = format_plan_rules(table)
    assert "verdict" in rendu
    assert SUPPORTED in rendu


def test_un_format_vide_le_dit():
    assert "Aucune semaine" in format_plan_rules(pd.DataFrame())
