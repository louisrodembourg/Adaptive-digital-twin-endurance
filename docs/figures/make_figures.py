"""
Figure de tete du README : le gain hors echantillon de chaque athlete.

Lit `cohort_replication.parquet`, produit par `python -m src.cli cohort`, et
ne recalcule rien : la figure doit montrer exactement les chiffres du
journal, pas une seconde estimation qui pourrait en diverger.

Deux fichiers, clair et sombre, parce que GitHub choisit l'image selon le
theme du lecteur (`<picture>` dans le README) et qu'un fond blanc fige dans
une page sombre se lit mal.

    python docs/figures/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from src.cli import COHORT_FILE  # noqa: E402
from src.evaluation.cohort import summarise_cohort  # noqa: E402

SORTIE = Path(__file__).resolve().parent

# Au-dela, un seul athlete (intervalle jusqu'a -18) ecraserait les 60 autres
# dans une bande illisible. Les marques coupees sont signalees par une fleche
# et leur valeur, jamais tronquees en silence.
Y_BAS, Y_HAUT = -1.6, 1.05

# Polarite et non categories : bleu et rouge sont les deux poles d'une paire
# divergente, le gris en est le milieu neutre — « l'intervalle ne conclut
# pas ». Palette validee (validate_palette.js) en clair et en sombre ; le gris
# sombre est fonce pour rester distinct du rouge chez un protanope, et la
# forme (plein / creux) double la couleur.
THEMES = {
    "light": {"fond": "#fcfcfb", "texte": "#0b0b0b", "texte2": "#52514e", "grille": "#e4e3df",
                  "aide": "#2a78d6", "neutre": "#9c9b96", "nuit": "#e34948"},
    "dark": {"fond": "#1a1a19", "texte": "#f0efec", "texte2": "#c3c2b7", "grille": "#33332f",
                 "aide": "#3987e5", "neutre": "#5e5d59", "nuit": "#e66767"},
}


def charger() -> tuple[pd.DataFrame, pd.Series]:
    chemin = RACINE / "data" / "processed" / COHORT_FILE
    if not chemin.exists():
        sys.exit(f"{chemin} absent : lancer `python -m src.cli cohort` d'abord.")
    table = pd.read_parquet(chemin)
    table = table[table["reference"] == "moyenne"].sort_values("skill").reset_index(drop=True)
    # Les comptes et le p du titre passent par la fonction qui les publie dans
    # le journal : recopies en dur, ils deviendraient faux le jour ou la
    # cohorte grandit, et personne ne s'en apercevrait.
    resume = summarise_cohort(table).iloc[0]
    # Meme regle que le journal : un intervalle qui exclut zero, et rien d'autre.
    table["groupe"] = "neutre"
    table.loc[table["ci_low"] > 0, "groupe"] = "aide"
    table.loc[table["ci_high"] < 0, "groupe"] = "nuit"
    assert (table["groupe"] == "aide").sum() == resume["ic_exclut_zero_pour"]
    return table, resume


def tracer(table: pd.DataFrame, resume: pd.Series, theme: str) -> Path:
    c = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    fig.patch.set_facecolor(c["fond"])
    ax.set_facecolor(c["fond"])

    for i, ligne in table.iterrows():
        couleur = c[ligne["groupe"]]
        bas, haut = max(ligne["ci_low"], Y_BAS), min(ligne["ci_high"], Y_HAUT)
        ax.plot([i, i], [bas, haut], color=couleur, lw=2, solid_capstyle="round", zorder=2)
        if ligne["ci_low"] < Y_BAS:
            ax.annotate("", xy=(i, Y_BAS - 0.02), xytext=(i, Y_BAS + 0.1),
                        arrowprops={"arrowstyle": "-|>", "color": couleur, "lw": 1.5}, zorder=2)
        y = min(max(ligne["skill"], Y_BAS + 0.04), Y_HAUT)
        plein = ligne["groupe"] != "neutre"
        ax.scatter(i, y, s=34, zorder=3, linewidths=1.6,
                   facecolors=couleur if plein else c["fond"], edgecolors=couleur)

    ax.axhline(0, color=c["texte2"], lw=1, zorder=1)
    ax.set_xlim(-1.5, len(table) + 0.5)
    ax.set_ylim(Y_BAS - 0.08, Y_HAUT + 0.05)
    ax.set_xticks([])
    ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis="y", colors=c["texte2"], length=0, labelsize=9)
    ax.grid(axis="y", color=c["grille"], lw=0.8, zorder=0)
    for cote in ax.spines.values():
        cote.set_visible(False)
    ax.set_ylabel("Skill gain vs. predicting the mean", color=c["texte2"], fontsize=9)
    ax.set_xlabel(f"{len(table)} athletes, ranked by gain  ·  bar = 95 % paired block-bootstrap interval",
                  color=c["texte2"], fontsize=9, labelpad=8)

    # Une note pour tous les points hors echelle plutot qu'une etiquette par
    # point : deux voisins coupes au meme plancher superposaient leurs valeurs.
    hors = table.loc[table["skill"] < Y_BAS, "skill"]
    if not hors.empty:
        valeurs = ", ".join(f"{v:+.2f}".replace("-", "−") for v in hors)
        # Au-dela du dernier intervalle coupe, sans quoi la note passe sur
        # les fleches des intervalles qui sortent eux aussi par le bas.
        x_libre = table.index[table["ci_low"] < Y_BAS].max() + 1
        ax.text(x_libre, Y_BAS + 0.04, f"◂ leftmost {len(hors)} off scale: {valeurs}",
                color=c["texte2"], fontsize=8, va="center")

    # Legende en ligne sous le sous-titre, a l'ecart des marques : posee dans
    # le trace, chaque etiquette recouvrait des intervalles. Comptes et forme
    # (plein / creux) dans la legende, pour que l'identite ne repose jamais
    # sur la couleur seule.
    n = table["groupe"].value_counts()
    poignees = [
        Line2D([], [], color=c[g], lw=2, marker="o", markersize=6, markeredgewidth=1.6,
               markerfacecolor=c["fond"] if g == "neutre" else c[g], label=libelle)
        for g, libelle in (("aide", f"Load helps · {n.get('aide', 0)}"),
                           ("neutre", f"Inconclusive · {n.get('neutre', 0)}"),
                           ("nuit", f"Load hurts · {n.get('nuit', 0)}"))
    ]
    fig.legend(handles=poignees, loc="upper left", bbox_to_anchor=(0.055, 0.875), ncol=3,
               frameon=False, fontsize=9.5, labelcolor=c["texte"], handlelength=1.6,
               columnspacing=1.8)

    fig.text(0.06, 0.95, f"Training load predicts performance for "
             f"{resume['ic_exclut_zero_pour']} of {resume['athletes']} athletes",
             color=c["texte"], fontsize=13, fontweight="bold", ha="left")
    fig.text(0.06, 0.895, "Bike load → bike power, out of sample, 28-day horizon. "
             f"Sign test across athletes: p = {resume['p_signe']:.3f}.",
             color=c["texte2"], fontsize=9.5, ha="left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.1)

    chemin = SORTIE / f"cohort_skill_{theme}.png"
    fig.savefig(chemin, facecolor=c["fond"])
    plt.close(fig)
    return chemin


if __name__ == "__main__":
    donnees, resume = charger()
    for nom in THEMES:
        print(tracer(donnees, resume, nom))
