"""Reproduit tous les calculs du rapport (cas 6, Maghreb Industrial Group).

    python examples/cas6_reproduire.py            -> ecrit results/
Sorties : cas6_resultats.json (tout), synthese_calculs.md (livrable compact),
          fig_bascule_c2.pdf (si matplotlib est installe).
Le script n'utilise que l'API publique de mcda_kit : rien n'est specifique au cas
en dehors du fichier examples/cas6_mig.json et de quelques scenarios nommes ici.
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mcda_kit import ahp, electre_is, load_problem, promethee, ranking, weighted_sum  # noqa: E402
from mcda_kit.cli import _jsonable  # noqa: E402
from mcda_kit.sensitivity import (SCORERS, monte_carlo, performance_perturbation,  # noqa: E402
                                  redistribute, switch_table)

OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
SEED = 2026

P = load_problem(os.path.join(HERE, "cas6_mig.json"))
P, eliminated = P.admissible()
ALTS = P.alternatives
W_M = P.weights()                                   # une voix par membre (partie 1)
A = ahp(P.extra["ahp"]["pairwise"])
W_A = A["weights"]                                  # AHP (partie 2)
WS, PR = SCORERS["weighted_sum"], SCORERS["promethee"]
R = {"eliminated": eliminated, "imputed": P.imputed_cells(), "weights_members": W_M,
     "ahp": {k: A[k] for k in ("weights", "lambda_max", "CI", "RI", "CR", "weights_geometric")}}


def fmt_rank(alts, scores, nd=3, sign=False):
    f = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    return " > ".join(f"{a} {f.format(s)}" for _, a, s in ranking(alts, scores))


# 1. Partie 1 : somme ponderee -------------------------------------------------
ws_m = weighted_sum(P, W_M)
R["ws_members"] = {"values": ws_m["values"], "contrib": ws_m["contributions"],
                   "scores": ws_m["scores"]}
V = ws_m["values"]
R["dominance"] = [(ALTS[a], ALTS[b]) for a, b in itertools.permutations(range(len(ALTS)), 2)
                  if np.all(V[a] >= V[b]) and np.any(V[a] > V[b])]

# 2. Partie 2 : PROMETHEE II (poids AHP) et ELECTRE ---------------------------
pr_a = promethee(P, W_A)
R["promethee_ahp"] = {k: pr_a[k] for k in ("pi", "phi_plus", "phi_minus", "phi", "unicriterion")}
R["promethee1_ahp"] = pr_a["promethee1_preferences"]

# 3. Decomposition poids x methode -----------------------------------------------
decomp = {}
for wl, w in (("membres", W_M), ("AHP", W_A)):
    decomp[f"WS/{wl}"] = weighted_sum(P, w)["scores"]
    decomp[f"PROMETHEE/{wl}"] = promethee(P, w)["phi"]
Pu = P
for c in P.criteria:
    Pu = Pu.with_criterion(c.name, pref_function="usual")
decomp["PROMETHEE usuel/membres"] = promethee(Pu, W_M)["phi"]
decomp["PROMETHEE usuel/AHP"] = promethee(Pu, W_A)["phi"]
R["decomposition"] = decomp

# 4. ELECTRE : deux definitions de la concordance, lambda de 0,50 a 0,90 ----------
Pc = P
for c in P.criteria:
    Pc = Pc.with_criterion(c.name, q=0.0, p=0.0)          # "vrai critere" (ELECTRE I)
el = {}
for lab, prob, mode in (("pseudo-criteres q/p, veto IS", P, "electre_is"),
                        ("vrai critere, veto simple", Pc, "simple")):
    for wl, w in (("AHP", W_A), ("membres", W_M)):
        rows = []
        for lam in np.round(np.arange(0.50, 0.901, 0.05), 2):
            e = electre_is(prob, w, float(lam), mode)
            rows.append({"lambda": float(lam), "kernel": e["kernel"],
                         "never_outranked": e["never_outranked"], "arcs": e["arcs"]})
        el[f"{lab} / {wl}"] = rows
e_ref = electre_is(P, W_A, 0.6)
e_crisp = electre_is(Pc, W_A, 0.6, "simple")
R["electre"] = el
R["electre_ref"] = {"concordance": e_ref["concordance"], "arcs": e_ref["arcs"],
                    "kernel": e_ref["kernel"], "vetoes": e_ref["veto_criteria"]}
R["electre_crisp"] = {"concordance": e_crisp["concordance"], "arcs": e_crisp["arcs"],
                      "kernel": e_crisp["kernel"], "vetoes": e_crisp["veto_criteria"]}

# 5. Sensibilite : poids -----------------------------------------------------------
R["switch"] = {"WS/membres": switch_table(P, WS, W_M), "WS/AHP": switch_table(P, WS, W_A),
               "PROMETHEE/membres": switch_table(P, PR, W_M),
               "PROMETHEE/AHP": switch_table(P, PR, W_A)}

# 6. Monte Carlo (protocoles des deux presentations + perturbation) --------------
mc = {}
for name, fn in (("WS", WS), ("PROMETHEE", PR)):
    mc[f"{name} Dirichlet filtre 10-40 % (1e5 tirages)"] = monte_carlo(
        P, fn, n=100_000, mode="dirichlet", bounds=(0.1, 0.4), seed=SEED)
    mc[f"{name} AHP +/-30 %"] = monte_carlo(P, fn, n=10_000, mode="perturb", w_ref=W_A,
                                           rel=0.3, seed=SEED)
    rng = np.random.default_rng(SEED)
    Wu = rng.uniform(0.1, 0.4, size=(10_000, P.n))
    Wu /= Wu.sum(axis=1, keepdims=True)
    firsts = np.array([int(np.argmax(fn(P, w))) for w in Wu])
    mc[f"{name} uniforme U(0,1;0,4) normalise"] = {
        "draws": 10_000, "first_place_share": {a: float(np.mean(firsts == i))
                                               for i, a in enumerate(ALTS)}}
R["monte_carlo"] = mc

# 7. Sensibilite : donnees et seuils -------------------------------------------------
cfg = P.extra["sensitivity"]
R["perturbation"] = {
    "WS/membres": performance_perturbation(P, WS, 1, cfg["performance_bounds"],
                                           cfg["performance_criteria"], W_M),
    "PROMETHEE/AHP": performance_perturbation(P, PR, 1, cfg["performance_bounds"],
                                              cfg["performance_criteria"], W_A)}
R["horizon_c2_4"] = PR(P.with_performance("Horizon", "C2", 4), W_A)
R["fees"] = {f"Optima {x}": PR(P.with_performance("Optima", "C4", x), W_A)
             for x in (2.0, 2.9, 3.6, 4.5)}
thr = {}
for q, p in ((0.0, 1.0), (0.5, 2.0), (0.0, 3.0), (1.0, 3.0), (0.0, 4.0)):
    Pq = P
    for cn in ("C1", "C2", "C3", "C5"):
        Pq = Pq.with_criterion(cn, q=q, p=p, v=max(p, 3.0))
    thr[f"q={q:g}, p={p:g}"] = {"AHP": PR(Pq, W_A), "membres": PR(Pq, W_M)}
R["thresholds"] = thr

# 8. Controles complementaires de la partie 1 ----------------------------------------
norm = {}
for spec in ({"type": "minmax", "lo": 1, "hi": 5}, {"type": "ratio", "hi": 5}):
    Pn = P.with_criterion("C4", normalization=spec)
    norm[spec["type"]] = {"C4": weighted_sum(Pn, W_M)["values"][:, 3],
                          "membres": weighted_sum(Pn, W_M)["scores"],
                          "AHP": weighted_sum(Pn, W_A)["scores"]}
R["normalisations"] = norm
R["other_weights"] = {"20 %": weighted_sum(P, np.full(5, 0.2))["scores"],
                      "mission 30/25/20/10/15": weighted_sum(P, [.3, .25, .2, .1, .15])["scores"]}
R["substitution_MDH_per_point"] = {
    "membres": {c.name: W_M[j] / W_M[3] * 0.4 for j, c in enumerate(P.criteria) if j != 3},
    "AHP": {c.name: W_A[j] / W_A[3] * 0.4 for j, c in enumerate(P.criteria) if j != 3}}

with open(os.path.join(OUT, "cas6_resultats.json"), "w", encoding="utf-8") as fh:
    json.dump(_jsonable(R), fh, ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- synthese markdown
L = ["# Cas 6 - Synthese des calculs (generee par examples/cas6_reproduire.py)", "",
     "## Entrees", "", "| Cabinet | C1 | C2 | C3 | C4 (MDH) | C5 |", "|---|---|---|---|---|---|"]
for i, a in enumerate(ALTS):
    row = P.performances[i]
    L.append(f"| {a} | " + " | ".join("3*" if np.isnan(x) else f"{x:g}" for x in row) + " |")
L += ["", "`*` information absente de l'enonce, valeur neutre 3 (hypothese H4).",
      "Contrainte : honoraires <= 4,5 MDH ; eliminations : " + (", ".join(eliminated) or "aucune"),
      "", "| Poids | C1 | C2 | C3 | C4 | C5 |", "|---|---|---|---|---|---|",
      "| Membres (partie 1) | " + " | ".join(f"{x:.3f}" for x in W_M) + " |",
      "| AHP (partie 2) | " + " | ".join(f"{x:.4f}" for x in W_A) + " |",
      f"\nAHP : lambda_max = {A['lambda_max']:.4f}, CI = {A['CI']:.5f}, RI = {A['RI']}, CR = {A['CR']:.4f}",
      "\nSeuils PROMETHEE/ELECTRE : q = 0,5 et p = 2 (notes), q = 0,3 et p = 1 MDH (C4) ; veto 3 points (notes).",
      "", "## Resultats de reference", "",
      "- Somme ponderee, poids membres : " + fmt_rank(ALTS, ws_m["scores"]),
      "- PROMETHEE II, poids AHP : " + fmt_rank(ALTS, pr_a["phi"], sign=True),
      "  - phi+ : " + ", ".join(f"{a} {x:.3f}" for a, x in zip(ALTS, pr_a["phi_plus"])),
      "  - phi- : " + ", ".join(f"{a} {x:.3f}" for a, x in zip(ALTS, pr_a["phi_minus"])),
      f"- ELECTRE IS (pseudo-criteres, lambda 0,6, AHP) : noyau {e_ref['kernel']}",
      f"- ELECTRE vrai critere + veto simple (lambda 0,6, AHP) : noyau {e_crisp['kernel']}",
      "", "## Decomposition poids x methode", ""]
for k, s in decomp.items():
    L.append(f"- {k} : " + fmt_rank(ALTS, s, sign=k.startswith("PROM")))
L += ["", "## Seuils de bascule (un poids varie, les autres au prorata)", "",
      "| Modele | Critere | Poids ref. | Bascule | Nouveau 1er |", "|---|---|---|---|---|"]
for mk, rows in R["switch"].items():
    for r in rows:
        for side, sym in (("switch_below", "<"), ("switch_above", ">")):
            sp = r[side]
            if sp:
                L.append(f"| {mk} | {r['criterion']} | {r['reference']:.3f} | "
                         f"{sym} {sp['threshold']:.4f} | {sp['new_first']} |")
L += ["", "## Tirages aleatoires de poids (part de premieres places)", ""]
for k, v in mc.items():
    L.append(f"- {k} ({v['draws']} tirages retenus) : " +
             ", ".join(f"{a} {100 * x:.1f} %" for a, x in v["first_place_share"].items()))
L += ["", "## Perturbation d'une note (+/-1)", ""]
for k, v in R["perturbation"].items():
    L.append(f"- {k} : {len(v['changes'])} cas sur {v['tested']} changent le premier "
             f"(reference {v['reference_first']})")
    for c in v["changes"]:
        L.append(f"  - {c['alternative']} {c['criterion']} {c['from']:g} -> {c['to']:g} : {c['new_first']}")
L += ["", "## Seuils q/p des notes (PROMETHEE II)", ""]
for k, v in thr.items():
    L.append(f"- {k} : AHP {fmt_rank(ALTS, v['AHP'], sign=True)} ; membres {fmt_rank(ALTS, v['membres'], sign=True)}")
with open(os.path.join(OUT, "synthese_calculs.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")

# ---------------------------------------------------------------- figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
if plt is not None:
    COLORS = {"NorthBridge": "#2a78d6", "Optima": "#eb6834", "Atlas": "#1baf7a",
              "Horizon": "#eda100"}                       # couleur fixe par cabinet
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    ts = np.linspace(0, 0.6, 601)
    panels = ((axes[0], WS, W_M, "Somme pondérée, poids « membres »", "score sur 5"),
              (axes[1], PR, W_A, "PROMETHEE II, proportions AHP", "flux net φ"))
    for ax, fn, wref, title, ylab in panels:
        S = np.array([fn(P, redistribute(wref, 1, t)) for t in ts])
        for i, a in enumerate(ALTS):
            ax.plot(ts * 100, S[:, i], color=COLORS[a], lw=1.6)
        # etiquettes directes en bout de courbe, ecartees si elles se chevauchent
        ends = sorted((S[-1, i], a) for i, a in enumerate(ALTS))
        gap = 0.07 * (S.max() - S.min())
        ys = [ends[0][0]]
        for y, _ in ends[1:]:
            ys.append(max(y, ys[-1] + gap))
        for (y0, a), y in zip(ends, ys):
            ax.annotate(a, (60, y0), xytext=(62, y), textcoords="data", va="center",
                        fontsize=7.5, color="#0b0b0b", annotation_clip=False)
        sw = [r for r in switch_table(P, fn, wref) if r["criterion"] == "C2"][0]
        sp = sw["switch_above"] or sw["switch_below"]
        ax.axvline(sp["threshold"] * 100, color="#52514e", ls="--", lw=0.8)
        ax.axvline(wref[1] * 100, color="#52514e", ls=":", lw=0.8)
        ymax = ax.get_ylim()[1]
        ax.text(sp["threshold"] * 100, ymax, f" bascule\n {100 * sp['threshold']:.1f} %".replace(".", ","),
                fontsize=7, va="top", color="#52514e")
        ax.text(wref[1] * 100, ax.get_ylim()[0], f" réf. {100 * wref[1]:.1f} %".replace(".", ","),
                fontsize=7, va="bottom", color="#52514e")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("poids de C2, crédibilité sectorielle (%)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 60)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e8e7e4", lw=0.6)
    fig.tight_layout(w_pad=3.5)
    fig.savefig(os.path.join(OUT, "fig_bascule_c2.pdf"))
print("Ecrit :", ", ".join(sorted(os.listdir(OUT))))
