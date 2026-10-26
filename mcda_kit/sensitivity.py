"""Analyse de sensibilite generique.

Les fonctions recoivent une fonction de score score_fn(problem, weights) -> vecteur
(plus grand = meilleur), ce qui les rend utilisables avec la somme ponderee comme
avec le flux net de PROMETHEE ou toute methode ajoutee plus tard.
"""
from __future__ import annotations

import numpy as np

from .methods import promethee, weighted_sum

SCORERS = {
    "weighted_sum": lambda prob, w: weighted_sum(prob, w)["scores"],
    "promethee": lambda prob, w: promethee(prob, w)["phi"],
}


def redistribute(w_ref, j: int, t: float) -> np.ndarray:
    """Fixe w_j = t et repartit 1 - t sur les autres criteres au prorata de w_ref."""
    w_ref = np.asarray(w_ref, dtype=float)
    others = np.arange(len(w_ref)) != j
    rest = w_ref[others].sum()
    w = np.empty_like(w_ref)
    w[others] = (1 - t) * w_ref[others] / rest if rest > 0 else (1 - t) / others.sum()
    w[j] = t
    return w


def _leader(score_fn, prob, w):
    return int(np.argmax(score_fn(prob, w)))


def weight_switch_points(problem, score_fn, j: int, w_ref=None, t_max: float = 0.99,
                         steps: int = 990, tol: float = 1e-6) -> list[dict]:
    """Valeurs du poids j ou la premiere place change (un poids varie, les autres
    gardent leurs proportions). Balayage puis dichotomie pour la precision."""
    w_ref = problem.weights() if w_ref is None else np.asarray(w_ref, float)
    ts = np.linspace(0, t_max, steps + 1)
    leaders = [_leader(score_fn, problem, redistribute(w_ref, j, t)) for t in ts]
    out = []
    for k in range(1, len(ts)):
        if leaders[k] != leaders[k - 1]:
            lo, hi = ts[k - 1], ts[k]
            while hi - lo > tol:
                mid = (lo + hi) / 2
                if _leader(score_fn, problem, redistribute(w_ref, j, mid)) == leaders[k - 1]:
                    lo = mid
                else:
                    hi = mid
            out.append({"criterion": problem.criteria[j].name, "threshold": (lo + hi) / 2,
                        "before": problem.alternatives[leaders[k - 1]],
                        "after": problem.alternatives[leaders[k]]})
    return out


def switch_table(problem, score_fn, w_ref=None) -> list[dict]:
    """Pour chaque critere : poids de reference et seuils de bascule les plus proches.
    new_first = alternative en tete une fois le seuil franchi (vers le bas ou le haut)."""
    w_ref = problem.weights() if w_ref is None else np.asarray(w_ref, float)
    rows = []
    for j, c in enumerate(problem.criteria):
        pts = weight_switch_points(problem, score_fn, j, w_ref)
        below = [p for p in pts if p["threshold"] < w_ref[j]]
        above = [p for p in pts if p["threshold"] > w_ref[j]]
        sb = dict(below[-1], new_first=below[-1]["before"]) if below else None
        sa = dict(above[0], new_first=above[0]["after"]) if above else None
        rows.append({"criterion": c.name, "reference": float(w_ref[j]),
                     "switch_below": sb, "switch_above": sa, "all": pts})
    return rows


def monte_carlo(problem, score_fn, n: int = 10000, mode: str = "dirichlet",
                w_ref=None, rel: float = 0.3, bounds=None, seed: int = 0) -> dict:
    """Frequence de premiere place sous des poids aleatoires.

    mode="dirichlet" : poids uniformes sur le simplexe, filtres par bounds=(lo, hi)
                       si fourni (on ne garde que les tirages dans les bornes).
    mode="perturb"   : chaque poids de reference multiplie par U(1-rel, 1+rel),
                       puis renormalise.
    """
    rng = np.random.default_rng(seed)
    k = problem.n
    if mode == "dirichlet":
        W = rng.dirichlet(np.ones(k), size=n)
        if bounds is not None:
            lo, hi = bounds
            W = W[np.all((W >= lo) & (W <= hi), axis=1)]
    elif mode == "perturb":
        w_ref = problem.weights() if w_ref is None else np.asarray(w_ref, float)
        W = w_ref * rng.uniform(1 - rel, 1 + rel, size=(n, k))
        W = W / W.sum(axis=1, keepdims=True)
    else:
        raise ValueError("mode : 'dirichlet' ou 'perturb'")
    firsts = np.array([_leader(score_fn, problem, w) for w in W], dtype=int)
    freq = {a: float(np.mean(firsts == i)) for i, a in enumerate(problem.alternatives)}
    return {"mode": mode, "draws": int(len(W)), "first_place_share": freq}


def performance_perturbation(problem, score_fn, delta: float = 1.0, bounds=None,
                             criteria=None, w=None) -> list[dict]:
    """Modifie une performance a la fois de +/- delta et liste les cas ou le
    premier change. bounds = (min, max) admissibles de l'echelle."""
    w = problem.weights() if w is None else np.asarray(w, float)
    base = problem.filled()
    ref = _leader(score_fn, base, w)
    crit_names = criteria or [c.name for c in problem.criteria]
    out, tested = [], 0
    for a in problem.alternatives:
        for cn in crit_names:
            j = problem.crit_index(cn)
            x = base.performances[problem.alt_index(a), j]
            for d in (-delta, delta):
                nx = x + d
                if bounds is not None and not (bounds[0] <= nx <= bounds[1]):
                    continue
                tested += 1
                pb = base.with_performance(a, cn, nx)
                s = score_fn(pb, w)
                lead = int(np.argmax(s))
                if lead != ref:
                    out.append({"alternative": a, "criterion": cn, "from": x, "to": nx,
                                "new_first": problem.alternatives[lead]})
    return {"tested": tested, "changes": out,
            "reference_first": problem.alternatives[ref]}
