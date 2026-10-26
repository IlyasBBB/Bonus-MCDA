"""Methodes d'agregation : somme ponderee, AHP, PROMETHEE II, ELECTRE IS.

Toutes les fonctions prennent un Problem (voir model.py) et, si besoin, un vecteur
de poids explicite ; elles renvoient des dictionnaires de resultats simples
(tableaux numpy + listes), faciles a exporter en JSON.
"""
from __future__ import annotations

import numpy as np

from .model import Problem, parse_number
from .normalize import value_matrix

# Indices aleatoires de Saaty (n = 1..10)
SAATY_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
            8: 1.41, 9: 1.45, 10: 1.49}


def ranking(alternatives, scores, higher_is_better: bool = True):
    """Liste (rang, alternative, score) ; les ex aequo exacts gardent l'ordre d'entree."""
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s if higher_is_better else s, kind="stable")
    return [(r + 1, alternatives[i], float(s[i])) for r, i in enumerate(order)]


def _weights(problem: Problem, weights=None) -> np.ndarray:
    w = problem.weights() if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (problem.n,):
        raise ValueError(f"{w.shape[0]} poids pour {problem.n} criteres")
    if np.any(w < 0) or abs(w.sum() - 1) > 1e-6:
        raise ValueError("les poids doivent etre positifs et de somme 1")
    return w


# ---------------------------------------------------------------------- somme ponderee
def weighted_sum(problem: Problem, weights=None) -> dict:
    """V(a) = sum_j w_j v_j(a), avec v_j la fonction de valeur declaree du critere j.

    Methode totalement compensatoire : un deficit sur un critere peut etre rachete
    par un gain sur un autre, au taux w_j / w_k.
    """
    w = _weights(problem, weights)
    V = value_matrix(problem)
    contrib = V * w
    scores = contrib.sum(axis=1)
    return {"method": "weighted_sum", "weights": w, "values": V,
            "contributions": contrib, "scores": scores,
            "ranking": ranking(problem.alternatives, scores)}


# ---------------------------------------------------------------------- AHP
def ahp(pairwise) -> dict:
    """Poids par la methode du vecteur propre de Saaty + controle de coherence.

    pairwise : matrice carree positive et reciproque (a_ij = 1 / a_ji), entrees
    numeriques ou chaines "1/3". Renvoie poids, lambda_max, CI, RI et CR.
    Un CR < 0,10 est le seuil usuel d'acceptation.
    """
    A = np.array([[parse_number(x) for x in row] for row in pairwise], dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("la matrice AHP doit etre carree")
    if np.any(A <= 0):
        raise ValueError("la matrice AHP doit etre strictement positive")
    if not np.allclose(A * A.T, 1.0, atol=1e-9):
        raise ValueError("la matrice AHP doit etre reciproque (a_ij * a_ji = 1)")
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    lam = float(vals[k].real)
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    ci = (lam - n) / (n - 1) if n > 2 else 0.0
    ri = SAATY_RI.get(n)
    if ri is None:
        raise ValueError("indice aleatoire de Saaty disponible pour n <= 10 seulement")
    cr = ci / ri if ri > 0 else 0.0
    # methode approchee (moyenne geometrique des lignes) : contrôle croise
    g = np.prod(A, axis=1) ** (1 / n)
    return {"method": "ahp", "weights": w, "lambda_max": lam, "CI": ci, "RI": ri,
            "CR": cr, "consistent": cr < 0.10, "weights_geometric": g / g.sum()}


# ---------------------------------------------------------------------- PROMETHEE
def preference_degree(d, kind: str, q: float = 0.0, p: float = 0.0, s: float | None = None):
    """Fonction de preference P(d) de PROMETHEE, d = avantage de a sur b (>= 0 utile).

    usual    (type I)   : 1 des que d > 0
    ushape   (type II)  : 1 si d > q
    vshape   (type III) : d/p entre 0 et p, puis 1
    level    (type IV)  : 0 si d <= q, 1/2 si q < d <= p, 1 au-dela
    linear   (type V)   : 0 si d <= q, (d-q)/(p-q) entre q et p, 1 au-dela
    gaussian (type VI)  : 1 - exp(-d^2 / 2 s^2)
    """
    d = np.asarray(d, dtype=float)
    out = np.zeros_like(d)
    pos = d > 0
    if kind == "usual":
        out[pos] = 1.0
    elif kind == "ushape":
        out[d > q] = 1.0
    elif kind == "vshape":
        if p <= 0:
            raise ValueError("vshape : p > 0 requis")
        out = np.clip(d / p, 0, 1)
    elif kind == "level":
        out[(d > q) & (d <= p)] = 0.5
        out[d > p] = 1.0
    elif kind == "linear":
        if p <= q:
            out[d > q] = 1.0
        else:
            out = np.clip((d - q) / (p - q), 0, 1)
    elif kind == "gaussian":
        if not s:
            raise ValueError("gaussian : parametre s requis")
        out[pos] = 1 - np.exp(-d[pos] ** 2 / (2 * s ** 2))
    else:
        raise ValueError(f"fonction de preference inconnue : {kind}")
    return out


def promethee(problem: Problem, weights=None) -> dict:
    """PROMETHEE I et II sur les performances brutes (pas de normalisation).

    pi(a,b)  = sum_j w_j P_j(a,b)           indice de preference agrege
    phi+(a)  = 1/(m-1) sum_x pi(a,x)        force (a prefere aux autres)
    phi-(a)  = 1/(m-1) sum_x pi(x,a)        faiblesse (les autres preferes a a)
    phi(a)   = phi+(a) - phi-(a)            flux net -> classement complet (II)
    PROMETHEE I : a P b si a fait au moins aussi bien sur phi+ et phi- et mieux sur l'un.
    """
    w = _weights(problem, weights)
    P = problem.filled().performances
    m, n = P.shape
    if m < 2:
        raise ValueError("PROMETHEE exige au moins deux alternatives")
    pref = np.zeros((n, m, m))
    for j, c in enumerate(problem.criteria):
        col = P[:, j]
        d = col[:, None] - col[None, :]
        if c.direction == "min":
            d = -d
        pref[j] = preference_degree(d, c.pref_function, c.q, c.p, c.s)
    pi = np.tensordot(w, pref, axes=1)
    np.fill_diagonal(pi, 0.0)
    phi_plus = pi.sum(axis=1) / (m - 1)
    phi_minus = pi.sum(axis=0) / (m - 1)
    phi = phi_plus - phi_minus
    # PROMETHEE I (preordre partiel)
    p1 = {}
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            ge = phi_plus[a] >= phi_plus[b] - 1e-12 and phi_minus[a] <= phi_minus[b] + 1e-12
            gt = phi_plus[a] > phi_plus[b] + 1e-12 or phi_minus[a] < phi_minus[b] - 1e-12
            if ge and gt:
                p1[(problem.alternatives[a], problem.alternatives[b])] = "P"
    return {"method": "promethee", "weights": w, "unicriterion": pref, "pi": pi,
            "phi_plus": phi_plus, "phi_minus": phi_minus, "phi": phi,
            "promethee1_preferences": sorted(p1),
            "ranking": ranking(problem.alternatives, phi)}


# ---------------------------------------------------------------------- ELECTRE IS
def electre_is(problem: Problem, weights=None, lam: float = 0.6,
               veto_mode: str = "electre_is") -> dict:
    """ELECTRE IS (pseudo-criteres, veto) : graphe de surclassement et noyau.

    Concordance partielle de "a surclasse b" sur j, avec delta = avantage de b sur a :
        c_j = 1 si delta <= q_j ; 0 si delta >= p_j ; (p_j - delta)/(p_j - q_j) entre.
    Concordance globale C(a,b) = sum_j w_j c_j(a,b).
    Veto sur j (veto_mode="electre_is", Roy et Skalka 1984) si
        delta >= v_j - q_j * eta_j,  eta_j = (1 - C(a,b) - w_j) / (1 - lam - w_j),
    soit un seuil effectif entre v_j - q_j et v_j. veto_mode="simple" : delta >= v_j.
    a S b si C(a,b) >= lam et aucun veto. Le noyau est calcule sur le graphe reduit
    (les circuits sont fusionnes, comme dans ELECTRE I/IS).
    """
    w = _weights(problem, weights)
    P = problem.filled().performances
    m, n = P.shape
    C = np.zeros((m, m))
    veto = np.zeros((m, m), dtype=bool)
    veto_on = [[[] for _ in range(m)] for _ in range(m)]
    partial = np.zeros((n, m, m))
    for j, c in enumerate(problem.criteria):
        col = P[:, j]
        delta = col[None, :] - col[:, None]          # delta[a, b] = g(b) - g(a)
        if c.direction == "min":
            delta = -delta
        if c.p > c.q:
            cj = np.clip((c.p - delta) / (c.p - c.q), 0, 1)
        else:
            cj = (delta <= c.q).astype(float)
        partial[j] = cj
    C = np.tensordot(w, partial, axes=1)
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            for j, c in enumerate(problem.criteria):
                if c.v is None:
                    continue
                d = P[b, j] - P[a, j]
                if c.direction == "min":
                    d = -d
                if veto_mode == "simple":
                    thr = c.v
                elif veto_mode == "electre_is":
                    denom = 1 - lam - w[j]
                    eta = (1 - C[a, b] - w[j]) / denom if denom > 1e-9 else 0.0
                    eta = min(max(eta, 0.0), 1.0)   # eta est dans [0, 1] quand C >= lambda
                    thr = c.v - c.q * eta
                else:
                    raise ValueError("veto_mode : 'electre_is' ou 'simple'")
                if d >= thr - 1e-9:
                    veto[a, b] = True
                    veto_on[a][b].append(c.name)
    S = (C >= lam - 1e-12) & ~veto
    np.fill_diagonal(S, False)
    alts = problem.alternatives
    kernel = _kernel(S)
    never_outranked = [alts[i] for i in range(m) if not S[:, i].any()]
    return {"method": "electre_is", "weights": w, "lambda": lam, "veto_mode": veto_mode,
            "concordance": C, "veto": veto,
            "veto_criteria": {(alts[a], alts[b]): veto_on[a][b]
                              for a in range(m) for b in range(m) if veto_on[a][b]},
            "outranking": S,
            "arcs": [(alts[a], alts[b]) for a in range(m) for b in range(m) if S[a, b]],
            "kernel": [alts[i] for i in kernel], "never_outranked": never_outranked}


def _sccs(S: np.ndarray) -> list[list[int]]:
    """Composantes fortement connexes (algorithme de Tarjan, version iterative simple)."""
    m = S.shape[0]
    reach = S.copy() | np.eye(m, dtype=bool)
    for k in range(m):                       # fermeture transitive (Warshall)
        reach |= reach[:, [k]] & reach[[k], :]
    comps, seen = [], set()
    for i in range(m):
        if i in seen:
            continue
        comp = [j for j in range(m) if reach[i, j] and reach[j, i]]
        seen.update(comp)
        comps.append(comp)
    return comps


def _kernel(S: np.ndarray) -> list[int]:
    """Noyau du graphe reduit : stable interieurement, absorbant exterieurement."""
    comps = _sccs(S)
    k = len(comps)
    G = np.zeros((k, k), dtype=bool)
    for x, cx in enumerate(comps):
        for y, cy in enumerate(comps):
            if x != y and S[np.ix_(cx, cy)].any():
                G[x, y] = True
    remaining, kernel = set(range(k)), []
    while remaining:
        sources = [x for x in remaining if not any(G[y, x] for y in remaining if y != x)]
        if not sources:                      # ne peut pas arriver sur un graphe reduit
            break
        kernel.extend(sources)
        dominated = {y for x in sources for y in remaining if G[x, y]}
        remaining -= set(sources) | dominated
    return sorted(i for x in kernel for i in comps[x])
