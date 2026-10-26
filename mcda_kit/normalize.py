"""Fonctions de valeur et normalisations.

Chaque critere peut porter une specification "normalization" :
  {"type": "identity"}                                 les notes sont deja comparables
  {"type": "anchored", "worst": W, "best": B, "lo": 1, "hi": 5}
        echelle lineaire fixe : W -> lo, B -> hi (les ancres ne dependent pas des offres,
        donc ajouter ou retirer une alternative ne change pas les notes des autres)
  {"type": "minmax", "lo": 0, "hi": 1}                 pire offre -> lo, meilleure -> hi
  {"type": "ratio", "hi": 1}                           x/max (max) ou min/x (min), fois hi
  {"type": "vector"}                                   x / ||x|| (TOPSIS), inverse pour min
Sans specification, un critere "max" est laisse tel quel et un critere "min" provoque
une erreur : additionner un cout brut avec des notes n'aurait pas de sens.
"""
from __future__ import annotations

import numpy as np


def anchored(x, worst: float, best: float, lo: float = 0.0, hi: float = 1.0):
    x = np.asarray(x, dtype=float)
    if worst == best:
        raise ValueError("ancres worst et best identiques")
    return lo + (hi - lo) * (x - worst) / (best - worst)


def min_max(x, direction: str = "max", lo: float = 0.0, hi: float = 1.0):
    x = np.asarray(x, dtype=float)
    a, b = x.min(), x.max()
    if a == b:
        return np.full_like(x, hi)
    worst, best = (a, b) if direction == "max" else (b, a)
    return anchored(x, worst, best, lo, hi)


def ratio(x, direction: str = "max", hi: float = 1.0):
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("la normalisation 'ratio' exige des valeurs strictement positives")
    return hi * (x / x.max() if direction == "max" else x.min() / x)


def vector(x, direction: str = "max"):
    x = np.asarray(x, dtype=float)
    r = x / np.linalg.norm(x)
    return r if direction == "max" else 1 - r


def normalize_column(x, direction: str, spec: dict | None):
    spec = spec or {}
    t = spec.get("type", "identity")
    if t == "identity":
        if direction == "min":
            raise ValueError("critere a minimiser sans normalisation : precisez-en une")
        return np.asarray(x, dtype=float)
    if t == "anchored":
        return anchored(x, spec["worst"], spec["best"], spec.get("lo", 0.0), spec.get("hi", 1.0))
    if t == "minmax":
        return min_max(x, direction, spec.get("lo", 0.0), spec.get("hi", 1.0))
    if t == "ratio":
        return ratio(x, direction, spec.get("hi", 1.0))
    if t == "vector":
        return vector(x, direction)
    raise ValueError(f"normalisation inconnue : {t}")


def value_matrix(problem) -> np.ndarray:
    """Matrice des valeurs v_j(a) utilisee par la somme ponderee."""
    P = problem.filled().performances
    cols = [normalize_column(P[:, j], c.direction, c.normalization)
            for j, c in enumerate(problem.criteria)]
    return np.column_stack(cols)
