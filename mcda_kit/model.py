"""Description d'un probleme de decision multicritere et controles de validite.

Un probleme = des alternatives, des criteres (sens, poids, seuils...), une matrice
de performances brutes et, eventuellement, des contraintes d'admissibilite.
Rien n'est code en dur : tout vient du fichier JSON ou des arguments.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any

import numpy as np

_OPS = {"<=": lambda x, v: x <= v, "<": lambda x, v: x < v,
        ">=": lambda x, v: x >= v, ">": lambda x, v: x > v,
        "==": lambda x, v: x == v}


def parse_number(x: Any) -> float:
    """Convertit 3, 0.5, "1/3" ou "0,5" en float (pratique pour les matrices AHP)."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(Fraction(x.strip().replace(",", ".")))
    raise TypeError(f"valeur numerique attendue, recu {x!r}")


@dataclass
class Criterion:
    """Un critere et tous les parametres dont les methodes peuvent avoir besoin.

    direction      : "max" (plus c'est haut, mieux c'est) ou "min" (cout, delai...).
    weight         : poids (peut etre absent si les poids viennent d'AHP).
    q, p, v        : seuils d'indifference, de preference stricte et de veto, exprimes
                     dans l'unite du critere (PROMETHEE, ELECTRE).
    pref_function  : fonction de preference PROMETHEE : usual, ushape, vshape, level,
                     linear, gaussian (s = parametre de la gaussienne).
    normalization  : fonction de valeur pour la somme ponderee, par ex.
                     {"type": "anchored", "worst": 4.5, "best": 2.9, "lo": 1, "hi": 5}.
    missing_value  : valeur a utiliser si une performance manque. Si elle n'est pas
                     donnee, une performance manquante provoque une erreur : le programme
                     ne transforme jamais un manque d'information en mauvaise note.
    """
    name: str
    direction: str = "max"
    weight: float | None = None
    label: str = ""
    unit: str = ""
    q: float = 0.0
    p: float = 0.0
    v: float | None = None
    pref_function: str = "linear"
    s: float | None = None
    normalization: dict | None = None
    missing_value: float | None = None

    def __post_init__(self):
        if self.direction not in ("max", "min"):
            raise ValueError(f"critere {self.name} : direction 'max' ou 'min' attendue, "
                             f"recu {self.direction!r}")
        if self.q < 0 or self.p < 0:
            raise ValueError(f"critere {self.name} : seuils q et p doivent etre >= 0")
        if self.p and self.q > self.p:
            raise ValueError(f"critere {self.name} : il faut q <= p (q={self.q}, p={self.p})")
        if self.v is not None and self.v < self.p:
            raise ValueError(f"critere {self.name} : le veto v doit etre >= p")


@dataclass
class Problem:
    alternatives: list[str]
    criteria: list[Criterion]
    performances: np.ndarray                     # forme (m alternatives, n criteres), NaN = manquant
    constraints: list[dict] = field(default_factory=list)
    name: str = ""
    extra: dict = field(default_factory=dict)    # parametres de methode (AHP, lambda...)

    # ------------------------------------------------------------------ acces
    @property
    def m(self) -> int:
        return len(self.alternatives)

    @property
    def n(self) -> int:
        return len(self.criteria)

    def crit_index(self, name: str) -> int:
        for j, c in enumerate(self.criteria):
            if c.name == name:
                return j
        raise KeyError(f"critere inconnu : {name}")

    def alt_index(self, name: str) -> int:
        return self.alternatives.index(name)

    def weights(self) -> np.ndarray:
        w = [c.weight for c in self.criteria]
        if any(x is None for x in w):
            raise ValueError("des poids manquent : renseignez 'weight' ou une matrice AHP")
        return np.array(w, dtype=float)

    # ------------------------------------------------------------------ controles
    def validate(self, require_weights: bool = True, tol: float = 1e-6) -> list[str]:
        """Leve une erreur si le probleme est inutilisable ; renvoie des avertissements."""
        warnings: list[str] = []
        P = self.performances
        if P.shape != (self.m, self.n):
            raise ValueError(f"matrice {P.shape} incompatible avec {self.m} alternatives "
                             f"x {self.n} criteres")
        if len(set(self.alternatives)) != self.m:
            raise ValueError("noms d'alternatives en double")
        if len({c.name for c in self.criteria}) != self.n:
            raise ValueError("noms de criteres en double")
        if require_weights:
            w = self.weights()
            if np.any(w < 0):
                raise ValueError("un poids est negatif")
            if abs(w.sum() - 1) > tol:
                raise ValueError(f"la somme des poids vaut {w.sum():.6f} au lieu de 1 "
                                 "(utilisez Problem.normalized_weights() si c'est voulu)")
        for i, j in zip(*np.where(np.isnan(P))):
            c = self.criteria[j]
            if c.missing_value is None:
                raise ValueError(f"performance manquante ({self.alternatives[i]}, {c.name}) "
                                 "sans 'missing_value' declaree pour ce critere")
            warnings.append(f"{self.alternatives[i]} / {c.name} : information absente, "
                            f"valeur conventionnelle {c.missing_value} utilisee")
        for c in self.criteria:
            if c.weight == 0:
                warnings.append(f"le critere {c.name} a un poids nul")
        return warnings

    def filled(self) -> "Problem":
        """Copie ou les performances manquantes sont remplacees par missing_value."""
        P = self.performances.copy()
        for i, j in zip(*np.where(np.isnan(P))):
            mv = self.criteria[j].missing_value
            if mv is None:
                raise ValueError(f"performance manquante ({self.alternatives[i]}, "
                                 f"{self.criteria[j].name}) sans valeur conventionnelle")
            P[i, j] = mv
        return replace(self, performances=P)

    def imputed_cells(self) -> list[tuple[str, str]]:
        return [(self.alternatives[i], self.criteria[j].name)
                for i, j in zip(*np.where(np.isnan(self.performances)))]

    def admissible(self) -> tuple["Problem", list[str]]:
        """Applique les contraintes (filtre conjonctif). Renvoie le probleme reduit et
        la liste des eliminations motivees."""
        keep, reasons = [], []
        for i, a in enumerate(self.alternatives):
            ok = True
            for ct in self.constraints:
                j = self.crit_index(ct["criterion"])
                x = self.performances[i, j]
                if np.isnan(x) or not _OPS[ct["op"]](x, ct["value"]):
                    ok = False
                    reasons.append(f"{a} elimine : {ct['criterion']} = {x} ne respecte pas "
                                   f"{ct['op']} {ct['value']}")
            if ok:
                keep.append(i)
        return self.subset(keep), reasons

    # ------------------------------------------------------------------ variantes
    def subset(self, idx: list[int]) -> "Problem":
        return replace(self, alternatives=[self.alternatives[i] for i in idx],
                       performances=self.performances[idx].copy())

    def with_weights(self, w) -> "Problem":
        w = np.asarray(w, dtype=float)
        crits = [replace(c, weight=float(x)) for c, x in zip(self.criteria, w)]
        return replace(self, criteria=crits)

    def normalized_weights(self) -> "Problem":
        w = self.weights()
        return self.with_weights(w / w.sum())

    def with_performance(self, alt: str, crit: str, value: float) -> "Problem":
        P = self.performances.copy()
        P[self.alt_index(alt), self.crit_index(crit)] = value
        return replace(self, performances=P)

    def with_criterion(self, crit: str, **changes) -> "Problem":
        j = self.crit_index(crit)
        crits = list(self.criteria)
        crits[j] = replace(crits[j], **changes)
        return replace(self, criteria=crits)


# ---------------------------------------------------------------------- JSON
def load_problem(path_or_dict) -> Problem:
    """Charge un probleme depuis un fichier JSON (ou un dict deja lu).

    Format minimal :
    {
      "alternatives": ["A", "B"],
      "criteria": [{"name": "prix", "direction": "min", "weight": 0.4}, ...],
      "performances": {"A": [12.0, ...], "B": [null, ...]},
      "constraints": [{"criterion": "prix", "op": "<=", "value": 20}]
    }
    Les autres cles (ahp, electre, promethee, sensitivity...) sont conservees dans
    Problem.extra et lues par la ligne de commande.
    """
    if isinstance(path_or_dict, dict):
        d = path_or_dict
    else:
        with open(path_or_dict, encoding="utf-8") as fh:
            d = json.load(fh)
    alts = list(d["alternatives"])
    crits = []
    for c in d["criteria"]:
        c = dict(c)
        for key in ("q", "p", "v", "weight", "s", "missing_value"):
            if key in c and c[key] is not None:
                c[key] = parse_number(c[key])
        crits.append(Criterion(**c))
    perf = d["performances"]
    rows = []
    for a in alts:
        row = perf[a]
        if isinstance(row, dict):
            row = [row.get(c.name) for c in crits]
        if len(row) != len(crits):
            raise ValueError(f"{a} : {len(row)} performances pour {len(crits)} criteres")
        rows.append([math.nan if x is None else parse_number(x) for x in row])
    known = {"alternatives", "criteria", "performances", "constraints", "name"}
    prob = Problem(alternatives=alts, criteria=crits,
                   performances=np.array(rows, dtype=float),
                   constraints=list(d.get("constraints", [])), name=d.get("name", ""),
                   extra={k: v for k, v in d.items() if k not in known})
    return prob
