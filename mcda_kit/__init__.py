"""mcda_kit : petite bibliotheque d'aide a la decision multicritere.

Briques disponibles
-------------------
- model        : description d'un probleme (alternatives, criteres, performances,
                 contraintes), chargement JSON, controles de validite.
- normalize    : fonctions de valeur / normalisations (ancree, min-max, ratio, vecteur).
- methods      : somme ponderee, AHP (poids + coherence), PROMETHEE II, ELECTRE IS.
- sensitivity  : seuils de bascule sur un poids, tirages aleatoires de poids,
                 perturbation d'une performance.
"""
from .model import Criterion, Problem, load_problem, parse_number
from .normalize import value_matrix
from .methods import weighted_sum, ahp, promethee, electre_is, ranking
from . import sensitivity

__all__ = ["Criterion", "Problem", "load_problem", "parse_number", "value_matrix",
           "weighted_sum", "ahp", "promethee", "electre_is", "ranking", "sensitivity"]
__version__ = "1.0.0"
