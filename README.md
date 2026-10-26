# mcda_kit : une brique réutilisable d'aide à la décision multicritère

Prototype développé pour le bonus du projet MCDA (École Centrale Casablanca, 2026-2027,
cas 6 « Maghreb Industrial Group ») par Ilyas BEN ALLA, Ayoub ASSEFFAR, Zineb BAGHZALI,
Salma AKAZI et Hatim TAOUIL.

## 1. Objectif

Le programme calcule un modèle multicritère décrit dans un simple fichier JSON :
alternatives, critères, sens de préférence, poids, performances, seuils et contraintes.
Il implémente les méthodes utilisées dans notre projet, sans rien coder en dur :

| Brique | Ce qu'elle fait |
|---|---|
| contraintes | filtre conjonctif (ex. honoraires ≤ 4,5 MDH) avec motif d'élimination |
| normalisations | échelle ancrée fixe, min-max, ratio, vectorielle |
| somme pondérée | scores, contributions par critère, classement (méthode de la partie 1) |
| AHP | poids par vecteur propre, λmax, IC, RI de Saaty, CR (partie 2) |
| PROMETHEE I et II | 6 fonctions de préférence, π(a,b), φ⁺, φ⁻, φ (partie 2) |
| ELECTRE IS | concordance à pseudo-critères, veto (IS ou simple), graphe, noyau (partie 2) |
| sensibilité | seuils de bascule d'un poids, tirages aléatoires de poids, perturbation d'une note |

Le prototype structure et calcule un modèle. Il ne choisit ni les critères, ni les
poids, ni l'interprétation : ces choix restent ceux du décideur et de l'analyste.

## 2. Installation

Prérequis : Python 3.9 ou plus récent. Seule dépendance obligatoire : `numpy`.

```bash
cd mcda_kit_project
python -m pip install -r requirements.txt     # numpy (+ matplotlib, facultatif)
```

Aucune installation de paquet n'est nécessaire : on lance les commandes depuis ce dossier.

## 3. Utilisation en ligne de commande

```bash
# poids du fichier (« une voix par membre »), trois méthodes
python -m mcda_kit examples/cas6_mig.json

# poids calculés par AHP, analyse de sensibilité, export JSON complet
python -m mcda_kit examples/cas6_mig.json --weights ahp --sensitivity --seed 2026 --json resultats.json

# un autre problème, sans toucher au code
python -m mcda_kit examples/fournisseurs_demo.json --sensitivity
```

Options : `--methods ws promethee electre`, `--weights given|ahp|equal`,
`--lambda 0.65` (seuil de concordance ELECTRE), `--draws`, `--seed`, `--json`.

Extrait de sortie (cas 6, poids AHP) :

```
PROMETHEE II
rang  alternative  phi+   phi-   phi
1     NorthBridge  0.448  0.248  +0.200
2     Optima       0.267  0.252  +0.015
3     Atlas        0.282  0.376  -0.094
4     Horizon      0.192  0.313  -0.121

ELECTRE IS (lambda = 0.6)
  vetos          : ... Optima S NorthBridge bloque par C2 ...
  noyau          : NorthBridge
```

## 4. Utilisation comme bibliothèque

```python
from mcda_kit import load_problem, ahp, weighted_sum, promethee, electre_is
from mcda_kit.sensitivity import SCORERS, switch_table, monte_carlo

p = load_problem("examples/cas6_mig.json")
w = ahp(p.extra["ahp"]["pairwise"])["weights"]
print(promethee(p, w)["ranking"])
print(switch_table(p, SCORERS["promethee"], w))      # seuils de bascule par critère
print(monte_carlo(p, SCORERS["promethee"], mode="perturb", w_ref=w, rel=0.3))
```

`SCORERS` associe un nom de méthode à une fonction `score(problème, poids)`. Ajouter une
méthode qui produit un score (TOPSIS par exemple) suffit pour lui donner accès à toute
l'analyse de sensibilité.

## 5. Remplacer les données par celles d'un autre problème

Copier `examples/fournisseurs_demo.json` et modifier :

```json
{
  "name": "Mon problème",
  "alternatives": ["X", "Y", "Z"],
  "criteria": [
    {"name": "prix", "direction": "min", "weight": 0.4, "q": 1, "p": 5,
     "normalization": {"type": "minmax"}},
    {"name": "qualite", "direction": "max", "weight": 0.6, "q": 0.5, "p": 2, "v": 3,
     "missing_value": 3}
  ],
  "performances": {"X": [100, 4], "Y": [120, null], "Z": [90, 2]},
  "constraints": [{"criterion": "prix", "op": "<=", "value": 130}],
  "ahp": {"pairwise": [[1, "1/2"], [2, 1]]},
  "electre": {"lambda": 0.65, "veto_mode": "electre_is"}
}
```

Règles de saisie et contrôles effectués :

- `direction` vaut `max` ou `min` ; un critère `min` doit déclarer une normalisation
  pour la somme pondérée (additionner un coût brut et des notes n'aurait pas de sens).
- Les poids doivent être positifs et de somme 1, sinon le programme s'arrête
  (on peut aussi les faire calculer par AHP avec `--weights ahp`).
- `q ≤ p ≤ v` est vérifié pour chaque critère.
- Une performance `null` n'est **jamais** convertie en mauvaise note : sans
  `missing_value` déclarée, le programme s'arrête ; avec, il utilise la valeur indiquée
  et le signale dans la sortie.
- La matrice AHP doit être positive et réciproque ; le CR est affiché (seuil usuel 0,10).

## 6. Formules implémentées

- Somme pondérée : `V(a) = Σ w_j v_j(a)` ; échelle ancrée `v = lo + (hi-lo)(x-worst)/(best-worst)`.
- AHP : `w` = vecteur propre principal normalisé ; `IC = (λmax - n)/(n - 1)` ; `CR = IC/RI`.
- PROMETHEE : `π(a,b) = Σ w_j P_j(a,b)` ; `φ⁺(a) = Σ_x π(a,x)/(m-1)` ; `φ⁻(a) = Σ_x π(x,a)/(m-1)` ;
  `φ = φ⁺ - φ⁻`. Fonction « linear » (type V) : `P = 0` si `d ≤ q`, `(d-q)/(p-q)` entre `q`
  et `p`, `1` au-delà.
- ELECTRE IS : `c_j(a,b) = 1` si `g_j(b) - g_j(a) ≤ q_j`, `0` si `≥ p_j`, linéaire entre ;
  `C = Σ w_j c_j` ; veto si `g_j(b) - g_j(a) ≥ v_j - q_j η_j` avec
  `η_j = (1 - C - w_j)/(1 - λ - w_j)` (option `simple` : `≥ v_j`) ; `a S b` si `C ≥ λ` sans veto.
  Le noyau est calculé sur le graphe réduit (circuits fusionnés).
- Bascule d'un poids : `w_j = t`, les autres reçoivent `1 - t` au prorata de leurs poids
  de référence ; balayage puis dichotomie (précision 10⁻⁶).

## 7. Tests

```bash
python -m unittest discover -s tests -v
```

23 tests : exemples calculés à la main (somme pondérée, flux PROMETHEE d'un cas à
3 alternatives, matrice AHP cohérente, veto et noyau ELECTRE, seuil de bascule
analytique à 0,5), contrôles de validité (poids, données manquantes, seuils, contraintes)
et régression sur les chiffres de nos deux présentations du cas 6.

## 8. Reproduire le rapport

```bash
python examples/cas6_reproduire.py
```

Écrit dans `results/` : `cas6_resultats.json` (tous les calculs), `synthese_calculs.md`
(synthèse compacte) et `fig_bascule_c2.pdf` (figure du chapitre 6, si matplotlib est installé).

## 9. Architecture

```
mcda_kit/
  model.py        Criterion, Problem, load_problem, contrôles, contraintes
  normalize.py    fonctions de valeur
  methods.py      weighted_sum, ahp, promethee, electre_is, ranking
  sensitivity.py  redistribute, switch_table, monte_carlo, performance_perturbation
  cli.py          ligne de commande et export JSON
examples/         cas6_mig.json, fournisseurs_demo.json, cas6_reproduire.py
tests/            test_core.py
```

## 10. Limites connues

- Pas d'interface graphique : la priorité est la justesse et la réutilisabilité du calcul.
- ELECTRE ne fournit pas de score : il n'entre pas dans `switch_table` ni `monte_carlo`
  (on fait varier λ ou les poids « à la main », comme dans `cas6_reproduire.py`).
- Les performances sont des valeurs ponctuelles ; les intervalles et les scénarios
  pessimiste/optimiste doivent être traités en lançant plusieurs fichiers.
- Le noyau ELECTRE est calculé après fusion des circuits ; un graphe très cyclique donne
  un noyau peu informatif (le programme l'affiche tel quel).
- La matrice AHP n'est pas corrigée automatiquement lorsqu'elle est incohérente.
