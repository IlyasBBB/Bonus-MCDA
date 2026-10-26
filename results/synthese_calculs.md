# Cas 6 - Synthese des calculs (generee par examples/cas6_reproduire.py)

## Entrees

| Cabinet | C1 | C2 | C3 | C4 (MDH) | C5 |
|---|---|---|---|---|---|
| Atlas | 5 | 2 | 2 | 4.4 | 5 |
| NorthBridge | 3* | 5 | 3* | 3.6 | 3 |
| Optima | 3* | 2 | 5 | 2.9 | 4 |
| Horizon | 2 | 3 | 4 | 4.1 | 4 |

`*` information absente de l'enonce, valeur neutre 3 (hypothese H4).
Contrainte : honoraires <= 4,5 MDH ; eliminations : aucune

| Poids | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| Membres (partie 1) | 0.250 | 0.250 | 0.250 | 0.125 | 0.125 |
| AHP (partie 2) | 0.2147 | 0.3760 | 0.2147 | 0.0738 | 0.1209 |

AHP : lambda_max = 5.0331, CI = 0.00828, RI = 1.12, CR = 0.0074

Seuils PROMETHEE/ELECTRE : q = 0,5 et p = 2 (notes), q = 0,3 et p = 1 MDH (C4) ; veto 3 points (notes).

## Resultats de reference

- Somme ponderee, poids membres : Optima 3.625 > NorthBridge 3.531 > Atlas 3.031 > Horizon 3.000
- PROMETHEE II, poids AHP : NorthBridge +0.200 > Optima +0.015 > Atlas -0.094 > Horizon -0.121
  - phi+ : Atlas 0.282, NorthBridge 0.448, Optima 0.267, Horizon 0.192
  - phi- : Atlas 0.376, NorthBridge 0.248, Optima 0.252, Horizon 0.313
- ELECTRE IS (pseudo-criteres, lambda 0,6, AHP) : noyau ['NorthBridge']
- ELECTRE vrai critere + veto simple (lambda 0,6, AHP) : noyau ['NorthBridge', 'Optima']

## Decomposition poids x methode

- WS/membres : Optima 3.625 > NorthBridge 3.531 > Atlas 3.031 > Horizon 3.000
- PROMETHEE/membres : Optima +0.135 > NorthBridge +0.060 > Atlas -0.058 > Horizon -0.137
- WS/AHP : NorthBridge 3.770 > Optima 3.322 > Horizon 3.047 > Atlas 2.951
- PROMETHEE/AHP : NorthBridge +0.200 > Optima +0.015 > Atlas -0.094 > Horizon -0.121
- PROMETHEE usuel/membres : Optima +0.208 > NorthBridge +0.083 > Horizon -0.125 > Atlas -0.167
- PROMETHEE usuel/AHP : NorthBridge +0.208 > Optima +0.038 > Horizon -0.042 > Atlas -0.204

## Seuils de bascule (un poids varie, les autres au prorata)

| Modele | Critere | Poids ref. | Bascule | Nouveau 1er |
|---|---|---|---|---|
| WS/membres | C1 | 0.250 | > 0.4217 | Atlas |
| WS/membres | C2 | 0.250 | > 0.2727 | NorthBridge |
| WS/membres | C3 | 0.250 | < 0.2131 | NorthBridge |
| WS/membres | C4 | 0.125 | < 0.0755 | NorthBridge |
| WS/membres | C5 | 0.125 | < 0.0345 | NorthBridge |
| WS/membres | C5 | 0.125 | > 0.4510 | Atlas |
| WS/AHP | C1 | 0.215 | > 0.4429 | Atlas |
| WS/AHP | C2 | 0.376 | < 0.2663 | Optima |
| WS/AHP | C3 | 0.215 | > 0.3586 | Optima |
| WS/AHP | C4 | 0.074 | > 0.2628 | Optima |
| WS/AHP | C5 | 0.121 | > 0.3763 | Atlas |
| PROMETHEE/membres | C1 | 0.250 | > 0.3520 | Atlas |
| PROMETHEE/membres | C2 | 0.250 | > 0.2872 | NorthBridge |
| PROMETHEE/membres | C3 | 0.250 | < 0.1954 | NorthBridge |
| PROMETHEE/membres | C4 | 0.125 | < 0.0217 | NorthBridge |
| PROMETHEE/membres | C5 | 0.125 | > 0.3501 | Atlas |
| PROMETHEE/AHP | C1 | 0.215 | > 0.3672 | Atlas |
| PROMETHEE/AHP | C2 | 0.376 | < 0.2845 | Optima |
| PROMETHEE/AHP | C3 | 0.215 | > 0.3266 | Optima |
| PROMETHEE/AHP | C4 | 0.074 | > 0.2641 | Optima |
| PROMETHEE/AHP | C5 | 0.121 | > 0.3051 | Atlas |

## Tirages aleatoires de poids (part de premieres places)

- WS Dirichlet filtre 10-40 % (1e5 tirages) (5458 tirages retenus) : Atlas 2.0 %, NorthBridge 16.8 %, Optima 81.2 %, Horizon 0.0 %
- WS AHP +/-30 % (10000 tirages retenus) : Atlas 0.0 %, NorthBridge 99.6 %, Optima 0.4 %, Horizon 0.0 %
- WS uniforme U(0,1;0,4) normalise (10000 tirages retenus) : Atlas 1.0 %, NorthBridge 11.5 %, Optima 87.5 %, Horizon 0.0 %
- PROMETHEE Dirichlet filtre 10-40 % (1e5 tirages) (5458 tirages retenus) : Atlas 11.1 %, NorthBridge 13.9 %, Optima 75.1 %, Horizon 0.0 %
- PROMETHEE AHP +/-30 % (10000 tirages retenus) : Atlas 0.0 %, NorthBridge 97.1 %, Optima 2.9 %, Horizon 0.0 %
- PROMETHEE uniforme U(0,1;0,4) normalise (10000 tirages retenus) : Atlas 7.3 %, NorthBridge 8.5 %, Optima 84.2 %, Horizon 0.0 %

## Perturbation d'une note (+/-1)

- WS/membres : 7 cas sur 28 changent le premier (reference Optima)
  - NorthBridge C1 3 -> 4 : NorthBridge
  - NorthBridge C3 3 -> 4 : NorthBridge
  - NorthBridge C5 3 -> 4 : NorthBridge
  - Optima C1 3 -> 2 : NorthBridge
  - Optima C2 2 -> 1 : NorthBridge
  - Optima C3 5 -> 4 : NorthBridge
  - Optima C5 4 -> 3 : NorthBridge
- PROMETHEE/AHP : 1 cas sur 28 changent le premier (reference NorthBridge)
  - Horizon C2 3 -> 4 : Horizon

## Seuils q/p des notes (PROMETHEE II)

- q=0, p=1 : AHP NorthBridge +0.194 > Optima +0.027 > Horizon -0.049 > Atlas -0.172 ; membres Optima +0.190 > NorthBridge +0.060 > Atlas -0.113 > Horizon -0.137
- q=0.5, p=2 : AHP NorthBridge +0.200 > Optima +0.015 > Atlas -0.094 > Horizon -0.121 ; membres Optima +0.135 > NorthBridge +0.060 > Atlas -0.058 > Horizon -0.137
- q=0, p=3 : AHP NorthBridge +0.219 > Optima +0.015 > Horizon -0.103 > Atlas -0.132 ; membres Optima +0.135 > NorthBridge +0.101 > Atlas -0.099 > Horizon -0.137
- q=1, p=3 : AHP NorthBridge +0.232 > Optima +0.009 > Atlas -0.112 > Horizon -0.130 ; membres NorthBridge +0.122 > Optima +0.107 > Atlas -0.092 > Horizon -0.137
- q=0, p=4 : AHP NorthBridge +0.167 > Optima +0.027 > Horizon -0.085 > Atlas -0.109 ; membres Optima +0.128 > NorthBridge +0.080 > Atlas -0.092 > Horizon -0.116
