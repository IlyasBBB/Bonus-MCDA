"""Tests independants : petits exemples calcules a la main, proprietes generales.
Lancement : python -m unittest discover -s tests -v
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcda_kit import Criterion, Problem, ahp, electre_is, load_problem, promethee, weighted_sum
from mcda_kit.methods import preference_degree
from mcda_kit.normalize import anchored, min_max, ratio
from mcda_kit.sensitivity import (SCORERS, monte_carlo, performance_perturbation,
                                  redistribute, weight_switch_points)


def tiny(perf, crits, alts=None):
    alts = alts or [f"a{i}" for i in range(len(perf))]
    return Problem(alternatives=alts, criteria=crits, performances=np.array(perf, float))


class TestNormalisation(unittest.TestCase):
    def test_anchored_fixed_scale(self):
        v = anchored([4.5, 2.9, 3.7], worst=4.5, best=2.9, lo=1, hi=5)
        np.testing.assert_allclose(v, [1, 5, 3])

    def test_minmax_and_ratio_directions(self):
        np.testing.assert_allclose(min_max([10, 20, 15], "min"), [1, 0, 0.5])
        np.testing.assert_allclose(ratio([2, 4], "min"), [1, 0.5])
        np.testing.assert_allclose(ratio([2, 4], "max"), [0.5, 1])

    def test_cost_without_normalisation_is_refused(self):
        p = tiny([[1.0], [2.0]], [Criterion("cout", "min", 1.0)])
        with self.assertRaises(ValueError):
            weighted_sum(p)


class TestModel(unittest.TestCase):
    def test_missing_value_is_never_silently_penalised(self):
        p = tiny([[np.nan], [2.0]], [Criterion("c", "max", 1.0)])
        with self.assertRaises(ValueError):
            p.validate()
        p2 = tiny([[np.nan], [2.0]], [Criterion("c", "max", 1.0, missing_value=3)])
        self.assertEqual(len(p2.validate()), 1)          # un avertissement explicite
        self.assertEqual(p2.filled().performances[0, 0], 3)

    def test_weights_must_sum_to_one(self):
        p = tiny([[1, 2]], [Criterion("x", weight=0.5), Criterion("y", weight=0.6)])
        with self.assertRaises(ValueError):
            p.validate()
        self.assertAlmostEqual(p.normalized_weights().weights().sum(), 1.0)

    def test_thresholds_checked(self):
        with self.assertRaises(ValueError):
            Criterion("x", q=2, p=1)
        with self.assertRaises(ValueError):
            Criterion("x", direction="up")

    def test_constraint_filter(self):
        p = tiny([[4.0], [5.0]], [Criterion("prix", "min", 1.0)], ["A", "B"])
        p.constraints = [{"criterion": "prix", "op": "<=", "value": 4.5}]
        kept, why = p.admissible()
        self.assertEqual(kept.alternatives, ["A"])
        self.assertIn("B", why[0])


class TestWeightedSum(unittest.TestCase):
    def test_hand_example(self):
        p = tiny([[4, 2], [3, 5]], [Criterion("x", weight=0.7), Criterion("y", weight=0.3)])
        r = weighted_sum(p)
        np.testing.assert_allclose(r["scores"], [3.4, 3.6])
        self.assertEqual(r["ranking"][0][1], "a1")


class TestAHP(unittest.TestCase):
    def test_consistent_matrix_recovers_weights(self):
        w = np.array([0.5, 0.3, 0.2])
        A = w[:, None] / w[None, :]
        r = ahp(A)
        np.testing.assert_allclose(r["weights"], w, atol=1e-9)
        self.assertAlmostEqual(r["CR"], 0.0, places=9)

    def test_fractions_and_reciprocity(self):
        r = ahp([[1, "1/3"], [3, 1]])
        np.testing.assert_allclose(r["weights"], [0.25, 0.75])
        with self.assertRaises(ValueError):
            ahp([[1, 2], [2, 1]])

    def test_inconsistent_matrix_flagged(self):
        r = ahp([[1, 9, "1/9"], ["1/9", 1, 9], [9, "1/9", 1]])
        self.assertFalse(r["consistent"])


class TestPromethee(unittest.TestCase):
    def test_preference_functions(self):
        d = np.array([-1, 0, 0.5, 1, 2, 3])
        np.testing.assert_allclose(preference_degree(d, "usual"), [0, 0, 1, 1, 1, 1])
        np.testing.assert_allclose(preference_degree(d, "linear", q=0.5, p=2),
                                   [0, 0, 0, 1 / 3, 1, 1])
        np.testing.assert_allclose(preference_degree(d, "level", q=0.5, p=2),
                                   [0, 0, 0, 0.5, 0.5, 1])
        np.testing.assert_allclose(preference_degree(d, "vshape", p=2),
                                   [0, 0, 0.25, 0.5, 1, 1])

    def test_hand_example_usual(self):
        # 3 alternatives, 2 criteres (poids 0,6 / 0,4), critere usuel
        crits = [Criterion("x", weight=0.6, pref_function="usual"),
                 Criterion("y", "min", 0.4, pref_function="usual")]
        p = tiny([[3, 10], [2, 5], [1, 7]], crits, ["A", "B", "C"])
        r = promethee(p)
        # pi(A,B)=0,6 ; pi(A,C)=0,6 ; pi(B,A)=0,4 ; pi(B,C)=1 ; pi(C,A)=0,4 ; pi(C,B)=0
        np.testing.assert_allclose(r["phi_plus"], [0.6, 0.7, 0.2])
        np.testing.assert_allclose(r["phi_minus"], [0.4, 0.3, 0.8])
        self.assertAlmostEqual(r["phi"].sum(), 0.0)
        self.assertEqual([a for _, a, _ in r["ranking"]], ["B", "A", "C"])


class TestElectre(unittest.TestCase):
    def test_veto_blocks_outranking(self):
        crits = [Criterion("x", weight=0.8, v=3), Criterion("y", weight=0.2, v=3)]
        p = tiny([[5, 1], [4, 5]], crits, ["A", "B"])
        r = electre_is(p, lam=0.7, veto_mode="simple")
        self.assertAlmostEqual(r["concordance"][0, 1], 0.8)   # A meilleur sur x
        self.assertFalse(r["outranking"][0, 1])               # mais veto sur y (ecart 4)
        self.assertEqual(r["veto_criteria"][("A", "B")], ["y"])

    def test_kernel_of_chain_and_cycle(self):
        crits = [Criterion("x", weight=1.0)]
        p = tiny([[3], [2], [1]], crits, ["A", "B", "C"])
        r = electre_is(p, lam=1.0)
        self.assertEqual(r["kernel"], ["A"])                    # A > B > C
        p2 = tiny([[1], [1]], crits, ["A", "B"])                # A S B et B S A : circuit
        self.assertEqual(electre_is(p2, lam=1.0)["kernel"], ["A", "B"])


class TestSensitivity(unittest.TestCase):
    def test_redistribution(self):
        w = redistribute([0.5, 0.3, 0.2], 0, 0.2)
        np.testing.assert_allclose(w, [0.2, 0.48, 0.32])

    def test_switch_point_analytic(self):
        # A=(1,0), B=(0,1) : A devant si w1 > 0,5
        p = tiny([[1, 0], [0, 1]], [Criterion("x", weight=0.7), Criterion("y", weight=0.3)],
                 ["A", "B"])
        pts = weight_switch_points(p, SCORERS["weighted_sum"], 0)
        self.assertEqual(len(pts), 1)
        self.assertAlmostEqual(pts[0]["threshold"], 0.5, places=4)

    def test_dominant_alternative_always_first(self):
        p = tiny([[5, 5, 5], [1, 2, 3]], [Criterion(c, weight=1 / 3) for c in "xyz"])
        for mode in ("dirichlet", "perturb"):
            r = monte_carlo(p, SCORERS["promethee"], n=500, mode=mode, seed=1)
            self.assertEqual(r["first_place_share"]["a0"], 1.0)
        self.assertEqual(performance_perturbation(p, SCORERS["weighted_sum"], 1, (1, 5))["changes"], [])


class TestCas6Regression(unittest.TestCase):
    """Les chiffres des deux presentations du groupe (cas 6) doivent etre retrouves."""

    @classmethod
    def setUpClass(cls):
        here = os.path.dirname(os.path.abspath(__file__))
        cls.p = load_problem(os.path.join(here, "..", "examples", "cas6_mig.json"))
        cls.a = ahp(cls.p.extra["ahp"]["pairwise"])

    def test_part1_weighted_sum(self):
        s = weighted_sum(self.p)["scores"]
        np.testing.assert_allclose(s, [3.03125, 3.53125, 3.625, 3.0])

    def test_part2_ahp(self):
        np.testing.assert_allclose(self.a["weights"], [0.2147, 0.3760, 0.2147, 0.0738, 0.1209],
                                   atol=5e-5)
        self.assertAlmostEqual(self.a["CR"], 0.0074, places=4)

    def test_part2_promethee(self):
        r = promethee(self.p, self.a["weights"])
        np.testing.assert_allclose(r["phi"], [-0.094, 0.200, 0.015, -0.121], atol=5e-4)
        self.assertAlmostEqual(r["pi"][1, 2], 0.376, places=3)

    def test_part2_electre(self):
        r = electre_is(self.p, self.a["weights"], lam=0.6)
        self.assertEqual(r["never_outranked"], ["NorthBridge"])

    def test_switch_points(self):
        ws = weight_switch_points(self.p, SCORERS["weighted_sum"], 1)
        self.assertAlmostEqual(ws[0]["threshold"], 0.2727, places=3)
        pr = weight_switch_points(self.p, SCORERS["promethee"], 1, self.a["weights"])
        self.assertAlmostEqual(pr[0]["threshold"], 0.2845, places=3)


if __name__ == "__main__":
    unittest.main()
