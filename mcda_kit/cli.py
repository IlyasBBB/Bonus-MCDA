"""Ligne de commande : python -m mcda_kit probleme.json [options]

Exemples
  python -m mcda_kit examples/cas6_mig.json
  python -m mcda_kit examples/cas6_mig.json --weights ahp --methods ws promethee electre
  python -m mcda_kit examples/cas6_mig.json --weights ahp --sensitivity --json resultats.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .methods import ahp, electre_is, promethee, weighted_sum
from .model import load_problem
from .sensitivity import SCORERS, monte_carlo, performance_perturbation, switch_table


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {(" > ".join(k) if isinstance(k, tuple) else str(k)): _jsonable(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "-" * len(line)]
    out += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def run(path: str, methods, weights_mode: str = "given", lam: float | None = None,
        sensitivity: bool = False, seed: int = 0, draws: int = 10000) -> dict:
    prob = load_problem(path)
    report = {"problem": prob.name, "alternatives": prob.alternatives,
              "criteria": [c.name for c in prob.criteria]}
    prob, eliminated = prob.admissible()
    report["eliminated"] = eliminated
    if weights_mode == "ahp":
        if "ahp" not in prob.extra:
            raise SystemExit("aucune matrice 'ahp' dans le fichier")
        a = ahp(prob.extra["ahp"]["pairwise"])
        prob = prob.with_weights(a["weights"])
        report["ahp"] = {k: a[k] for k in ("weights", "lambda_max", "CI", "RI", "CR", "consistent")}
    elif weights_mode == "equal":
        prob = prob.with_weights(np.full(prob.n, 1 / prob.n))
    report["warnings"] = prob.validate()
    report["weights"] = dict(zip(report["criteria"], prob.weights()))
    if "ws" in methods:
        r = weighted_sum(prob)
        report["weighted_sum"] = {"scores": dict(zip(prob.alternatives, r["scores"])),
                                  "ranking": r["ranking"], "values": r["values"]}
    if "promethee" in methods:
        r = promethee(prob)
        report["promethee"] = {k: dict(zip(prob.alternatives, r[k]))
                               for k in ("phi_plus", "phi_minus", "phi")}
        report["promethee"]["ranking"] = r["ranking"]
        report["promethee"]["pi"] = r["pi"]
    if "electre" in methods:
        cfg = prob.extra.get("electre", {})
        lam_ = lam if lam is not None else cfg.get("lambda", 0.6)
        r = electre_is(prob, lam=lam_, veto_mode=cfg.get("veto_mode", "electre_is"))
        report["electre_is"] = {"lambda": lam_, "concordance": r["concordance"],
                                "arcs": r["arcs"], "kernel": r["kernel"],
                                "never_outranked": r["never_outranked"],
                                "vetoes": r["veto_criteria"]}
    if sensitivity:
        sens = {}
        for key, name in (("ws", "weighted_sum"), ("promethee", "promethee")):
            if key not in methods:
                continue
            fn = SCORERS[name]
            sens[name] = {
                "switch_points": switch_table(prob, fn),
                "monte_carlo_perturb_30pct": monte_carlo(prob, fn, n=draws, mode="perturb",
                                                         rel=0.3, seed=seed),
                "monte_carlo_dirichlet": monte_carlo(prob, fn, n=draws, mode="dirichlet",
                                                     seed=seed),
            }
            cfg = prob.extra.get("sensitivity", {})
            if "performance_bounds" in cfg:
                sens[name]["performance_perturbation"] = performance_perturbation(
                    prob, fn, cfg.get("performance_delta", 1), cfg["performance_bounds"],
                    cfg.get("performance_criteria"))
        report["sensitivity"] = sens
    return report


def print_report(rep: dict) -> None:
    print(f"\n{rep['problem']}")
    for e in rep["eliminated"]:
        print("  [contrainte]", e)
    for w in rep["warnings"]:
        print("  [info]", w)
    if "ahp" in rep:
        a = rep["ahp"]
        print(f"\nAHP : lambda_max = {a['lambda_max']:.4f}, CR = {a['CR']:.4f} "
              f"({'coherent' if a['consistent'] else 'INCOHERENT'})")
    print("\nPoids : " + ", ".join(f"{k} {v:.3f}" for k, v in rep["weights"].items()))
    if "weighted_sum" in rep:
        print("\nSomme ponderee")
        print(_table([(r, a, f"{s:.4f}") for r, a, s in rep["weighted_sum"]["ranking"]],
                     ["rang", "alternative", "score"]))
    if "promethee" in rep:
        p = rep["promethee"]
        print("\nPROMETHEE II")
        print(_table([(r, a, f"{p['phi_plus'][a]:.3f}", f"{p['phi_minus'][a]:.3f}", f"{s:+.3f}")
                      for r, a, s in p["ranking"]], ["rang", "alternative", "phi+", "phi-", "phi"]))
    if "electre_is" in rep:
        e = rep["electre_is"]
        print(f"\nELECTRE IS (lambda = {e['lambda']})")
        print("  surclassements :", ", ".join(f"{a} S {b}" for a, b in e["arcs"]) or "aucun")
        print("  vetos          :", ", ".join(f"{a} S {b} bloque par {'/'.join(c)}"
                                            for (a, b), c in e["vetoes"].items()) or "aucun")
        print("  noyau          :", ", ".join(e["kernel"]))
        print("  jamais surclasse :", ", ".join(e["never_outranked"]) or "aucun")
    for name, s in rep.get("sensitivity", {}).items():
        print(f"\nSensibilite ({name}) : un poids varie, les autres gardent leurs proportions")
        rows = []
        for r in s["switch_points"]:
            for side in ("switch_below", "switch_above"):
                sp = r[side]
                if sp:
                    rows.append((r["criterion"], f"{r['reference']:.3f}",
                                 ("<" if side == "switch_below" else ">") + f" {sp['threshold']:.4f}",
                                 sp["new_first"]))
        print(_table(rows, ["critere", "poids ref", "bascule si", "nouveau 1er"]) if rows
              else "  aucune bascule entre 0 et 99 %")
        for key in ("monte_carlo_perturb_30pct", "monte_carlo_dirichlet"):
            mc = s[key]
            print(f"  {key} ({mc['draws']} tirages) : " +
                  ", ".join(f"{a} {100 * v:.1f} %" for a, v in mc["first_place_share"].items()))
        if "performance_perturbation" in s:
            pp = s["performance_perturbation"]
            print(f"  perturbation d'une performance : {len(pp['changes'])} cas sur {pp['tested']} "
                  "changent le premier")
            for c in pp["changes"]:
                print(f"    {c['alternative']} {c['criterion']} {c['from']:g} -> {c['to']:g} : "
                      f"{c['new_first']} premier")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mcda_kit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", help="fichier JSON decrivant le probleme")
    ap.add_argument("--methods", nargs="+", default=["ws", "promethee", "electre"],
                    choices=["ws", "promethee", "electre"])
    ap.add_argument("--weights", choices=["given", "ahp", "equal"], default="given",
                    help="poids du fichier, poids AHP ou poids egaux")
    ap.add_argument("--lambda", dest="lam", type=float, default=None,
                    help="seuil de concordance ELECTRE (sinon valeur du fichier ou 0,6)")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="ecrit tous les resultats dans ce fichier")
    args = ap.parse_args(argv)
    rep = run(args.problem, args.methods, args.weights, args.lam, args.sensitivity,
              args.seed, args.draws)
    print_report(rep)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(rep), fh, ensure_ascii=False, indent=2)
        print(f"\nResultats ecrits dans {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
