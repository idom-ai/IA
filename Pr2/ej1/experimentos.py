#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimentos: mapa de España y validación frente a un solver centralizado."""

from __future__ import annotations

import random

from problema import ColoringProblem, solve_centralized, spain_problem
from simulador import random_initial_values, run_distributed
from utilidades import banner, check, print_table
from visualizacion import COLOR_NAME, Output, plot_run, save_animation


def experiment_spain(out: Output, rng: random.Random, make_gif: bool) -> None:
    """Experimento 1: colorear el mapa de España con 15 agentes (caso visual principal).

    Se resuelve con AWC (prioridades dinámicas) y con ABT, partiendo de los mismos colores
    iniciales aleatorios, y se comprueba que la solución final es válida.

    Args:
        out: Gestor de salida.
        rng: Generador de números aleatorios.
        make_gif: Si es ``True``, guarda también un GIF.
    """
    banner("EXPERIMENTO 1 · Colorear el mapa de España: 15 agentes, uno por comunidad autónoma")
    for k in (3, 4):
        problem = spain_problem(k)
        solvable = solve_centralized(problem) is not None
        print(f"Con {k} colores el problema {'SÍ' if solvable else 'NO'} tiene solución "
              f"(comprobado con un solver centralizado). Agentes: {problem.n}, fronteras (restricciones): {len(problem.edges)}")
        initial = random_initial_values(problem, rng)
        rows = []
        runs = {}
        for algo in ("ABT", "AWC"):
            res = run_distributed(problem, algo, initial, seed=1, record_trace=True)
            runs[algo] = res
            rows.append([algo, res.status, res.cycles, res.messages, problem.n_violations(res.values)])
        print_table(["algoritmo", "resultado", "ciclos", "mensajes", "conflictos finales"], rows)
        check(all((r.status == "resuelto") == solvable for r in runs.values()),
              "el resultado de los agentes coincide con el del solver centralizado.",
              "los agentes discrepan del solver centralizado")
        if solvable:
            res = runs["AWC"]
            check(problem.n_violations(res.values) == 0, "AWC: ningún par de vecinos comparte color.", "AWC dejó conflictos")
            print("   Coloreado final (AWC): " + ", ".join(
                f"{problem.labels[i]}={COLOR_NAME[c]}" for i, c in enumerate(res.values)))
            plot_run(problem, res, out, f"exp1_espana_{k}colores",
                     f"AWC sobre el mapa de España con {k} colores ({res.cycles} ciclos, {res.messages} mensajes)")
            if make_gif:
                gif = out.path(f"exp1_espana_{k}colores.gif")
                if save_animation(problem, res, gif):
                    print(f"   [animación] guardada en {gif}")
        else:
            print("   ABT/AWC deducen que no hay solución al generar el nogood vacío (completitud del algoritmo).")
        print()


def experiment_validation(out: Output, rng: random.Random, n_problems: int) -> None:
    """Experimento 2: validación de ABT y AWC frente a un solver centralizado.

    Sobre grafos pequeños aleatorios (con y sin solución), comprueba que cada algoritmo (a) devuelve
    una solución válida cuando existe y (b) detecta que no existe cuando es así.

    Args:
        out: Gestor de salida.
        rng: Generador de números aleatorios.
        n_problems: Número de problemas aleatorios.
    """
    del out
    banner("EXPERIMENTO 2 · Validación: ABT y AWC frente a un solver centralizado (grafos pequeños)")
    print("Ejemplo mínimo: Cantabria, País Vasco y Castilla y León son vecinas dos a dos (un triángulo).")
    rows = []
    for k in (2, 3):
        triangle = ColoringProblem("triángulo", k, [(0, 1), (0, 2), (1, 2)], ["Can", "PV", "CyL"])
        for algo in ("ABT", "AWC"):
            res = run_distributed(triangle, algo, [0] * 3, seed=1)
            answer = ("solución: " + ", ".join(f"{l}={COLOR_NAME[c]}" for l, c in zip(triangle.labels, res.values))
                      if res.status == "resuelto" else "demuestra que NO hay solución")
            rows.append([k, algo, answer])
    print_table(["colores", "algoritmo", "resultado"], rows)
    print()
    totals = {"ABT": [0, 0, 0], "AWC": [0, 0, 0]}  # aciertos, con solución, sin solución
    mismatches = 0
    n_solvable = n_unsolvable = 0
    for _ in range(n_problems):
        n = rng.randint(5, 9)
        k = rng.choice([2, 3, 3])
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edges = rng.sample(all_pairs, rng.randint(n - 1, min(len(all_pairs), 2 * n + 2)))
        problem = ColoringProblem("aleatorio", k, edges, [f"x{i + 1}" for i in range(n)])
        solvable = solve_centralized(problem) is not None
        n_solvable += solvable
        n_unsolvable += not solvable
        initial = random_initial_values(problem, rng)
        for algo in ("ABT", "AWC"):
            res = run_distributed(problem, algo, initial, seed=rng.randrange(10 ** 6), max_cycles=5000)
            correct = (res.status == "resuelto" and solvable and problem.n_violations(res.values) == 0) or \
                      (res.status == "sin_solucion" and not solvable)
            totals[algo][0] += correct
            totals[algo][1 if solvable else 2] += 1
            mismatches += not correct
    rows = [[a, f"{v[0]}/{n_problems}", v[1], v[2]] for a, v in totals.items()]
    print_table(["algoritmo", "respuestas correctas", "problemas con solución", "problemas sin solución"], rows)
    check(mismatches == 0, f"ABT y AWC coinciden con el solver centralizado en los {n_problems} problemas "
                           f"({n_solvable} con solución y {n_unsolvable} sin ella).",
          f"{mismatches} respuestas incorrectas")
