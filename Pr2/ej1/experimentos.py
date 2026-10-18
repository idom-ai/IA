#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimento: coloreado distribuido del mapa de España con ABT y AWC (2, 3 y 4 colores)."""

from __future__ import annotations

import random

from problema import ColoringProblem, solve_centralized, spain_problem
from simulador import random_initial_values, run_distributed
from utilidades import banner, check, print_table
from visualizacion import COLOR_NAME, Output, plot_nogood_proof, plot_run, save_animation


def experiment_spain(out: Output, rng: random.Random, make_gif: bool) -> None:
    """Experimento: colorear el mapa de España con 15 agentes (caso visual principal).

    Se resuelve con AWC (prioridades dinámicas) y con ABT, partiendo de los mismos colores
    iniciales aleatorios, con 3, 4 y 2 colores. Con 3 y 4 colores encuentran un coloreado válido; con
    2 colores no hay solución (Cantabria, País Vasco y Castilla y León son vecinas dos a dos) y los
    agentes lo demuestran. En todos los casos el resultado se contrasta con un solver centralizado.

    Args:
        out: Gestor de salida.
        rng: Generador de números aleatorios.
        make_gif: Si es ``True``, guarda también un GIF.
    """
    banner("EXPERIMENTO · Colorear el mapa de España: 15 agentes, uno por comunidad autónoma")
    for k in (3, 4, 2):
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
            print("   Motivo: Cantabria, País Vasco y Castilla y León son vecinas dos a dos (un triángulo),")
            print("   y tres regiones vecinas entre sí no pueden pintarse con solo 2 colores.")
            explain_triangle(out)
        print()



def explain_triangle(out: Output) -> None:
    """Muestra, paso a paso, cómo ABT demuestra que el triángulo Can-PV-CyL no admite 2 colores.

    Ejecuta ABT sobre el triángulo (prioridad Can > PV > CyL), imprime la cadena de nogoods que se
    envían los agentes hasta llegar al nogood vacío y dibuja la figura correspondiente.

    Args:
        out: Gestor de salida.
    """
    triangle = ColoringProblem("triángulo", 2, [(0, 1), (0, 2), (1, 2)], ["Can", "PV", "CyL"],
                               positions=[(-1.0, 0.8), (1.0, 0.8), (0.0, -0.9)])
    res = run_distributed(triangle, "ABT", [0, 1, 0], seed=1, record_trace=True)
    print()
    print("   Ejemplo mínimo (solo el triángulo, ABT): cadena de nogoods hasta la demostración")
    step = 1
    seen = set()
    for c, msgs in enumerate(res.log[:res.cycles]):
        for dest, msg in msgs:
            if msg[0] == "nogood":
                ng = ", ".join(f"{triangle.labels[a]}={COLOR_NAME[v]}" for a, v in sorted(msg[2]))
                if msg[2] in seen:
                    print(f"        (ciclo {c}: {triangle.labels[msg[1]]} repite el nogood {{{ng}}}; no aporta nada nuevo)")
                    continue
                seen.add(msg[2])
                print(f"     {step}. ciclo {c}: {triangle.labels[msg[1]]} se bloquea y envía a "
                      f"{triangle.labels[dest]} el nogood {{{ng}}}")
                step += 1
    print(f"     {step}. ciclo {res.cycles}: {triangle.labels[res.failed_agent]} tiene prohibidos los dos colores "
          f"sin depender de nadie -> nogood vacío {{ }} -> NO hay solución")
    check(res.status == "sin_solucion", "ABT demuestra que el triángulo no se puede pintar con 2 colores.",
          "ABT no llegó al nogood vacío")
    plot_nogood_proof(triangle, res, out, "exp1_triangulo_2colores",
                      full_names=["Cantabria", "País Vasco", "Castilla y León"])
