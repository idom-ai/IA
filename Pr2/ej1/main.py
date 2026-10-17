#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Práctica 2 · Ejercicio 1 — Resolución de conflictos entre agentes: coloreado distribuido de un mapa.

Problema (se explica en 5 segundos): cada comunidad autónoma del mapa de España es un AGENTE que
debe elegir un color. Dos comunidades vecinas NO pueden elegir el mismo color (conflicto). Cada
agente solo conoce su propio color y habla únicamente con sus vecinos; nadie tiene una visión
global. Es el "Distributed Graph Coloring", un caso clásico de **Distributed Constraint
Satisfaction Problem (DisCSP)**. Los agentes no usan LLMs: son agentes clásicos que intercambian
mensajes y negocian para resolver sus conflictos.

Algoritmos (ambos de la referencia):
    - ABT  (Asynchronous Backtracking): los agentes tienen una jerarquía FIJA de prioridades. Cuando
      un agente no tiene ningún color posible, aprende un "nogood" (combinación prohibida de
      colores de sus superiores) y se lo envía al agente de menor prioridad implicado.
    - AWC  (Asynchronous Weak-Commitment search): igual, pero la jerarquía es DINÁMICA. Cuando un
      agente se bloquea, SUBE su prioridad y obliga a los demás a adaptarse a él, en lugar de
      explorar exhaustivamente. Es la mejora central del artículo.

Referencia (formalización y algoritmos salen de ella):
    Yokoo, M., Durfee, E. H., Ishida, T. y Kuwabara, K. (1998). "The Distributed Constraint
    Satisfaction Problem: Formalization and Algorithms". IEEE Transactions on Knowledge and Data
    Engineering, 10(5), 673-685.

Correspondencia con el artículo:
    - Definición de CSP distribuido (agentes, variables, restricciones, mensajes) ... Sec. 2
    - Asynchronous Backtracking (ok?, nogood, agent_view, prioridades fijas) ......... Sec. 4
    - Asynchronous Weak-Commitment (prioridades dinámicas, heurística min-conflict) .. Sec. 5
    - Modelo de simulación por ciclos y límite de 1000 ciclos ....................... Sec. 6

Implementación: propia, escrita a partir de la descripción de los procedimientos del artículo. Solo
requiere ``numpy`` y ``matplotlib``. Nota: el pseudocódigo del artículo está en figuras (Fig. 2 y 5);
la implementación sigue su descripción textual y los ejemplos de ejecución (Fig. 3, 4 y 6).

Uso:
    python main.py                  # ejecución completa (guarda PNG/GIF y abre ventanas)
    python main.py --sin-pantalla   # no abre ventanas (solo guarda ficheros)
    python main.py --rapido         # versión reducida de los experimentos
    python main.py --help           # todas las opciones

Dependencias: Python >= 3.9, numpy, matplotlib (``pip install numpy matplotlib``).

Estructura del proyecto (todos los ficheros en la misma carpeta):
    main.py            punto de entrada: lee los argumentos y ejecuta los experimentos
    problema.py        modelo del problema, solver centralizado e instancia del mapa
    agentes.py         agentes y algoritmos ABT / AWC
    simulador.py       simulación por ciclos de los agentes
    visualizacion.py   figuras y animaciones
    utilidades.py      impresión por consola (encabezados, tablas, comprobaciones)
    experimentos.py    los dos experimentos
"""

from __future__ import annotations

import argparse
import os
import random
import time

import matplotlib.pyplot as plt

from experimentos import experiment_spain, experiment_validation
from utilidades import banner
from visualizacion import Output


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de órdenes.

    Returns:
        Espacio de nombres con las opciones elegidas.
    """
    parser = argparse.ArgumentParser(description="Práctica 2 · Ej. 1: coloreado distribuido (ABT/AWC).")
    parser.add_argument("--sin-pantalla", action="store_true", help="no abrir ventanas con las figuras (solo guardarlas)")
    parser.add_argument("--rapido", action="store_true", help="versión reducida de los experimentos")
    parser.add_argument("--sin-gif", action="store_true", help="no generar las animaciones GIF")
    parser.add_argument("--semilla", type=int, default=7, help="semilla aleatoria (por defecto 7)")
    parser.add_argument("--salida", default="resultados_ej1", help="carpeta de salida (por defecto resultados_ej1)")
    return parser.parse_args()


def main() -> None:
    """Ejecuta los dos experimentos, guarda las figuras y muestra un resumen final."""
    args = parse_args()
    if args.sin_pantalla:
        plt.switch_backend("Agg")
    plt.rcParams.update({"figure.max_open_warning": 0, "font.size": 9})
    rng = random.Random(args.semilla)
    out = Output(args.salida)
    t0 = time.perf_counter()

    experiment_spain(out, rng, make_gif=not args.sin_gif)
    experiment_validation(out, rng, n_problems=300 if not args.rapido else 60)

    banner("FIN")
    print(f"Tiempo total: {time.perf_counter() - t0:.1f} s. Resultados guardados en: {os.path.abspath(out.directory)}")
    if not args.sin_pantalla:
        print("Se abrirán las figuras; ciérralas para terminar el programa.")
        plt.show()


if __name__ == "__main__":
    main()
