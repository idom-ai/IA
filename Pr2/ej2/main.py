#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ejercicio 2 - Comunicación entre agentes: promediado distribuido (gossip).

Referencia: S. Boyd, A. Ghosh, B. Prabhakar y D. Shah, "Randomized Gossip
Algorithms", IEEE Transactions on Information Theory, 52(6), 2508-2530, 2006.
DOI: 10.1109/TIT.2006.874516

Experimento: 20 sensores miden la misma temperatura con ruido. Sin ningún nodo
central, intercambiando mensajes solo con sus vecinos, todos acaban con el
mismo valor: el promedio de las lecturas. Se compara en tres redes (anillo,
geométrica aleatoria y completa) cuánto tardan en ponerse de acuerdo, medido con
K = tiempo de epsilon-promediado (Definición 1 del paper).

Uso:  python main.py     (necesita numpy y matplotlib; tarda unos segundos)
"""

import matplotlib.pyplot as plt
import numpy as np

from simulation import averaging_time, run_gossip
from topology import complete, geometric, ring

N_AGENTS = 20        # número de sensores
TRUE_VALUE = 20.0    # temperatura real (ºC)
NOISE_STD = 2.0      # desviación típica del ruido de cada sensor (ºC)
EPSILON = 0.1        # K: al menos el 90 % de las ejecuciones con error < 10 % del inicial
N_RUNS = 100         # ejecuciones para medir K
DEMO_TOL = 0.01      # la ejecución de demostración se dibuja hasta error < 1 % del inicial
SEED = 1


def draw_network(ax, pos, neighbors, values, vmin, vmax):
    """Dibuja una red con cada agente coloreado según su valor.

    Args:
        ax: Eje de matplotlib.
        pos: Posiciones (n, 2) de los agentes.
        neighbors: Lista de vecinos de cada agente.
        values: Valor de cada agente.
        vmin: Valor mínimo de la escala de color.
        vmax: Valor máximo de la escala de color.

    Returns:
        El objeto de dispersión (para la barra de color).
    """
    for i, vecinos in enumerate(neighbors):
        for j in vecinos:
            if i < j:
                ax.plot(*zip(pos[i], pos[j]), color="gray", lw=0.7, alpha=0.5, zorder=1)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=values, cmap="coolwarm", vmin=vmin, vmax=vmax,
                    s=170, edgecolors="k", zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    return sc


def main() -> None:
    """Ejecuta el experimento en las tres redes, imprime K y dibuja las figuras."""
    rng = np.random.default_rng(SEED)
    networks = {"Anillo": ring(N_AGENTS), "Geométrica": geometric(N_AGENTS, rng),
                "Completa": complete(N_AGENTS)}
    readings = TRUE_VALUE + NOISE_STD * rng.standard_normal(N_AGENTS)  # mismas para las 3 redes
    average = readings.mean()
    vmin, vmax = readings.min(), readings.max()

    print(f"{N_AGENTS} agentes. Lecturas de {vmin:.2f} a {vmax:.2f} ºC; promedio {average:.3f} ºC")
    print(f"K con epsilon = {EPSILON} ({N_RUNS} ejecuciones por red)\n")
    print(f"{'Red':<12}{'Enlaces':>9}{'K (ticks)':>12}{'Mensajes (2K)':>16}")

    results, demos = {}, {}
    for name, (pos, neighbors) in networks.items():
        k, _ = averaging_time(neighbors, N_RUNS, EPSILON, TRUE_VALUE, NOISE_STD, rng)
        history, _ = run_gossip(neighbors, readings, rng, DEMO_TOL)
        results[name], demos[name] = k, history
        print(f"{name:<12}{sum(map(len, neighbors)) // 2:>9}{k:>12}{2 * k:>16}")

    # ---- Figura 1: antes (red) y durante (convergencia) en cada topología ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for col, (name, (pos, neighbors)) in enumerate(networks.items()):
        sc = draw_network(axes[0, col], pos, neighbors, readings, vmin, vmax)
        axes[0, col].set_title(f"{name}  ({sum(map(len, neighbors)) // 2} enlaces)")
        axes[1, col].plot(demos[name], lw=1, alpha=0.7)
        axes[1, col].axhline(average, color="k", ls="--", lw=1.5)
        axes[1, col].set_ylim(vmin - 0.3, vmax + 0.3)
        axes[1, col].set_xlabel("Ticks (intercambios entre dos agentes)")
        axes[1, col].grid(alpha=0.3)
    axes[1, 0].set_ylabel("Valor de cada agente (ºC)")
    fig.colorbar(sc, ax=axes[0, :], shrink=0.8, label="Lectura inicial (ºC)")
    fig.suptitle(f"Todos los agentes convergen al promedio ({average:.2f} ºC, línea discontinua)",
                 fontsize=13, weight="bold")
    fig.savefig("convergencia.png", dpi=140)

    # ---- Figura 2: K en cada topología ----
    fig2, ax = plt.subplots(figsize=(7, 4.6))
    names = list(results)
    bars = ax.bar(names, [results[n] for n in names], color=["#d95f02", "#7570b3", "#1b9e77"])
    for bar, name in zip(bars, names):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"K = {results[name]}\n({2 * results[name]} mensajes)", ha="center", va="bottom")
    ax.set_ylabel("K: ticks hasta error < 10 % en el 90 % de ejecuciones")
    ax.set_title(f"Tiempo de ε-promediado K (ε = {EPSILON}, {N_AGENTS} agentes)")
    ax.set_ylim(0, max(results.values()) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("K.png", dpi=140)
    plt.show()


if __name__ == "__main__":
    main()
