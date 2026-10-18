#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualización: dibujo del grafo coloreado, fotos de la ejecución, curva de conflictos y GIF."""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from problema import ColoringProblem
from simulador import RunResult

COLOR_HEX = ["#e6550d", "#31a354", "#3182bd", "#9e9ac8", "#bdbdbd"]
COLOR_NAME = ["naranja", "verde", "azul", "morado", "gris"]


class Output:
    """Gestiona la carpeta de resultados y las figuras generadas.

    Attributes:
        directory: Carpeta donde se guardan los ficheros.
        figures: Figuras creadas (se muestran todas al final).
    """

    def __init__(self, directory: str) -> None:
        """Crea la carpeta de salida si no existe.

        Args:
            directory: Ruta de la carpeta de resultados.
        """
        self.directory = directory
        self.figures: List[plt.Figure] = []
        os.makedirs(directory, exist_ok=True)

    def path(self, filename: str) -> str:
        """Ruta completa de un fichero dentro de la carpeta de resultados.

        Args:
            filename: Nombre del fichero.

        Returns:
            Ruta completa.
        """
        return os.path.join(self.directory, filename)

    def save(self, fig: plt.Figure, name: str) -> None:
        """Guarda la figura como PNG y la registra para mostrarla al final.

        Args:
            fig: Figura de matplotlib.
            name: Nombre del fichero sin extensión.
        """
        fig.savefig(self.path(f"{name}.png"), dpi=150, bbox_inches="tight")
        self.figures.append(fig)
        print(f"   [figura] guardada en {self.path(name + '.png')}")


def layout_positions(problem: ColoringProblem) -> List[Tuple[float, float]]:
    """Posiciones para dibujar el grafo (las del problema o, si no hay, en círculo).

    Args:
        problem: Problema de coloreado.

    Returns:
        Lista de coordenadas (x, y), una por agente.
    """
    if problem.positions is not None:
        return problem.positions
    return [(math.cos(2 * math.pi * i / problem.n), math.sin(2 * math.pi * i / problem.n)) for i in range(problem.n)]


def draw_coloring(ax: plt.Axes, problem: ColoringProblem, values: Sequence[int],
                  priorities: Optional[Sequence[int]] = None, title: str = "",
                  node_size: float = 0.42) -> None:
    """Dibuja el grafo con cada agente del color que ha elegido; los conflictos salen en rojo.

    Args:
        ax: Ejes de matplotlib.
        problem: Problema de coloreado.
        values: Color de cada agente.
        priorities: Prioridad de cada agente (AWC); si no es 0 se escribe bajo el nombre.
        title: Título del gráfico.
        node_size: Radio de los nodos en unidades de datos.
    """
    pos = layout_positions(problem)
    bad = set(problem.violated_edges(values))
    for i, j in problem.edges:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                color="#d62728" if (i, j) in bad else "#9a9a9a",
                lw=3.2 if (i, j) in bad else 1.3, zorder=1 if (i, j) not in bad else 2)
    for i, (x, y) in enumerate(pos):
        ax.add_patch(plt.Circle((x, y), node_size, facecolor=COLOR_HEX[values[i] % len(COLOR_HEX)],
                                edgecolor="black", lw=1.0, zorder=3))
        label = problem.labels[i]
        if priorities is not None and priorities[i] > 0:
            label += f"\np={priorities[i]}"
        ax.text(x, y, label, ha="center", va="center", fontsize=7, fontweight="bold", color="white", zorder=4)
    xs, ys = [p[0] for p in pos], [p[1] for p in pos]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def choose_snapshot_cycles(trace_len: int, n_panels: int = 5) -> List[int]:
    """Elige instantes repartidos de la ejecución para las "fotos".

    Args:
        trace_len: Número de elementos de la traza (ciclo 0 incluido).
        n_panels: Número máximo de fotos.

    Returns:
        Índices de ciclo, ordenados y sin repetir (siempre incluye el primero y el último).
    """
    return sorted(set(np.linspace(0, trace_len - 1, min(n_panels, trace_len)).round().astype(int).tolist()))


def plot_run(problem: ColoringProblem, result: RunResult, out: Output, name: str, title: str) -> None:
    """Figura de una ejecución: fotos del mapa en varios ciclos y curva de conflictos.

    Args:
        problem: Problema de coloreado.
        result: Resultado con traza (``record_trace=True``).
        out: Gestor de salida.
        name: Nombre del fichero (sin extensión).
        title: Título general.
    """
    cycles = choose_snapshot_cycles(len(result.trace), 5)
    fig = plt.figure(figsize=(3.6 * len(cycles), 7.6))
    gs = GridSpec(2, len(cycles), height_ratios=[1.35, 1])
    for col, c in enumerate(cycles):
        ax = fig.add_subplot(gs[0, col])
        vals, prios, viol, _ = result.trace[c]
        draw_coloring(ax, problem, vals, prios, f"ciclo {c}  ·  {viol} conflictos")
    ax = fig.add_subplot(gs[1, :])
    viol = [t[2] for t in result.trace]
    ax.plot(range(len(viol)), viol, "o-", color="#d62728", label="aristas en conflicto")
    ax2 = ax.twinx()
    ax2.bar(range(len(viol)), [t[3] for t in result.trace], color="#9ecae1", alpha=0.6, label="mensajes enviados")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel("ciclo de simulación")
    ax.set_ylabel("conflictos")
    ax2.set_ylabel("mensajes por ciclo")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out.save(fig, name)


def save_animation(problem: ColoringProblem, result: RunResult, filename: str, fps: float = 2.0) -> bool:
    """Guarda un GIF con la evolución de los colores ciclo a ciclo.

    Args:
        problem: Problema de coloreado.
        result: Resultado con traza.
        filename: Ruta del GIF de salida.
        fps: Ciclos mostrados por segundo.

    Returns:
        ``True`` si se guardó, ``False`` si falló (p. ej. falta Pillow).
    """
    try:
        fig, ax = plt.subplots(figsize=(7, 6))

        def update(c: int) -> None:
            ax.clear()
            vals, prios, viol, _ = result.trace[c]
            draw_coloring(ax, problem, vals, prios, f"{result.algorithm} · ciclo {c} · {viol} conflictos", 0.45)

        anim = FuncAnimation(fig, update, frames=len(result.trace), interval=1000 / fps)
        anim.save(filename, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return True
    except Exception as exc:  # noqa: BLE001 - la animación es opcional
        print(f"   [aviso] no se pudo guardar la animación: {exc}")
        return False


def _nogood_text(nogood: Sequence[Tuple[int, int]], problem: ColoringProblem) -> str:
    """Traduce un nogood a una frase corta en castellano.

    Args:
        nogood: Pares (agente, color) que no pueden darse a la vez.
        problem: Problema (para los nombres de los agentes).

    Returns:
        Frase con el significado del nogood.
    """
    pairs = sorted(nogood)
    if not pairs:
        return "{ }  (nogood vacío)"
    if len(pairs) == 1:
        a, v = pairs[0]
        return f"«{problem.labels[a]} no puede ser {COLOR_NAME[v]}»"
    parts = " y ".join(f"{problem.labels[a]}={COLOR_NAME[v]}" for a, v in pairs)
    return f"«{parts}\nno pueden darse a la vez»"


def plot_nogood_proof(problem: ColoringProblem, result: RunResult, out: Output, name: str,
                      full_names: Optional[Sequence[str]] = None) -> None:
    """Figura paso a paso de cómo los agentes demuestran que no hay solución.

    Dibuja un panel por ciclo de la ejecución: el color de cada agente al final del ciclo, los
    mensajes enviados en ese ciclo (flechas grises ``ok?`` y flechas rojas ``nogood``) y, debajo,
    qué combinación prohíbe cada nogood. El último panel marca al agente que deduce el nogood vacío.

    Args:
        problem: Problema pequeño sin solución (p. ej. un triángulo con 2 colores).
        result: Resultado con traza y registro de mensajes (``record_trace=True``).
        out: Gestor de salida.
        name: Nombre del fichero (sin extensión).
        full_names: Nombres largos de los agentes para el pie de figura (opcional).
    """
    from matplotlib.patches import FancyArrowPatch

    pos = layout_positions(problem)
    n_panels = len(result.trace)
    n_cols = min(4, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.9 * n_rows + 0.8))
    axes = np.atleast_1d(axes).ravel()
    radius = 0.32
    last = n_panels - 1
    seen = set()
    for c, ax in enumerate(axes):
        ax.axis("off")
        if c >= n_panels:
            continue
        vals = result.trace[c][0]
        final = c == last and result.failed_agent is not None
        for i, j in problem.edges:
            same = vals[i] == vals[j]
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="#222222" if same else "#9a9a9a",
                    lw=3 if same else 1.4, linestyle=":" if same else "-", zorder=1)
        for i, (x, y) in enumerate(pos):
            failed = final and i == result.failed_agent
            ax.add_patch(plt.Circle((x, y), radius, facecolor=COLOR_HEX[vals[i]],
                                    edgecolor="#d62728" if failed else "black", lw=4 if failed else 1, zorder=3))
            ax.text(x, y, problem.labels[i], ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white", zorder=4)
        lines = []
        if not final:
            for dest, msg in result.log[c]:
                src = msg[1]
                is_nogood = msg[0] == "nogood"
                repeated = is_nogood and msg[2] in seen
                arrow = FancyArrowPatch(pos[src], pos[dest], connectionstyle="arc3,rad=0.25",
                                        arrowstyle="-|>", mutation_scale=14, shrinkA=22, shrinkB=22,
                                        color=("#f4a6a6" if repeated else "#d62728") if is_nogood else "#7f7f7f",
                                        lw=2.2 if is_nogood else 1.1, linestyle="--" if repeated else "-", zorder=2)
                ax.add_patch(arrow)
                if repeated:
                    lines.append(f"({problem.labels[src]} repite un nogood ya enviado)")
                elif is_nogood:
                    seen.add(msg[2])
                    lines.append(f"{problem.labels[src]} se bloquea y avisa a {problem.labels[dest]}:\n"
                                 f"{_nogood_text(msg[2], problem)}")
            if not lines:
                lines.append("inicio: cada agente avisa de su color (ok?)" if c == 0
                             else "los agentes cambian de color y se avisan (ok?)")
            caption, box = "\n".join(lines), "#fdecea" if any("bloquea" in ln for ln in lines) else "#f2f2f2"
        else:
            ag = problem.labels[result.failed_agent]
            caption = (f"{ag} tiene prohibidos TODOS sus colores\nsin depender de nadie ⇒ nogood vacío {{ }}\n"
                       f"⇒ QUEDA DEMOSTRADO QUE NO HAY SOLUCIÓN")
            box = "#d62728"
        ax.text(0.5, -0.02, caption, transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                color="white" if final else "black", fontweight="bold" if final else "normal",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=box, edgecolor="none"))
        xs, ys = [p[0] for p in pos], [p[1] for p in pos]
        ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
        ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
        ax.set_aspect("equal")
        ax.set_title(f"ciclo {c}", fontsize=11, fontweight="bold")
    legend = ("flecha gris = mensaje ok? (mi color)   ·   flecha roja = mensaje nogood (combinación prohibida)"
              "   ·   roja discontinua = nogood repetido   ·   línea negra punteada = vecinas del mismo color")
    if full_names:
        legend += "\n" + "   ·   ".join(f"{s} = {f}" for s, f in zip(problem.labels, full_names))
    fig.suptitle(f"{result.algorithm} con {problem.n_colors} colores: cómo demuestran los agentes que NO hay "
                 f"solución\n(colores al final de cada ciclo; prioridad "
                 f"{' > '.join(problem.labels)})", fontsize=13)
    fig.text(0.5, 0.01, legend, ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.subplots_adjust(hspace=0.55)
    out.save(fig, name)
