#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modelo del problema: coloreado distribuido de un grafo (Sec. 2 del artículo de Yokoo et al., 1998).

Contiene la definición formal del CSP distribuido de coloreado (un agente por nodo, una restricción
"color distinto" por arista), un solver centralizado (solo para validar a los agentes) y los
instancia del mapa de España.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ColoringProblem:
    """CSP distribuido de coloreado: un agente por nodo, una restricción "distinto color" por arista.

    Es el caso de la Sec. 2.2 del artículo con las tres simplificaciones habituales: cada agente
    tiene exactamente una variable (su color), todas las restricciones son binarias y cada agente
    conoce las restricciones que afectan a su variable.

    Attributes:
        name: Nombre del problema.
        n_colors: Número de colores disponibles (tamaño del dominio de cada agente).
        edges: Aristas (i, j) con i < j; cada una es una restricción ``color_i != color_j``.
        labels: Nombre de cada agente (el índice es su identificador y su prioridad inicial:
            menor índice = mayor prioridad).
        positions: Coordenadas (x, y) de cada agente para dibujar (opcional).
        neighbors: Vecinos de cada agente (se calcula automáticamente).
    """

    name: str
    n_colors: int
    edges: List[Tuple[int, int]]
    labels: List[str]
    positions: Optional[List[Tuple[float, float]]] = None
    neighbors: List[List[int]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Normaliza las aristas y calcula los vecinos.

        Raises:
            ValueError: Si hay aristas con índices fuera de rango o bucles (i, i).
        """
        n = len(self.labels)
        clean = set()
        for i, j in self.edges:
            if i == j or not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"Arista no válida: {(i, j)}")
            clean.add((min(i, j), max(i, j)))
        self.edges = sorted(clean)
        self.neighbors = [[] for _ in range(n)]
        for i, j in self.edges:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

    @property
    def n(self) -> int:
        """Número de agentes."""
        return len(self.labels)

    def violated_edges(self, values: Sequence[int]) -> List[Tuple[int, int]]:
        """Aristas cuyos dos extremos tienen el mismo color (conflictos).

        Args:
            values: Color elegido por cada agente.

        Returns:
            Lista de aristas en conflicto.
        """
        return [(i, j) for i, j in self.edges if values[i] == values[j]]

    def n_violations(self, values: Sequence[int]) -> int:
        """Número de conflictos de una asignación completa.

        Args:
            values: Color elegido por cada agente.

        Returns:
            Número de aristas con el mismo color en ambos extremos.
        """
        return len(self.violated_edges(values))


def solve_centralized(problem: ColoringProblem) -> Optional[List[int]]:
    """Resuelve el coloreado con backtracking CENTRALIZADO (solo para validar a los agentes).

    Args:
        problem: Problema de coloreado (pensado para grafos pequeños).

    Returns:
        Una asignación válida de colores o ``None`` si el problema no tiene solución.
    """
    n, k = problem.n, problem.n_colors
    order = sorted(range(n), key=lambda i: -len(problem.neighbors[i]))
    values: Dict[int, int] = {}

    def rec(pos: int) -> bool:
        if pos == n:
            return True
        v = order[pos]
        for c in range(k):
            if all(values.get(w) != c for w in problem.neighbors[v]):
                values[v] = c
                if rec(pos + 1):
                    return True
                del values[v]
        return False

    return [values[i] for i in range(n)] if rec(0) else None


def spain_problem(n_colors: int = 3) -> ColoringProblem:
    """Mapa de las 15 comunidades autónomas peninsulares de España (vecindad por frontera terrestre).

    Args:
        n_colors: Número de colores disponibles.

    Returns:
        Problema de coloreado con un agente por comunidad autónoma.
    """
    regions = [
        ("Galicia", "Gal", (-8.0, 42.8)), ("Asturias", "Ast", (-5.9, 43.3)),
        ("Cantabria", "Can", (-4.0, 43.2)), ("País Vasco", "PV", (-2.7, 43.0)),
        ("Navarra", "Nav", (-1.65, 42.7)), ("La Rioja", "Rio", (-2.5, 42.3)),
        ("Aragón", "Ara", (-0.8, 41.5)), ("Cataluña", "Cat", (1.5, 41.8)),
        ("C. Valenciana", "Val", (-0.7, 39.5)), ("Murcia", "Mur", (-1.5, 38.0)),
        ("Andalucía", "And", (-4.5, 37.5)), ("Extremadura", "Ext", (-6.2, 39.2)),
        ("Castilla-La Mancha", "CLM", (-3.0, 39.5)), ("Madrid", "Mad", (-3.7, 40.4)),
        ("Castilla y León", "CyL", (-4.7, 41.7)),
    ]
    idx = {full: i for i, (full, _, _) in enumerate(regions)}
    borders = [
        ("Galicia", "Asturias"), ("Galicia", "Castilla y León"), ("Asturias", "Cantabria"),
        ("Asturias", "Castilla y León"), ("Cantabria", "País Vasco"), ("Cantabria", "Castilla y León"),
        ("País Vasco", "Navarra"), ("País Vasco", "La Rioja"), ("País Vasco", "Castilla y León"),
        ("Navarra", "La Rioja"), ("Navarra", "Aragón"), ("La Rioja", "Aragón"),
        ("La Rioja", "Castilla y León"), ("Aragón", "Cataluña"), ("Aragón", "C. Valenciana"),
        ("Aragón", "Castilla-La Mancha"), ("Aragón", "Castilla y León"), ("Cataluña", "C. Valenciana"),
        ("C. Valenciana", "Castilla-La Mancha"), ("C. Valenciana", "Murcia"),
        ("Murcia", "Castilla-La Mancha"), ("Murcia", "Andalucía"), ("Andalucía", "Castilla-La Mancha"),
        ("Andalucía", "Extremadura"), ("Extremadura", "Castilla-La Mancha"),
        ("Extremadura", "Castilla y León"), ("Castilla-La Mancha", "Madrid"),
        ("Castilla-La Mancha", "Castilla y León"), ("Madrid", "Castilla y León"),
    ]
    edges = [(idx[a], idx[b]) for a, b in borders]
    return ColoringProblem("España peninsular (15 CCAA)", n_colors, edges, [s for _, s, _ in regions],
                           [p for _, _, p in regions])
