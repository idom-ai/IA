# -*- coding: utf-8 -*-
"""Las tres topologías de red: anillo, geométrica aleatoria y completa.

Cada función devuelve (posiciones para dibujar, lista de vecinos de cada nodo).
La completa ("todos con todos") y la geométrica aleatoria son las redes que
estudia el paper (Boyd et al., 2006: grafo completo en el Cor. 3, y grafo
geométrico aleatorio en la Sec. VI-A y la Fig. 3). El anillo es el caso
unidimensional más simple (nodos en una circunferencia, cada uno con sus dos
vecinos inmediatos).
"""

from collections import deque
from typing import List, Tuple

import numpy as np

Network = Tuple[np.ndarray, List[List[int]]]


def _circle(n: int) -> np.ndarray:
    """Posiciones de n nodos repartidos en una circunferencia.

    Args:
        n: Número de nodos.

    Returns:
        Array (n, 2) con las posiciones.
    """
    angles = 2 * np.pi * np.arange(n) / n
    return 0.5 + 0.45 * np.column_stack([np.cos(angles), np.sin(angles)])


def ring(n: int) -> Network:
    """Anillo: cada nodo conectado a sus dos vecinos inmediatos.

    Args:
        n: Número de nodos.

    Returns:
        Tupla (posiciones, lista de vecinos).
    """
    return _circle(n), [[(i - 1) % n, (i + 1) % n] for i in range(n)]


def complete(n: int) -> Network:
    """Completa ("todos con todos"): cada nodo conectado a todos los demás.

    Args:
        n: Número de nodos.

    Returns:
        Tupla (posiciones, lista de vecinos).
    """
    return _circle(n), [[j for j in range(n) if j != i] for i in range(n)]


def _is_connected(adjacency: np.ndarray) -> bool:
    """Comprueba con una búsqueda en anchura que todos los nodos se alcanzan.

    Args:
        adjacency: Matriz de adyacencia (n, n) con 0/1.

    Returns:
        True si el grafo es conexo.
    """
    seen = {0}
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v in np.flatnonzero(adjacency[u]):
            if int(v) not in seen:
                seen.add(int(v))
                queue.append(int(v))
    return len(seen) == adjacency.shape[0]


def geometric(n: int, rng: np.random.Generator) -> Network:
    """Grafo geométrico aleatorio conexo (modelo de red de sensores).

    Se colocan n nodos al azar en el cuadrado unidad y se conectan los que están
    a distancia <= sqrt(log(n) / n) (orden del umbral de conectividad, Lema 9
    del paper). Se vuelve a sortear hasta que el grafo sea conexo.

    Args:
        n: Número de nodos.
        rng: Generador aleatorio.

    Returns:
        Tupla (posiciones, lista de vecinos).
    """
    radius = np.sqrt(np.log(n) / n)
    while True:
        pos = rng.random((n, 2))
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        adjacency = ((dist <= radius) & (dist > 0)).astype(int)
        if _is_connected(adjacency):
            return pos, [list(np.flatnonzero(row)) for row in adjacency]
