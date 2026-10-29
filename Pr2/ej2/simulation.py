# -*- coding: utf-8 -*-
"""Simulación asíncrona del gossip y medida del tiempo de ε-promediado K."""

import math
from typing import List, Tuple

import numpy as np

from agents import GossipAgent


def run_gossip(neighbors: List[List[int]], readings: np.ndarray, rng: np.random.Generator,
               rel_tol: float, max_ticks: int = 200000) -> Tuple[np.ndarray, int]:
    """Simula el gossip hasta que el error baja de rel_tol * (error inicial).

    En cada tick se despierta un agente elegido al azar, que intercambia dos
    mensajes con un vecino. Los agentes no conocen el promedio: la condición de
    parada solo la usa el observador (nosotros) para saber cuándo terminar.

    Args:
        neighbors: Lista de vecinos de cada agente.
        readings: Lecturas iniciales de los sensores.
        rng: Generador aleatorio.
        rel_tol: Se para cuando ||x(t) - promedio|| < rel_tol * ||x(0) - promedio||.
        max_ticks: Límite de seguridad de ticks.

    Returns:
        Tupla (historial de valores de forma (ticks + 1, n), número de mensajes).
    """
    n = len(readings)
    agents = [GossipAgent(i, readings[i], neighbors[i]) for i in range(n)]
    average = float(np.mean(readings))
    limit = rel_tol * np.linalg.norm(readings - average)
    history = [readings.astype(float)]
    messages = 0
    for _ in range(max_ticks):
        request = agents[rng.integers(n)].wake_up(rng.random())
        reply = agents[request.receiver].receive(request)
        agents[reply.receiver].receive(reply)
        messages += 2
        values = np.array([a.value for a in agents])
        history.append(values)
        if np.linalg.norm(values - average) < limit:
            break
    return np.array(history), messages


def averaging_time(neighbors: List[List[int]], n_runs: int, epsilon: float,
                   true_value: float, noise_std: float, rng: np.random.Generator) -> Tuple[int, np.ndarray]:
    """Mide K, el tiempo de ε-promediado, repitiendo el experimento muchas veces.

    En cada ejecución (lecturas y relojes distintos) se anota el tick en el que
    el error baja de epsilon * (error inicial). Como el error nunca aumenta, K
    es el menor número de ticks tras el cual a lo sumo una fracción epsilon de
    las ejecuciones aún no ha llegado.

    Args:
        neighbors: Lista de vecinos de cada agente.
        n_runs: Número de ejecuciones.
        epsilon: Precisión pedida (y fracción de ejecuciones que se tolera fallar).
        true_value: Valor real que miden los sensores.
        noise_std: Desviación típica del ruido de los sensores.
        rng: Generador aleatorio.

    Returns:
        Tupla (K en ticks, ticks que necesitó cada ejecución).
    """
    ticks = []
    for _ in range(n_runs):
        readings = true_value + noise_std * rng.standard_normal(len(neighbors))
        _, messages = run_gossip(neighbors, readings, rng, epsilon)
        ticks.append(messages // 2)
    ticks = np.sort(ticks)
    needed = math.ceil(round((1.0 - epsilon) * n_runs, 9))
    return int(ticks[needed - 1]), ticks
