#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulador por ciclos de los agentes distribuidos (Sec. 6 del artículo de Yokoo et al., 1998).

Un ciclo = cada agente lee sus mensajes, calcula y envía; un mensaje enviado en el ciclo t se
entrega en el ciclo t + 1. Se detiene al llegar a un estado estable, al deducirse que no hay
solución o al alcanzar el límite de ciclos.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from agentes import Agent, Message
from problema import ColoringProblem


@dataclass
class RunResult:
    """Resultado de ejecutar un algoritmo distribuido sobre un problema.

    Attributes:
        algorithm: ``"ABT"`` o ``"AWC"``.
        status: ``"resuelto"``, ``"sin_solucion"`` (nogood vacío), ``"limite"`` (se agotaron los
            ciclos) o ``"error"`` (estado estable inconsistente; no debería ocurrir).
        cycles: Ciclos de simulación ejecutados.
        messages: Total de mensajes enviados.
        values: Colores finales de los agentes.
        trace: Para cada ciclo: ``(colores, prioridades, nº de conflictos, mensajes del ciclo)``;
            el elemento 0 es la situación inicial.
        log: Para cada ciclo, los mensajes enviados como ``(destinatario, mensaje)``; el elemento 0
            son los mensajes iniciales. Solo se rellena con ``record_trace=True``.
        failed_agent: Agente que dedujo el nogood vacío (``None`` si no ocurrió).
    """

    algorithm: str
    status: str
    cycles: int
    messages: int
    values: List[int]
    trace: List[Tuple[List[int], List[int], int, int]] = field(default_factory=list)
    log: List[List[Tuple[int, Message]]] = field(default_factory=list)
    failed_agent: Optional[int] = None


def run_distributed(problem: ColoringProblem, algorithm: str, initial_values: Sequence[int],
                    seed: int = 0, max_cycles: int = 1000, record_trace: bool = False) -> RunResult:
    """Simula la ejecución concurrente de los agentes, ciclo a ciclo.

    Un ciclo consiste en que cada agente lee los mensajes que le llegan, hace su cómputo local y
    envía mensajes; un mensaje enviado en el ciclo t se entrega en el ciclo t + 1 (Sec. 6). La
    simulación termina cuando no circula ningún mensaje (estado estable), cuando algún agente
    deduce que no hay solución o cuando se alcanza ``max_cycles`` (1000 en el artículo).

    Args:
        problem: Problema de coloreado.
        algorithm: ``"ABT"`` o ``"AWC"``.
        initial_values: Color inicial de cada agente.
        seed: Semilla de los desempates aleatorios de los agentes.
        max_cycles: Límite de ciclos.
        record_trace: Si es ``True``, guarda la evolución completa (para dibujarla).

    Returns:
        El resultado de la simulación.
    """
    rng = random.Random(seed)
    agents = [Agent(i, algorithm.lower(), problem.n_colors, problem.neighbors[i], initial_values[i], rng)
              for i in range(problem.n)]
    inboxes: List[List[Message]] = [[] for _ in agents]
    total_msgs = 0
    start_out: List[Tuple[int, Message]] = []
    for ag in agents:
        ag.start(start_out)
    for dest, msg in start_out:
        inboxes[dest].append(msg)
    total_msgs += len(start_out)
    trace: List[Tuple[List[int], List[int], int, int]] = []
    log: List[List[Tuple[int, Message]]] = [list(start_out)] if record_trace else []

    def snapshot(n_msgs: int) -> None:
        if record_trace:
            vals = [a.value for a in agents]
            trace.append((vals, [a.priority for a in agents], problem.n_violations(vals), n_msgs))

    snapshot(len(start_out))
    cycles = 0
    status = "limite"
    while cycles < max_cycles:
        if not any(inboxes):  # nadie tiene mensajes pendientes: estado estable
            status = "estable"
            break
        delivering, inboxes = inboxes, [[] for _ in agents]
        sent = 0
        cycle_log: List[Tuple[int, Message]] = []
        for ag, msgs in zip(agents, delivering):
            if msgs:
                out: List[Tuple[int, Message]] = []
                ag.receive(msgs, out)
                for dest, msg in out:
                    inboxes[dest].append(msg)
                sent += len(out)
                cycle_log.extend(out)
        if record_trace:
            log.append(cycle_log)
        cycles += 1
        total_msgs += sent
        snapshot(sent)
        if any(a.no_solution for a in agents):
            status = "sin_solucion"
            break
    else:
        status = "limite" if any(inboxes) else "estable"
    values = [a.value for a in agents]
    if status == "estable":
        status = "resuelto" if problem.n_violations(values) == 0 else "error"
    failed = next((a.idx for a in agents if a.no_solution), None)
    return RunResult(algorithm.upper(), status, cycles, total_msgs, values, trace, log, failed)


def random_initial_values(problem: ColoringProblem, rng: random.Random) -> List[int]:
    """Colores iniciales aleatorios, uno por agente (como en el artículo).

    Args:
        problem: Problema de coloreado.
        rng: Generador de números aleatorios.

    Returns:
        Lista con un color aleatorio por agente.
    """
    return [rng.randrange(problem.n_colors) for _ in range(problem.n)]
