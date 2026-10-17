#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agentes y algoritmos ABT / AWC (Sec. 4 y 5 del artículo de Yokoo et al., 1998).

Cada agente controla una variable (su color), mantiene su visión de los demás (``agent_view``) y
solo se comunica mediante mensajes ``ok?``, ``nogood`` y ``addlink``. No depende de ningún otro
módulo del proyecto.
"""

from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

Pair = Tuple[int, int]          # (agente, color)
Nogood = FrozenSet[Pair]        # conjunto de pares (agente, color) que no pueden darse a la vez
Message = Tuple                 # ("ok", emisor, color, prioridad) | ("nogood", emisor, nogood) | ("addlink", emisor)


class Agent:
    """Agente que controla UNA variable (su color) y habla solo con sus vecinos.

    Implementa los dos algoritmos del artículo con el mismo código; la diferencia es el modo:

    - ``"abt"``: prioridades fijas (menor identificador = mayor prioridad). Cada agente envía su
      valor ("ok?") solo a los vecinos de menor prioridad y, al bloquearse, envía un "nogood" al
      agente de menor prioridad implicado (Sec. 4).
    - ``"awc"``: prioridades dinámicas. Se envía el valor a todos los vecinos junto con la
      prioridad actual; si el agente se bloquea, envía el nogood a todos los implicados, sube su
      prioridad a 1 + la máxima de su entorno y elige el color que menos conflictos cause
      (heurística min-conflict) (Sec. 5).

    Attributes:
        idx: Identificador del agente (también su prioridad base).
        mode: ``"abt"`` o ``"awc"``.
        n_colors: Tamaño del dominio.
        neighbors: Agentes con los que comparte una restricción.
        value: Color actual.
        priority: Valor de prioridad actual (siempre 0 en ABT).
        view: ``agent_view``: lo que el agente cree que valen los demás, ``{agente: (color, prioridad)}``.
        nogoods: Nogoods aprendidos o recibidos (restricciones derivadas).
        sent_nogoods: Nogoods ya enviados (AWC: evita repetir un cambio de prioridad).
        out_links: Agentes a los que debe enviar sus cambios de valor.
        no_solution: ``True`` si el agente ha derivado el nogood vacío (el problema no tiene solución).
    """

    def __init__(self, idx: int, mode: str, n_colors: int, neighbors: Sequence[int],
                 initial_value: int, rng: random.Random) -> None:
        """Crea el agente con un color inicial.

        Args:
            idx: Identificador del agente.
            mode: ``"abt"`` o ``"awc"``.
            n_colors: Número de colores del dominio.
            neighbors: Vecinos del agente en el grafo.
            initial_value: Color inicial (aleatorio en el artículo).
            rng: Generador de números aleatorios (para desempates).

        Raises:
            ValueError: Si ``mode`` no es ``"abt"`` ni ``"awc"``.
        """
        if mode not in ("abt", "awc"):
            raise ValueError("mode debe ser 'abt' o 'awc'")
        self.idx, self.mode, self.n_colors = idx, mode, n_colors
        self.neighbors: Set[int] = set(neighbors)
        self.value = initial_value
        self.priority = 0
        self.view: Dict[int, Tuple[int, int]] = {}
        self.nogoods: Set[Nogood] = set()
        self.sent_nogoods: Set[Nogood] = set()
        self.no_solution = False
        self.rng = rng
        # ABT: solo se informa a los vecinos de menor prioridad. AWC: a todos los vecinos.
        self.out_links: Set[int] = {j for j in self.neighbors if mode == "awc" or j > idx}

    # ---------- utilidades ----------
    def _is_higher(self, j: int, priority_j: int) -> bool:
        """Indica si el agente ``j`` tiene más prioridad que este agente.

        Mayor valor de prioridad = más prioridad; a igualdad, gana el de menor identificador.

        Args:
            j: Identificador del otro agente.
            priority_j: Valor de prioridad del otro agente.

        Returns:
            ``True`` si ``j`` tiene prioridad sobre este agente.
        """
        return (priority_j, -j) > (self.priority, -self.idx)

    def _reason(self, d: int) -> Optional[FrozenSet[Pair]]:
        """Motivo por el que el color ``d`` es inaceptable dada la visión actual (o ``None``).

        El color es inaceptable si un vecino de MAYOR prioridad ya lo tiene, o si algún nogood
        guardado es compatible con la visión actual y con ``(yo, d)`` y todos los demás agentes del
        nogood tienen más prioridad que yo. Si hay varios motivos se
        devuelve el más pequeño (cuanto menor es un nogood, más información aporta y es la única
        forma de llegar a deducir el nogood vacío, que prueba que no hay solución).

        Args:
            d: Color a evaluar.

        Returns:
            Conjunto de pares (agente, color) de los demás agentes que justifican el rechazo, o
            ``None`` si el color es consistente.
        """
        reasons: List[FrozenSet[Pair]] = []
        for j, (vj, pj) in self.view.items():
            if j in self.neighbors and vj == d and self._is_higher(j, pj):
                reasons.append(frozenset({(j, vj)}))
        for ng in self.nogoods:
            compatible = True
            for a, va in ng:
                if a == self.idx:
                    if va != d:
                        compatible = False
                        break
                elif a not in self.view or self.view[a][0] != va or not self._is_higher(a, self.view[a][1]):
                    # Un nogood solo me obliga a mí si TODOS los demás agentes que lo forman tienen
                    # más prioridad que yo; si no, son ellos quienes deben adaptarse.
                    compatible = False
                    break
            if compatible:
                reasons.append(frozenset(p for p in ng if p[0] != self.idx))
        return min(reasons, key=len) if reasons else None

    def _conflicts_with_lower(self, d: int) -> int:
        """Número de vecinos de menor prioridad que ya tienen el color ``d`` (heurística min-conflict).

        Args:
            d: Color a evaluar.

        Returns:
            Número de violaciones de restricciones con vecinos de menor prioridad.
        """
        return sum(1 for j, (vj, pj) in self.view.items()
                   if j in self.neighbors and vj == d and not self._is_higher(j, pj))

    def _broadcast_value(self, out: List[Tuple[int, Message]]) -> None:
        """Envía el color y la prioridad actuales a los agentes enlazados (mensaje ``ok?``).

        Args:
            out: Lista donde se acumulan los mensajes salientes ``(destino, mensaje)``.
        """
        for j in sorted(self.out_links):
            out.append((j, ("ok", self.idx, self.value, self.priority)))

    # ---------- protocolo ----------
    def start(self, out: List[Tuple[int, Message]]) -> None:
        """Primer paso: cada agente comunica su color inicial.

        Args:
            out: Lista donde se acumulan los mensajes salientes.
        """
        self._broadcast_value(out)

    def receive(self, messages: Sequence[Message], out: List[Tuple[int, Message]]) -> None:
        """Procesa los mensajes recibidos en un ciclo y reacciona (Fig. 2 y 5 del artículo).

        Primero se actualiza el conocimiento del agente con todos los mensajes y después se
        ejecuta ``check_agent_view`` una sola vez (como permiten las notas 5 y 10 del artículo).

        Args:
            messages: Mensajes entregados en este ciclo.
            out: Lista donde se acumulan los mensajes salientes.
        """
        replies: List[int] = []
        old_value, old_priority = self.value, self.priority
        for msg in messages:
            kind = msg[0]
            if kind == "ok":
                _, j, vj, pj = msg
                self.view[j] = (vj, pj)
            elif kind == "nogood":
                _, j, ng = msg
                self.nogoods.add(ng)
                self.out_links.add(j)  # el emisor del nogood debe enterarse de mis cambios de valor
                for a, va in ng:  # nogood con agentes desconocidos: pedir enlace y anotar su valor
                    if a != self.idx and a not in self.view:
                        self.view[a] = (va, 0)
                        out.append((a, ("addlink", self.idx)))
                replies.append(j)
            elif kind == "addlink":
                _, j = msg
                self.out_links.add(j)
                out.append((j, ("ok", self.idx, self.value, self.priority)))
        self._check_agent_view(out)
        already = {dest for dest, m in out if m[0] == "ok"}
        if (self.value, self.priority) == (old_value, old_priority):
            for j in replies:  # el emisor del nogood debe conocer mi valor actual
                if j not in already:
                    out.append((j, ("ok", self.idx, self.value, self.priority)))

    def _check_agent_view(self, out: List[Tuple[int, Message]]) -> None:
        """Comprueba si el valor actual es consistente y, si no, lo cambia o hace backtrack.

        Args:
            out: Lista donde se acumulan los mensajes salientes.
        """
        if self._reason(self.value) is None:
            return
        candidates = [d for d in range(self.n_colors) if self._reason(d) is None]
        if candidates:
            if self.mode == "awc":  # min-conflict con los vecinos de menor prioridad
                best = min(self._conflicts_with_lower(d) for d in candidates)
                candidates = [d for d in candidates if self._conflicts_with_lower(d) == best]
            self.value = self.rng.choice(candidates)
            self._broadcast_value(out)
            return
        self._backtrack(out)

    def _backtrack(self, out: List[Tuple[int, Message]]) -> None:
        """Ningún color es consistente: genera un nogood y lo comunica.

        El nogood es la unión de los motivos por los que se rechaza cada color ("hiperresolvente").
        ABT lo envía al agente de menor prioridad del nogood, lo borra de su visión y reintenta.
        AWC lo envía a todos los agentes del nogood (si es nuevo), sube su prioridad por encima de
        todos y elige el color con menos conflictos.

        Args:
            out: Lista donde se acumulan los mensajes salientes.
        """
        nogood: Nogood = frozenset().union(*(self._reason(d) for d in range(self.n_colors)))
        if not nogood:  # nogood vacío: ninguna asignación es posible
            self.no_solution = True
            return
        if self.mode == "abt":
            lowest = max(a for a, _ in nogood)
            out.append((lowest, ("nogood", self.idx, nogood)))
            del self.view[lowest]
            self._check_agent_view(out)
            return
        if nogood in self.sent_nogoods:  # ya avisé de esto: espero a recibir novedades
            return
        self.sent_nogoods.add(nogood)
        for a, _ in sorted(nogood):
            out.append((a, ("nogood", self.idx, nogood)))
        self.priority = 1 + max(p for _, p in self.view.values())
        # Con la nueva prioridad máxima ya no hay vecinos "superiores": solo quedan los nogoods.
        allowed = [d for d in range(self.n_colors) if self._reason(d) is None] or list(range(self.n_colors))
        conflicts = {d: sum(1 for j, (vj, _) in self.view.items() if j in self.neighbors and vj == d)
                     for d in allowed}
        best = min(conflicts.values())
        self.value = self.rng.choice([d for d in allowed if conflicts[d] == best])
        self._broadcast_value(out)
