# -*- coding: utf-8 -*-
"""Agentes y mensajes.

Cada agente solo conoce su propio valor y a sus vecinos. Se comunican
únicamente mediante mensajes. Protocolo de un tick (paper, Sec. I-A):
    1. Al agente i le salta el reloj, elige un vecino j y le envía REQUEST(x_i).
    2. j responde REPLY(x_j) y ambos fijan su valor a la media (x_i + x_j) / 2.
"""

from dataclasses import dataclass
from typing import List, Optional

REQUEST = "REQUEST"
REPLY = "REPLY"


@dataclass(frozen=True)
class Message:
    """Mensaje entre dos agentes.

    Attributes:
        kind: REQUEST o REPLY.
        sender: Identificador del emisor.
        receiver: Identificador del receptor.
        value: Valor del emisor antes de promediar.
    """

    kind: str
    sender: int
    receiver: int
    value: float


class GossipAgent:
    """Sensor que promedia su valor con el de un vecino elegido al azar.

    Attributes:
        ident: Identificador del agente.
        value: Valor actual (empieza siendo la lectura de su sensor).
        neighbors: Vecinos con los que puede comunicarse.
    """

    def __init__(self, ident: int, value: float, neighbors: List[int]) -> None:
        """Crea el agente.

        Args:
            ident: Identificador del agente.
            value: Lectura inicial del sensor.
            neighbors: Vecinos con los que puede comunicarse.
        """
        self.ident = ident
        self.value = float(value)
        self.neighbors = list(neighbors)

    def wake_up(self, u: float) -> Message:
        """Salta el reloj: elige un vecino al azar y le envía su valor.

        Args:
            u: Número uniforme en [0, 1) que decide qué vecino se elige.

        Returns:
            Mensaje REQUEST dirigido al vecino elegido.
        """
        j = self.neighbors[int(u * len(self.neighbors))]
        return Message(REQUEST, self.ident, j, self.value)

    def receive(self, msg: Message) -> Optional[Message]:
        """Procesa un mensaje y actualiza el valor del agente.

        Args:
            msg: Mensaje recibido.

        Returns:
            Un REPLY si el mensaje era un REQUEST; None en caso contrario.
        """
        reply = None
        if msg.kind == REQUEST:
            reply = Message(REPLY, self.ident, msg.sender, self.value)
        self.value = (self.value + msg.value) / 2.0
        return reply
