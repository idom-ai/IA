#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilidades de impresión por consola: encabezados, tablas alineadas y comprobaciones."""

from __future__ import annotations

from typing import Sequence


def banner(text: str) -> None:
    """Imprime un encabezado de sección.

    Args:
        text: Texto del encabezado.
    """
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def print_table(headers: Sequence[str], rows: Sequence[Sequence]) -> None:
    """Imprime una tabla de texto alineada.

    Args:
        headers: Nombres de las columnas.
        rows: Filas de la tabla.
    """
    cells = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for n, row in enumerate(cells):
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
        if n == 0:
            print("-+-".join("-" * w for w in widths))


def check(condition: bool, ok_msg: str, bad_msg: str) -> None:
    """Imprime el resultado de una comprobación.

    Args:
        condition: Resultado de la comprobación.
        ok_msg: Mensaje si es verdadera.
        bad_msg: Mensaje si es falsa.
    """
    print(f"   [{'OK' if condition else 'ERROR'}] {ok_msg if condition else bad_msg}")
