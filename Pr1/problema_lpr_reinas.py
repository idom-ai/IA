from itertools import combinations


class ProblemaNReinasCNF:
    """Formalización SAT (FNC) del problema de las N-Reinas.

    Colocar N reinas en un tablero N x N de modo que ningún par se ataque
    (distinta fila, columna y diagonal).

    Variable proposicional x_{i,j}: "hay una reina en la casilla (i, j)".

    Atributos:
        n (int): Tamaño del tablero (N x N) y número de reinas a colocar.
    """

    def __init__(self, n: int):
        """Inicializa el problema para un tablero de tamaño N."""
        self.n = n

    def obtener_id_variable(self, fila: int, columna: int) -> int:
        """Identificador entero único (1..N*N) de la variable x_{fila,columna}."""
        return fila * self.n + columna + 1

    def generar_base_conocimiento(self) -> tuple[list[list[int]], list[int]]:
        """Construye las cláusulas FNC y el conjunto de variables del problema.

        Returns:
            tuple[list[list[int]], list[int]]: Cláusulas y variables proposicionales.
        """
        n = self.n
        clausulas = []
        variables = [self.obtener_id_variable(f, c) for f in range(n) for c in range(n)]

        for i in range(n):
            # Al menos una reina en la fila i y en la columna i.
            clausulas.append([self.obtener_id_variable(i, j) for j in range(n)])
            clausulas.append([self.obtener_id_variable(j, i) for j in range(n)])

            # A lo sumo una reina por fila y por columna: para cada PAR de casillas, no ambas.
            for j1, j2 in combinations(range(n), 2):
                clausulas.append([-self.obtener_id_variable(i, j1), -self.obtener_id_variable(i, j2)])
                clausulas.append([-self.obtener_id_variable(j1, i), -self.obtener_id_variable(j2, i)])

        # A lo sumo una reina por diagonal: para cada par de casillas en la misma diagonal, no ambas.
        casillas = [(f, c) for f in range(n) for c in range(n)]
        for (f1, c1), (f2, c2) in combinations(casillas, 2):
            if abs(f1 - f2) == abs(c1 - c2):
                clausulas.append([-self.obtener_id_variable(f1, c1), -self.obtener_id_variable(f2, c2)])

        return clausulas, variables

    def extraer_reinas(self, modelo: dict[int, bool]) -> list[tuple[int, int]]:
        """Devuelve las coordenadas (fila, columna) con reina según el modelo."""
        return [(i, j) for i in range(self.n) for j in range(self.n)
                if modelo.get(self.obtener_id_variable(i, j), False)]

    def es_solucion_valida(self, modelo: dict[int, bool]) -> bool:
        """Verificación independiente del solver: N reinas y ningún par se ataca."""
        reinas = self.extraer_reinas(modelo)
        if len(reinas) != self.n:
            return False
        return all(a[0] != b[0] and a[1] != b[1] and abs(a[0] - b[0]) != abs(a[1] - b[1])
                   for a, b in combinations(reinas, 2))