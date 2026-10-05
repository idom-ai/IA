import z3


class ProblemaDominacionLPO:
    """Formalización del problema de Dominación de Reinas en Lógica de Primer Orden.

    Se buscan, como mucho, gamma reinas que dominen (ataquen u ocupen) todas las casillas.
    Opcionalmente, las reinas no pueden atacarse entre sí (dominación independiente).

    Utiliza el solucionador Z3 para definir predicados, axiomas y cuantificadores
    sobre un dominio finito de coordenadas enteras.

    Attributes:
        n (int): Dimensión del tablero (N x N).
        gamma (int): Número máximo de reinas permitidas (cardinalidad).
        solver (z3.Solver): Instancia del solucionador Z3.
        reina (z3.FuncDeclRef): Predicado LPO Reina(x, y).
        variables_posicion (list): Pares de variables (r_k, c_k) con las posiciones
            de los 'asientos' disponibles para las reinas.
    """

    def __init__(self, n: int, gamma: int, independiente: bool = False):
        """Inicializa el entorno lógico.

        Args:
            n (int): Tamaño del tablero.
            gamma (int): Número máximo de reinas.
            independiente (bool): Exigir que las reinas no se ataquen entre sí.
        """
        
        self.n = n
        self.gamma = gamma
        self.independiente = independiente
        self.solver = z3.Solver()
        self.reina = z3.Function('Reina', z3.IntSort(), z3.IntSort(), z3.BoolSort())
        self.variables_posicion = []

    def formular_axiomas(self) -> None:
        """Construye e inyecta los axiomas de la Lógica de Primer Orden en el solucionador."""
        x, y, i, j = z3.Ints('x y i j')

        def en_tablero(fila, columna) -> z3.BoolRef:
            """Dominio válido para las variables cuantificadas."""
            return z3.And(fila >= 0, fila < self.n, columna >= 0, columna < self.n)

        def ataca(r1, c1, r2, c2) -> z3.BoolRef:
            """Ataque de reina: misma fila, columna o diagonal (incluye la propia casilla)."""
            return z3.Or(r1 == r2, c1 == c2, r1 - r2 == c1 - c2, r1 - r2 == c2 - c1)

        # Axioma 1 (dominación): toda casilla está atacada por alguna reina.
        self.solver.add(z3.ForAll([x, y],
            z3.Implies(
                en_tablero(x, y),
                z3.Exists([i, j], z3.And(en_tablero(i, j), self.reina(i, j), ataca(i, j, x, y)))
            )
        ))

        # Axioma 2 (cardinalidad): a lo sumo gamma reinas, en gamma posiciones concretas.
        self.variables_posicion = [z3.Ints(f'r_{k} c_{k}') for k in range(self.gamma)]
        for r_k, c_k in self.variables_posicion:
            self.solver.add(en_tablero(r_k, c_k))

        posibles = [z3.And(x == r_k, y == c_k) for r_k, c_k in self.variables_posicion]
        self.solver.add(z3.ForAll([x, y], z3.Implies(self.reina(x, y), z3.Or(posibles))))

        # Axioma 3 (opcional, independencia): las gamma reinas no pueden atacarse entre sí
        # (ni compartir casilla, ya que ataca() incluye r1 == r2 y c1 == c2).
        if self.independiente:
            for k in range(self.gamma):
                for l in range(k + 1, self.gamma):
                    r_k, c_k = self.variables_posicion[k]
                    r_l, c_l = self.variables_posicion[l]
                    self.solver.add(z3.Not(ataca(r_k, c_k, r_l, c_l)))

    def resolver(self) -> z3.ModelRef | None:
        """Devuelve el modelo si los axiomas son satisfacibles; None en caso contrario."""
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None

    def obtener_reinas(self, modelo: z3.ModelRef) -> set[tuple[int, int]]:
        """Extrae las casillas donde el predicado Reina(x, y) es verdadero en el modelo.

        Se evalúa el predicado (no solo las variables de posición), de modo que
        nunca se dibuja una reina que el modelo no considere realmente colocada.
        """
        reinas = set()
        for r_var, c_var in self.variables_posicion:
            r = modelo.evaluate(r_var, model_completion=True).as_long()
            c = modelo.evaluate(c_var, model_completion=True).as_long()
            if z3.is_true(modelo.evaluate(self.reina(z3.IntVal(r), z3.IntVal(c)), model_completion=True)):
                reinas.add((r, c))
        return reinas

    def verificar_dominacion(self, reinas: set[tuple[int, int]]) -> bool:
        """Comprobación independiente de Z3: ¿toda casilla está atacada por alguna reina?"""
        return all(
            any(r == x or c == y or abs(r - x) == abs(c - y) for r, c in reinas)
            for x in range(self.n) for y in range(self.n)
        )

    def verificar_independencia(self, reinas: set[tuple[int, int]]) -> bool:
        """Comprobación independiente de Z3: ¿ningún par de reinas se ataca?"""
        lista = sorted(reinas)
        return all(
            a[0] != b[0] and a[1] != b[1] and abs(a[0] - b[0]) != abs(a[1] - b[1])
            for k, a in enumerate(lista) for b in lista[k + 1:]
        )