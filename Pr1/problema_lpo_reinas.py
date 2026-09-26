import z3
from typing import Tuple, List

class ProblemaDominacionLPO:
    """Formalizacion del problema de Dominacion de Reinas usando Logica de Primer Orden.
    
    Utiliza el solucionador Z3 para definir predicados, axiomas y cuantificadores
    sobre un dominio finito de coordenadas enteras.
    
    Attributes:
        n (int): Dimension del tablero (N x N).
        gamma (int): Limite maximo de reinas permitidas (cardinalidad).
        solver (z3.Solver): Instancia del solucionador de teoremas Z3.
        reina (z3.Function): Predicado LPO Reina(x, y) que devuelve True si hay una reina.
        variables_posicion (List[Tuple[z3.ArithRef, z3.ArithRef]]): Lista de tuplas 
            que almacenan las variables existenciales para las posiciones de las reinas.
    """

    def __init__(self, n: int, gamma: int):
        """Inicializa el entorno logico y las estructuras fundamentales.
        
        Args:
            n (int): Tamano del tablero.
            gamma (int): Numero maximo de reinas.
        """
        self.n = n
        self.gamma = gamma
        self.solver = z3.Solver()
        self.reina = z3.Function('Reina', z3.IntSort(), z3.IntSort(), z3.BoolSort())
        self.variables_posicion = []

    def formular_axiomas(self) -> None:
        """Construye e inyecta los axiomas de la Logica de Primer Orden en el solucionador."""
        x, y, i, j = z3.Ints('x y i j')
        
        def en_tablero(fila: z3.ArithRef, columna: z3.ArithRef) -> z3.BoolRef:
            """Define el dominio valido para las variables cuantificadas."""
            return z3.And(fila >= 0, fila < self.n, columna >= 0, columna < self.n)
            
        def ataca(r1: z3.ArithRef, c1: z3.ArithRef, r2: z3.ArithRef, c2: z3.ArithRef) -> z3.BoolRef:
            """Axioma geometrico de ataque (incluye ataque a si misma)."""
            return z3.Or(
                r1 == r2,
                c1 == c2,
                r1 - r2 == c1 - c2,
                r1 - r2 == c2 - c1
            )
            
        # Axioma 1: Restriccion de Dominacion (Cobertura total)
        # Para todo (x,y) en el tablero, existe un (i,j) en el tablero tal que 
        # hay una Reina en (i,j) y ataca a (x,y).
        axioma_dominacion = z3.ForAll([x, y],
            z3.Implies(
                en_tablero(x, y),
                z3.Exists([i, j],
                    z3.And(
                        en_tablero(i, j),
                        self.reina(i, j),
                        ataca(i, j, x, y)
                    )
                )
            )
        )
        self.solver.add(axioma_dominacion)
        
        # Axioma 2: Restriccion de Cardinalidad
        # Declaramos existencialmente gamma coordenadas concretas.
        self.variables_posicion = [z3.Ints(f'r_{k} c_{k}') for k in range(self.gamma)]
        
        for r_k, c_k in self.variables_posicion:
            self.solver.add(en_tablero(r_k, c_k))
            
        condiciones_equivalencia = [z3.And(x == r_k, y == c_k) for r_k, c_k in self.variables_posicion]
        
        # Si el predicado Reina(x,y) es verdadero, (x,y) debe ser una de las gamma posiciones.
        axioma_cardinalidad = z3.ForAll([x, y],
            z3.Implies(
                self.reina(x, y),
                z3.Or(condiciones_equivalencia)
            )
        )
        self.solver.add(axioma_cardinalidad)

    def resolver(self) -> z3.ModelRef | None:
        """Ejecuta el solucionador SMT sobre los axiomas definidos.
        
        Returns:
            z3.ModelRef | None: El modelo matematico si el conjunto de axiomas es 
                satisfacible, o None en caso contrario.
        """
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None