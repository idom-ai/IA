class ProblemaNReinasCNF:
    """Generador de la formalización SAT en FNC para el problema de las N-Reinas.
    
    Atributos:
        n (int): Tamaño del tablero (N x N) y número de reinas a colocar.
    """

    def __init__(self, n: int):
        """Inicializa el problema para un tablero de tamaño N.
        
        Args:
            n (int): Dimensión del tablero.
        """
        self.n = n

    def obtener_id_variable(self, fila: int, columna: int) -> int:
        """Asigna un identificador entero único a la coordenada del tablero.
        
        Args:
            fila (int): Índice de la fila (0 a n-1).
            columna (int): Índice de la columna (0 a n-1).
            
        Returns:
            int: Identificador único positivo para la variable proposicional.
        """
        return fila * self.n + columna + 1

    def generar_base_conocimiento(self) -> tuple[list[list[int]], list[int]]:
        """Construye las cláusulas FNC y el conjunto de variables del problema.
        
        Returns:
            tuple[list[list[int]], list[int]]: Tupla que contiene la lista de 
                cláusulas y la lista de variables proposicionales.
        """
        clausulas = []
        variables = [self.obtener_id_variable(f, c) for f in range(self.n) for c in range(self.n)]
        
        for i in range(self.n):
            clausulas.append([self.obtener_id_variable(i, j) for j in range(self.n)])
            clausulas.append([self.obtener_id_variable(j, i) for j in range(self.n)])
            
            for j1 in range(self.n):
                for j2 in range(j1 + 1, self.n):
                    clausulas.append([-self.obtener_id_variable(i, j1), -self.obtener_id_variable(i, j2)])
                    clausulas.append([-self.obtener_id_variable(j1, i), -self.obtener_id_variable(j2, i)])
                    
        for i1 in range(self.n):
            for j1 in range(self.n):
                for i2 in range(self.n):
                    for j2 in range(self.n):
                        if i1 == i2 and j1 == j2:
                            continue
                        # Restricción de diagonal principal
                        if i1 - j1 == i2 - j2:
                            clausulas.append([-self.obtener_id_variable(i1, j1), -self.obtener_id_variable(i2, j2)])
                        # Restricción de diagonal secundaria
                        if i1 + j1 == i2 + j2:
                            clausulas.append([-self.obtener_id_variable(i1, j1), -self.obtener_id_variable(i2, j2)])
                            
        # Se elimina la duplicidad de cláusulas
        clausulas_unicas = []
        vistas = set()
        for clausula in clausulas:
            tupla_clausula = tuple(sorted(clausula))
            if tupla_clausula not in vistas:
                vistas.add(tupla_clausula)
                clausulas_unicas.append(clausula)
                
        return clausulas_unicas, variables