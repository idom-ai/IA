import random

class SatCaminarSolver:
    """Implementación del algoritmo de búsqueda local SAT-CAMINAR (WalkSAT).
    
    Este solucionador evalúa una base de conocimiento en Forma Normal Conjuntiva (FNC)
    e intenta encontrar una asignación de verdad que satisfaga todas las cláusulas
    mediante intercambios heurísticos.
    
    Atributos:
        probabilidad_aleatoria (float): Probabilidad (0.0 a 1.0) de realizar un paso 
            de exploración aleatoria en lugar de un paso determinista (min-conflictos).
        max_iteraciones (int): Número máximo de intercambios permitidos antes 
            de declarar un fallo en la búsqueda.
    """

    def __init__(self, probabilidad_aleatoria: float = 0.5, max_iteraciones: int = 100000):
        """Inicializa el solucionador con los parámetros de búsqueda.
        
        Args:
            probabilidad_aleatoria (float): Probabilidad de paso aleatorio.
            max_iteraciones (int): Límite de iteraciones del algoritmo.
        """
        self.probabilidad_aleatoria = probabilidad_aleatoria
        self.max_iteraciones = max_iteraciones

    def _evaluar_clausula(self, clausula: list[int], modelo: dict[int, bool]) -> bool:
        """Determina si una cláusula específica se satisface bajo el modelo actual.
        
        Args:
            clausula (list[int]): Lista de literales enteros (positivos o negativos).
            modelo (dict[int, bool]): Asignación actual de variables proposicionales.
            
        Returns:
            bool: True si la cláusula es verdadera, False en caso contrario.
        """
        for literal in clausula:
            variable = abs(literal)
            valor = modelo.get(variable, False)
            if (literal > 0 and valor) or (literal < 0 and not valor):
                return True
        return False

    def resolver(self, clausulas: list[list[int]], variables: list[int]) -> dict[int, bool] | None:
        """Ejecuta el algoritmo SAT-CAMINAR sobre el conjunto de cláusulas.
        
        Args:
            clausulas (list[list[int]]): Lista de cláusulas, donde cada cláusula es una 
                lista de literales enteros.
            variables (list[int]): Lista de identificadores de todas las variables.
            
        Returns:
            dict[int, bool] | None: Diccionario con el modelo satisfactorio si se
                encuentra; None si se alcanza el límite de iteraciones sin éxito.
        """
        modelo = {v: random.choice([True, False]) for v in variables}
        
        for _ in range(self.max_iteraciones):
            clausulas_falsas = [c for c in clausulas if not self._evaluar_clausula(c, modelo)]
            
            if not clausulas_falsas:
                return modelo
                
            clausula_objetivo = random.choice(clausulas_falsas)
            
            if random.random() < self.probabilidad_aleatoria:
                variable_a_cambiar = abs(random.choice(clausula_objetivo))
                modelo[variable_a_cambiar] = not modelo[variable_a_cambiar]
            else:
                mejor_variable = None
                minimos_conflictos = float('inf')
                
                for literal in clausula_objetivo:
                    variable = abs(literal)
                    modelo[variable] = not modelo[variable]
                    
                    conflictos_actuales = sum(1 for c in clausulas if not self._evaluar_clausula(c, modelo))
                    
                    if conflictos_actuales < minimos_conflictos:
                        minimos_conflictos = conflictos_actuales
                        mejor_variable = variable
                        
                    modelo[variable] = not modelo[variable]
                    
                if mejor_variable is not None:
                    modelo[mejor_variable] = not modelo[mejor_variable]
                
        return None