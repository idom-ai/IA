import logging
import matplotlib.pyplot as plt
import numpy as np
from sat_solver import SatCaminarSolver
from problema_reinas import ProblemaNReinasCNF

class VisualizadorNReinas:
    """Clase encargada de renderizar la solución gráfica del problema.
    
    Atributos:
        n (int): Tamaño del tablero.
    """

    def __init__(self, n: int):
        """Inicializa el visualizador para un tablero de tamaño N.
        
        Args:
            n (int): Dimensión del tablero.
        """
        self.n = n

    def guardar_grafico_tablero(self, modelo: dict[int, bool], problema: ProblemaNReinasCNF, ruta_archivo: str) -> None:
        """Genera y guarda una imagen PNG que representa el tablero y las reinas.
        
        Args:
            modelo (dict[int, bool]): Diccionario resultante del solver SAT.
            problema (ProblemaNReinasCNF): Instancia del generador del problema.
            ruta_archivo (str): Ruta de destino para guardar la imagen.
        """
        tablero = np.zeros((self.n, self.n))
        tablero[1::2, ::2] = 1
        tablero[::2, 1::2] = 1

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(tablero, cmap='gray_r', origin='upper')

        for i in range(self.n):
            for j in range(self.n):
                id_var = problema.obtener_id_variable(i, j)
                if modelo.get(id_var, False):
                    ax.text(j, i, '♕', fontsize=30, ha='center', va='center', color='black')

        ax.set_xticks(np.arange(self.n) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.n) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Solución SAT-CAMINAR para un tablero {self.n}x{self.n}")

        plt.show()

def main() -> None:
    """Punto de entrada de la demostración."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    n_reinas = 20
    logging.info(f"Formalizando el problema de las {n_reinas}-Reinas en FNC.")
    
    generador_problema = ProblemaNReinasCNF(n_reinas)
    clausulas, variables = generador_problema.generar_base_conocimiento()
    
    logging.info(f"Variables generadas: {len(variables)}")
    logging.info(f"Cláusulas generadas: {len(clausulas)}")
    
    solver = SatCaminarSolver(probabilidad_aleatoria=0.4, max_iteraciones=300000)
    logging.info("Ejecutando algoritmo SAT-CAMINAR. Esto puede tomar unos instantes...")
    
    modelo = solver.resolver(clausulas, variables)
    
    if modelo:
        logging.info("Modelo satisfactorio encontrado. Generando visualización...")
        visualizador = VisualizadorNReinas(n_reinas)
        archivo_salida = 'solucion_reinas.png'
        visualizador.guardar_grafico_tablero(modelo, generador_problema, archivo_salida)
        logging.info(f"Visualización exportada exitosamente como '{archivo_salida}'.")
    else:
        logging.warning("El algoritmo alcanzó el límite de iteraciones sin encontrar una solución.")

if __name__ == "__main__":
    main()