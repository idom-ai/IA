import time
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sat_solver import SatCaminarSolver
from problema_lpr_reinas import ProblemaNReinasCNF


def _fuente_disponible() -> str:
    """Devuelve la primera fuente con glifos de ajedrez disponible en el sistema."""
    instaladas = {f.name for f in font_manager.fontManager.ttflist}
    for nombre in ('DejaVu Sans', 'Apple Symbols', 'Arial Unicode MS', 'Segoe UI Symbol'):
        if nombre in instaladas:
            return nombre
    return 'sans-serif'


class VisualizadorNReinas:
    """Clase encargada de renderizar la solución gráfica del problema.

    Atributos:
        n (int): Tamaño del tablero.
    """

    def __init__(self, n: int):
        """Inicializa el visualizador para un tablero de tamaño N."""
        self.n = n

    def guardar_grafico_tablero(self, modelo: dict[int, bool], problema: ProblemaNReinasCNF, ruta_archivo: str) -> None:
        """Genera, guarda en disco y muestra una imagen del tablero con las reinas.

        La reina es blanca sobre casillas negras y negra sobre casillas blancas,
        para que sea visible en todas las casillas.

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

        tam_fuente = max(8, int(250 / self.n))  # escala con el tamaño del tablero
        for i, j in problema.extraer_reinas(modelo):
            color_pieza = 'white' if tablero[i, j] == 1 else 'black'
            ax.text(j, i, '\u265B', fontsize=tam_fuente, ha='center', va='center', color=color_pieza,
                    fontfamily=_fuente_disponible())

        ax.set_xticks(np.arange(self.n) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.n) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="both", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Solución SAT-CAMINAR para un tablero {self.n}x{self.n}")

        fig.savefig(ruta_archivo, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)


def main() -> None:
    """Punto de entrada de la demostración. El tamaño del tablero se cambia aquí."""
    n_reinas = 8  # tablero N x N con N reinas

    problema = ProblemaNReinasCNF(n_reinas)
    clausulas, variables = problema.generar_base_conocimiento()
    print(f"Tablero {n_reinas}x{n_reinas}. Variables: {len(variables)} | Cláusulas: {len(clausulas)}")

    solver = SatCaminarSolver(probabilidad_aleatoria=0.4, max_iteraciones=300000)

    inicio = time.perf_counter()
    modelo = solver.resolver(clausulas, variables)
    duracion = time.perf_counter() - inicio

    if modelo:
        valida = problema.es_solucion_valida(modelo)
        print(f"Modelo encontrado en {duracion:.2f} s: {len(problema.extraer_reinas(modelo))} reinas, "
              f"verificación independiente: {'CORRECTA' if valida else 'INCORRECTA'}.")
        archivo_salida = f'solucion_reinas_{n_reinas}.png'
        VisualizadorNReinas(n_reinas).guardar_grafico_tablero(modelo, problema, archivo_salida)
        print(f"Imagen guardada como '{archivo_salida}'.")
    else:
        print("SAT-Caminar alcanzó el límite de iteraciones sin encontrar una solución.")


if __name__ == "__main__":
    main()