import logging
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from problema_lpo_dominacion import ProblemaDominacionLPO


def _fuente_disponible() -> str:
    """Devuelve la primera fuente con glifos de ajedrez disponible en el sistema."""
    instaladas = {f.name for f in font_manager.fontManager.ttflist}
    for nombre in ('DejaVu Sans', 'Apple Symbols', 'Arial Unicode MS', 'Segoe UI Symbol'):
        if nombre in instaladas:
            return nombre
    return 'sans-serif'


class VisualizadorLPO:
    """Renderiza el modelo de Z3 como un tablero con reinas."""

    def __init__(self, n: int):
        self.n = n

    def guardar_grafico(self, reinas: set[tuple[int, int]], ruta_archivo: str) -> None:
        """Guarda en disco y muestra el tablero con las reinas dadas."""
        tablero = np.zeros((self.n, self.n))
        tablero[1::2, ::2] = 1
        tablero[::2, 1::2] = 1

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(tablero, cmap='gray_r', origin='upper')

        for r, c in reinas:
            color_pieza = 'white' if tablero[r, c] == 1 else 'black'
            ax.text(c, r, '\u265B', fontsize=max(8, int(250 / self.n)), ha='center', va='center',
                    color=color_pieza,
                    fontfamily=_fuente_disponible())

        ax.set_xticks(np.arange(self.n) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.n) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(f"Solución LPO (Z3): {len(reinas)} reinas (tablero {self.n}x{self.n})", pad=15)

        fig.savefig(ruta_archivo, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    n_tablero = 5
    gamma_max = 3
    independiente = False   # True: además, las reinas no pueden atacarse entre sí

    # Se prueba gamma = 1, 2, ... hasta hallar el menor número de reinas que domina el tablero.
    for gamma in range(1, gamma_max + 1):
        logging.info(f"Tablero {n_tablero}x{n_tablero}: probando con a lo sumo {gamma} reina(s)...")
        problema = ProblemaDominacionLPO(n_tablero, gamma, independiente)
        problema.formular_axiomas()

        inicio = time.perf_counter()
        modelo = problema.resolver()
        duracion = time.perf_counter() - inicio

        if modelo is None:
            logging.info(f"gamma={gamma}: INSATISFACIBLE ({duracion:.2f} s). No hay forma de dominar el tablero.")
            continue

        reinas = problema.obtener_reinas(modelo)
        valida = problema.verificar_dominacion(reinas) and (not independiente or problema.verificar_independencia(reinas))
        logging.info(f"gamma={gamma}: SATISFACIBLE ({duracion:.2f} s). Reinas en {sorted(reinas)}; "
                     f"verificación independiente: {'CORRECTA' if valida else 'INCORRECTA'}.")
        archivo = 'dominacion_lpo_solucion.png'
        VisualizadorLPO(n_tablero).guardar_grafico(reinas, archivo)
        logging.info(f"Imagen guardada como '{archivo}'.")
        return

    logging.warning(f"Ningún gamma <= {gamma_max} domina el tablero.")


if __name__ == "__main__":
    main()