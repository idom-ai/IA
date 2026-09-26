import logging
import matplotlib.pyplot as plt
import numpy as np
from problema_lpo_reinas import ProblemaDominacionLPO

class VisualizadorLPO:
    """Procesa el modelo matematico de Z3 y renderiza el resultado grafico con figuras de ajedrez."""

    def __init__(self, n: int):
        self.n = n

    def guardar_grafico(self, modelo, problema: ProblemaDominacionLPO, ruta_archivo: str) -> None:
        tablero = np.zeros((self.n, self.n))
        tablero[1::2, ::2] = 1
        tablero[::2, 1::2] = 1

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(tablero, cmap='gray_r', origin='upper')

        posiciones_unicas = set()
        for r_var, c_var in problema.variables_posicion:
            r_val = modelo.evaluate(r_var).as_long()
            c_val = modelo.evaluate(c_var).as_long()
            posiciones_unicas.add((r_val, c_val))

        # Renderizado de la reina (Unicode U+265B) forzando fuentes compatibles en macOS
        for r, c in posiciones_unicas:
            color_pieza = 'white' if tablero[r, c] == 1 else 'black'
            ax.text(c, r, '\u265B', fontsize=48, ha='center', va='center', 
                    color=color_pieza, fontfamily=['Apple Symbols', 'Arial Unicode MS', 'sans-serif'])

        ax.set_xticks(np.arange(self.n) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.n) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        
        cantidad_reinas = len(posiciones_unicas)
        ax.set_title(f"Solucion LPO (Z3): {cantidad_reinas} Reinas (Tablero {self.n}x{self.n})", pad=15)

        plt.show()
        
def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    n_tablero = 5
    gamma_reinas = 3
    
    logging.info(f"Iniciando construccion LPO para tablero {n_tablero}x{n_tablero} con maximo {gamma_reinas} reinas.")
    
    problema = ProblemaDominacionLPO(n_tablero, gamma_reinas)
    problema.formular_axiomas()
    
    logging.info("Ejecutando motor de demostracion Z3...")
    
    modelo = problema.resolver()
    
    if modelo:
        logging.info("Demostracion exitosa. Generando imagen...")
        visualizador = VisualizadorLPO(n_tablero)
        visualizador.guardar_grafico(modelo, problema, 'dominacion_lpo_solucion.png')
        logging.info("Guardado como 'dominacion_lpo_solucion.png'. Revisa el archivo.")
    else:
        logging.warning("Insatisfacible.")

if __name__ == "__main__":
    main()