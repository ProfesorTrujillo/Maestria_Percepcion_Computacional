"""
Tarea 02 - 2.1 Manipulación de Imágenes
Percepción Computacional
Autor: Francisco Javier Trujillo Silva
Fecha: 09/04/2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime


class ProcesadorImagen:
    """
    Clase que encapsula las operaciones de manipulación de imágenes
    """

    def __init__(self, ruta_imagen, nombre_autor):
        self.imagen_color = cv2.imread(ruta_imagen)

        if self.imagen_color is None:
            raise FileNotFoundError(
                f"No se pudo cargar la imagen: {ruta_imagen}"
            )

        self.imagen_gris = cv2.cvtColor(self.imagen_color, cv2.COLOR_BGR2GRAY)
        self.nombre_autor = nombre_autor
        self.fecha = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if not os.path.exists("resultados"):
            os.makedirs("resultados")

    def agregar_marca_agua(self, imagen):
        texto = f"{self.nombre_autor} - {self.fecha}"
        imagen_marca = imagen.copy()

        cv2.putText(
            imagen_marca,
            texto,
            (10, imagen_marca.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        return imagen_marca

    def histograma(self):
        hist = cv2.calcHist([self.imagen_gris], [0], None, [256], [0, 256])

        plt.figure()
        plt.plot(hist, color="black")
        plt.title("Histograma de la imagen")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")
        plt.grid()
        plt.savefig("resultados/histograma.png")
        plt.close()

    def espejo(self):
        imagen = self.agregar_marca_agua(cv2.flip(self.imagen_gris, 1))
        cv2.imwrite("resultados/espejo.png", imagen)

    def rotaciones(self):
        cv2.imwrite("resultados/rotacion_90.png",
                     self.agregar_marca_agua(cv2.rotate(self.imagen_gris, cv2.ROTATE_90_CLOCKWISE)))
        cv2.imwrite("resultados/rotacion_180.png",
                     self.agregar_marca_agua(cv2.rotate(self.imagen_gris, cv2.ROTATE_180)))
        cv2.imwrite("resultados/rotacion_270.png",
                     self.agregar_marca_agua(cv2.rotate(self.imagen_gris, cv2.ROTATE_90_COUNTERCLOCKWISE)))

    def negativo(self):
        imagen = self.agregar_marca_agua(255 - self.imagen_gris)
        cv2.imwrite("resultados/negativo.png", imagen)

    def contraste(self, alpha=1.5, beta=0):
        imagen = self.agregar_marca_agua(
            cv2.convertScaleAbs(self.imagen_gris, alpha=alpha, beta=beta))
        cv2.imwrite("resultados/contraste.png", imagen)

    def desenfoque_gaussiano(self):
        imagen = self.agregar_marca_agua(
            cv2.GaussianBlur(self.imagen_gris, (7, 7), 0))
        cv2.imwrite("resultados/gaussiano.png", imagen)


def seleccionar_imagen():
    carpeta = "imagenes"
    extensiones_validas = (".jpg", ".png", ".jpeg", ".bmp")

    imagenes = [f for f in os.listdir(carpeta)
                if f.lower().endswith(extensiones_validas)]

    if not imagenes:
        raise FileNotFoundError("No hay imágenes en la carpeta 'imagenes'")

    print("\nImágenes disponibles:")
    for i, img in enumerate(imagenes):
        print(f"[{i + 1}] {img}")

    opcion = int(input("\nSeleccione el número de la imagen: ")) - 1

    if opcion < 0 or opcion >= len(imagenes):
        raise ValueError("Selección inválida")

    return os.path.join(carpeta, imagenes[opcion])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autor",
        default="Francisco Javier Trujillo Silva",
        help="Nombre del autor para la marca de agua"
    )
    args = parser.parse_args()

    ruta_imagen = seleccionar_imagen()
    procesador = ProcesadorImagen(ruta_imagen, args.autor)

    procesador.histograma()
    procesador.espejo()
    procesador.rotaciones()
    procesador.negativo()
    procesador.contraste()
    procesador.desenfoque_gaussiano()

    print("\n✅ Procesamiento finalizado correctamente.")
    print("📁 Resultados guardados en la carpeta 'resultados'")
if __name__ == "__main__":
    main()