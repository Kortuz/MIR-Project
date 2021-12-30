import sys
import os.path
import numpy as np
import cv2

# Verificando los inputs del comando
if len(sys.argv) < 2:
    print("Uso: {} [videos] [datos]".format(sys.argv[0]))
    sys.exit(1)

videos = sys.argv[1]
datos = sys.argv[2]

if not os.path.isdir(videos):
    print("No existe directorio {}".format(videos))
    sys.exit(1)

if not os.path.isdir(datos):
    print("Creando directorio {}".format(datos))
    os.mkdir(datos)


# Calcula el SIFT de un frame, retorna los keypoints y descriptores
def calcular_sift(imagen):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(imagen_gris)
    keypoints, descriptores = sift.compute(imagen_gris, keypoints)
    return keypoints, descriptores

# Calcula los keypoints y descriptores de los frames de un video
def procesar_video(datos_dir, video_dir, video, step=500, frames=False):
    vidcap = cv2.VideoCapture(video_dir + "/" + video)
    success, image = vidcap.read()
    count = 0

    # Creamos la carpeta en donde guardamos los keypoints de cada imagen
    if not os.path.isdir(datos_dir + "/" + video + "/keypoints"):
        print("Creando directorio {}".format(datos_dir + "/" + video + "/keypoints"))
        os.mkdir(datos_dir + "/" + video + "/keypoints")
        print("Carpeta keypoints creada")

    # Creamos la carpeta en donde guardamos los descriptores de cada imagen
    if not os.path.isdir(datos_dir + "/" + video + "/descriptores"):
        print("Creando directorio {}".format(datos_dir + "/" + video + "/descriptores"))
        os.mkdir(datos_dir + "/" + video + "/descriptores")
        print("Carpeta descriptores creada")

    # Creamos la carpeta en donde guardamos los frames de cada imagen (si se requiere)
    if not os.path.isdir(datos_dir + "/" + video + "/frames") and frames:
        print("Creando directorio {}".format(datos_dir + "/" + video + "/frames"))
        os.mkdir(datos_dir + "/" + video + "/frames")
        print("Carpeta frames creada")

    while success:
        # Guardamos los frames si se requiere
        if frames:
            cv2.imwrite(datos_dir + "/" + video + "/frames/frame%d.jpg" % (count * step), image)

        # Calculamos SIFT del frame
        keypoints, descriptores = calcular_sift(image)

        # Guardamos los keypoints en una lista para guardar en archivo
        ks = []
        for k in keypoints:
            ks.append([k.pt[0], k.pt[1]])
        ks = np.array(ks)

        # Escribimos los keypoints en un binario con el nombre del archivo en la carpeta keypoints
        with open(datos_dir + "/" + video + "/keypoints/%d" % (count * step), "wb") as file:
            np.save(file, ks)

        # Escribimos los descriptores en un binario con el nombre del archivo
        with open(datos_dir + "/" + video + "/descriptores/%d" % (count * step), "wb") as file:
            np.save(file, descriptores)

        # Avanzamos frames y capturamos la siguiente imagen
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * step))
        success, image = vidcap.read()
        count += 1

# Procesamos cada video en la carpeta
for video in os.listdir(videos):
    # Creamos la arpeta para cada video
    if not os.path.isdir(datos + "/" + video):
        print("Creando directorio {}".format(datos + "/" + video))
        os.mkdir(datos + "/" + video)

    procesar_video(datos, videos, video, frames=True)







