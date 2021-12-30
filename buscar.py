import sys
import os
import numpy
import cv2

# Verificando los inputs del comando
if len(sys.argv) < 3:
    print("Uso: {} [imagenes] [datos] [resultados]".format(sys.argv[0]))
    sys.exit(1)

imagenes = sys.argv[1]
datos_videos = sys.argv[2]
output = sys.argv[3]

if not os.path.isdir(imagenes):
    print("No existe directorio {}".format(imagenes))
    sys.exit(1)

if not os.path.isdir(datos_videos):
    print("No existe directorio {}".format(datos_videos))
    sys.exit(1)

# Calcula el SIFT de una imagen, retorna los keypoints y descriptores
def calcular_sift(imagen):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(imagen_gris)
    keypoints, descriptores = sift.compute(imagen_gris, keypoints)
    return keypoints, descriptores

# Busca la mejor homografía en la imagen
def buscar_modelo(keypoints1, keypoints2, matches, maxError, maxCiclos):
    origenes = numpy.zeros((len(matches), 2), dtype=numpy.float32)
    destinos = numpy.zeros((len(matches), 2), dtype=numpy.float32)
    for i in range(len(matches)):
        match = matches[i]
        pos_origen = keypoints1[match.queryIdx].pt
        pos_destino = keypoints2[match.trainIdx]
        origenes[i] = pos_origen
        destinos[i] = pos_destino
    M, mask = cv2.findHomography(origenes, destinos, method=cv2.RANSAC, ransacReprojThreshold=maxError, maxIters=maxCiclos)
    inliers = []
    for i in range(len(matches)):
        if mask[i] == 1:
            inliers.append(matches[i])
    return M, inliers

# Busca los matches entre descriptores
def buscar_matches(descriptores1, descriptores2, ratioAceptacion):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    lista_knn_matches = bf.knnMatch(descriptores1, descriptores2, k=2)
    buenos_matches = []
    for knn_matches in lista_knn_matches:
        match = knn_matches[0]
        match2 = knn_matches[1]
        ratio = match.distance / match2.distance
        if ratio > ratioAceptacion:
            continue
        buenos_matches.append(match)
    return buenos_matches

# Guarda los keypoints y descriptores de un video en los datos
def cargar_video(dir, video):
    # Keypoints y tiempos
    keypoints = []
    tiempos = []
    for keypoint in os.listdir(dir + "/" + video + "/keypoints"):
        # Guardaremos la refencia temporal del video
        tiempos.append(keypoint)
        with open(dir + "/" + video + "/keypoints/" + keypoint, 'rb') as file:
            d = numpy.load(file)
        keypoints.append(d)

    # Descriptores
    descriptores = []
    for descriptor in os.listdir(dir + "/" + video + "/descriptores"):
        with open(dir + "/" + video + "/descriptores/" + descriptor, 'rb') as file:
            d = numpy.load(file)
        descriptores.append(d)

    return [video, tiempos, keypoints, descriptores]

# Carga la carpeta de los datos de los videos
def cargar_carpeta(folder):
    datos = []
    for video in os.listdir(folder):
        print(f"Cargando video: {video}")
        datos.append(cargar_video(folder, video))
        print(f"{video} listo")
    return datos

# Calcula los SIFT de las imagenes
def carga_imagenes(imagenes):
    # Creamos los datos de las imágenes
    imagen_data = []
    for imagen in os.listdir(imagenes):
        print(f"Cargando imagen: {imagen}")
        keypoints, descriptores = calcular_sift(cv2.imread(imagenes + "/" + imagen))
        imagen_data.append([imagen, keypoints, descriptores])
        print(f"{imagen} lista")
    return imagen_data

def busqueda(imagen_data, video_data, output, ratioAceptacion=0.9, maxError=1, maxCiclos=100):
    resultados = []
    # Buscamos para cada imagen
    for i in imagen_data:
        print(f"Buscando imagen: {i[0]}")
        candidatos = []
        distancias = []
        # Revisamos en cada video
        for v in video_data:
            # Iteramos por cada frame del video
            for t, k, d in zip(v[1], v[2], v[3]):
                # Encontramos los matches entre las imágenes y los frames
                matches = buscar_matches(i[2], d, ratioAceptacion)
                # Si hay los suficientes matches encontramos la homografía, sino dejamos con 0 inliers la búsqueda
                if len(matches) >= 4:
                    modelo, inliers = buscar_modelo(i[1], k, matches, maxError, maxCiclos)
                    inliers = len(inliers)
                else:
                    inliers = 0
                # Función de distancia
                dist = 1 - (inliers / len(i[1]))

                candidatos.append([i[0], v[0], t])
                distancias.append(dist)
        # Ordenamos los candidatos en función de su distancia
        distancias, candidatos = zip(*sorted(zip(distancias, candidatos)))

        for c, d in zip(candidatos[:4], distancias[:4]):
            # Tiempos a segundos y minutos
            s = int((int(c[2]) / 1000) % 60)
            m = int((int(c[2]) / 1000) / 60)
            # Imagen Video Minuto Segundo Distancia
            result = [c[0], c[1], m, s, d]
            resultados.append(result)
    # Escribimos en resultados
    numpy.savetxt(output, resultados, delimiter='\t', fmt="%s")


# Cargamos los videos
video_data = cargar_carpeta(datos_videos)

# Cargamos la imagen
imagen_data = carga_imagenes(imagenes)

# Busqueda de las imagenes en los videos
busqueda(imagen_data, video_data, output, ratioAceptacion=0.9, maxError=1, maxCiclos=100)