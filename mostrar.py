import os, sys
import numpy as np
import cv2

# Verificando los inputs del comando
if len(sys.argv) < 1:
    print("Uso: {} [txt]".format(sys.argv[0]))
    sys.exit(1)


imagenes = sys.argv[1]
videos = sys.argv[2]
txt = sys.argv[3]

if not os.path.isdir(imagenes):
    print("No existe directorio {}".format(imagenes))
    sys.exit(1)

if not os.path.isdir(videos):
    print("No existe directorio {}".format(videos))
    sys.exit(1)

def mostrar4(imagenes, videos, resultados):
    w = 384
    h = 216
    i = 0
    while i < len(resultados):
        name = resultados[i][0]
        imagen = cv2.imread(imagenes + "/" + name)
        imagen = cv2.resize(imagen, (w*2, h*2))
        # Carga primer candidato
        v = cv2.VideoCapture(videos + "/" + resultados[i][1])
        v.set(cv2.CAP_PROP_POS_MSEC, 1000*((int(resultados[i][2]) * 60) + int(resultados[i][3])))
        success, candidato1 = v.read()
        i+=1
        # Carga segundo candidato
        v = cv2.VideoCapture(videos + "/" + resultados[i][1])
        v.set(cv2.CAP_PROP_POS_MSEC, 1000*((int(resultados[i][2]) * 60) + int(resultados[i][3])))
        success, candidato2 = v.read()
        i+=1
        # Carga tercer candidato
        v = cv2.VideoCapture(videos + "/" + resultados[i][1])
        v.set(cv2.CAP_PROP_POS_MSEC, 1000*((int(resultados[i][2]) * 60) + int(resultados[i][3])))
        success, candidato3 = v.read()
        i+=1
        # Carga cuarto candidato
        v = cv2.VideoCapture(videos + "/" + resultados[i][1])
        v.set(cv2.CAP_PROP_POS_MSEC, 1000*((int(resultados[i][2]) * 60) + int(resultados[i][3])))
        success, candidadto4 = v.read()
        i+=1
        # Unir imagenes
        h1 = np.hstack((cv2.resize(candidato1,(w,h)), cv2.resize(candidato2,(w,h))))
        h2 = np.hstack((cv2.resize(candidato3,(w,h)), cv2.resize(candidadto4,(w,h))))
        candidatos = np.vstack((h1,h2))
        comparacion = np.hstack((imagen, candidatos))

        cv2.imshow(name, comparacion)
    cv2.waitKey()
    cv2.destroyAllWindows()


resultados = np.loadtxt(txt, dtype='str')
mostrar4(imagenes, videos, resultados)

