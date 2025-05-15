import numpy as np
import cv2 
import matplotlib.pyplot as plt
import sys
import random

## Tenemos que sacar el ransac para obtener la matriz fundamental:
    # Calculamos H para epoder estimar el error (mirar código pract panorama)
    # Ahora podemos sacar las correspondencias y podremos obtener D_high solo con las mejores

# Matriz de proyección
P = np.array([
    [1541.24, 0, 993.53, 0],  
    [0, 1538.17, 757.98, 0],  
    [0, 0, 1, 0]   
])

# Función que nos descompone P
def krt_descomposition(P):
    # Aplicamos la descomposicón QR
    Q, U = np.linalg.qr(np.linalg.inv(P[:3, :3]))
    print(f'resultado 1 Q = {Q}')
    print(f'resultado 1 U = {U}')
    #D = np.diag(np.sign(np.diag(U)) * np.array([[-1, 0, 0],
                                                #[0, -1, 0],
                                                #[0, 0, 1]]))
    #print(f'resultado 1 D = {D}')
    #Q = Q * D
    print(f'resultado 2 Q = {Q}')
    #U = D * U
    print(f'resultado 2 U = {U}')
    
    # Obtenemos s para forzar que la matriz de rotación R resultante 
    # tenga determinante positivo, ya que es una forma de garantizar que la rotación sea pura
    s = np.linalg.det(Q)
    print(f'resultado s = {s}')

    ## Calculamos los parámetros extrínsecos de la cámara
    # Matriz de rotación
    R = s * np.transpose(Q)
    print(f'resultado R = {R}')

    # Vector de traslación
    t = s*U*P[:3,-1]
    print(f'resultado t = {t}')

    ## Calculamos los parámetros intrínsecos de la cámara
    # Matriz K 
    K = np.linalg.inv(U/U[2,2])
    print(f'resultado K = {K}')
    return K, R, t




## Obtenemos las correspondencias
def correspendencias(img_l,img_d):
    # Creamos un SIFT detector
    sift = cv2.SIFT_create()

    # Detectamos los puntos clave
    puntos_clave_cv_l, descriptores_l = sift.detectAndCompute(img_l, None)
    puntos_clave_cv_d, descriptores_d = sift.detectAndCompute(img_d, None)

    puntos_clave_l = np.array([kp.pt for kp in puntos_clave_cv_l], dtype=np.float32)
    puntos_clave_d = np.array([kp.pt for kp in puntos_clave_cv_d], dtype=np.float32)
    
    print("Coordenadas img_l:", puntos_clave_l[:5])  
    print("Coordenadas img_d:", puntos_clave_d[:5])
    
    # Dibujamos los puntos clave en la imagen
    img_puntos_clave_l = cv2.drawKeypoints(
        img_l, puntos_clave_cv_l, None,
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img_puntos_clave_d = cv2.drawKeypoints(
        img_d, puntos_clave_cv_d, None,
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("SIFT puntos clave img_l")
    plt.imshow(cv2.cvtColor(img_puntos_clave_l, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("SIFT puntos clave img_d")
    plt.imshow(cv2.cvtColor(img_puntos_clave_d, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return puntos_clave_l, puntos_clave_d 

    
def construir_matriz_A(points_l, points_d):
    A = []
    for (x1, y1), (x2, y2) in zip(points_l, points_d):
        A.append([
            x1 * x2, y1 * x2, x2,
            x1 * y2, y1 * y2, y2,
            x1, y1, 1
        ])
    return np.array(A)

def eight_point_algorithm(points_l, points_d, T1, T2):
    A = construir_matriz_A(points_l, points_d)
    #print("Forma de A:", A.shape)
    #print("Valores de A:", A[:5])
    U,S,Vt = np.linalg.svd(A)
    V = Vt.T
    z = V[-1]
    F_vec = z
    F = np.reshape(F_vec, (3, 3)) 
    F = F / F[2,2]

    Uf, Sf, Vtf = np.linalg.svd(F)
    #print("Valores singulares de F antes de forzar rango 2:", Sf)
    Sf[-1] = 0  # anular el menor valor singular
    F_rank2 = Uf @ np.diag(Sf) @ Vtf 
    F_denorm = T2.T @ F_rank2 @ T1
    F_denorm = F_denorm/F_denorm[2,2]
    #print("Matriz fundamental antes de desnormalizar:", F_rank2)
    #print("Matriz fundamental después de desnormalizar:", F_denorm)



    return F_denorm


def normalizar_puntos(puntos):
     # Calcular el centroide
    centroide = np.mean(puntos, axis=0)
    # Desplazar los puntos al centroide
    puntos_desplazados = puntos - centroide
    # Calcular la distancia media desde el centroide
    distancia_media = np.mean(np.sqrt(np.sum(puntos_desplazados**2, axis=1)))
    # Escalar para que la distancia distancia_mediab sea sqrt(2)
    factor_escala = np.sqrt(2) / distancia_media
    # Matriz de transformación
    T = np.array([[factor_escala, 0, -factor_escala * centroide[0]],
                  [0, factor_escala, -factor_escala * centroide[1]],
                  [0, 0, 1]])
    # Aplicar la transformación a los puntos
    puntos_homogeneos = np.hstack((puntos, np.ones((puntos.shape[0], 1))))
    puntos_normalizados = (T @ puntos_homogeneos.T).T
    return puntos_normalizados[:, :2], T
 

def estima_error_1(puntos_l, puntos_d, F):
    #Pasamos a homogeneo
    recta_l = F.T * puntos_l
    recta_d = F       * puntos_d
    dpd_l = abs(normalizar_puntos(puntos_l) * normalizar_puntos(recta_l))
    dpd_d = abs(normalizar_puntos(puntos_d) * normalizar_puntos(recta_d))
    epsilon = dpd_l + dpd_d
    return epsilon

def estima_error(puntos_l, puntos_d, F):
    # Convertimos los puntos a coordenadas homogéneas
    puntos_l_h = np.hstack((puntos_l, np.ones((puntos_l.shape[0], 1))))
    puntos_d_h = np.hstack((puntos_d, np.ones((puntos_d.shape[0], 1))))

    # Multiplicamos correctamente con la matriz fundamental
    recta_l = F.T @ puntos_d_h.T  # (3x3) @ (3xN) → (3xN)
    recta_d = F @ puntos_l_h.T    # (3x3) @ (3xN) → (3xN)

    # Normalización de líneas epipolares
    recta_l = recta_l / np.linalg.norm(recta_l[:2], axis=0)
    recta_d = recta_d / np.linalg.norm(recta_d[:2], axis=0)

    # Calculamos la distancia punto-línea epipolar
    dpd_l = np.abs(np.sum(recta_l.T * puntos_l_h, axis=1)) #PREGUNTAR COMO VA ESTO DE DPd
    dpd_d = np.abs(np.sum(recta_d.T * puntos_d_h, axis=1))

    # Error epipolar total
    epsilon = np.mean(dpd_l + dpd_d)
    #print(epsilon)
    return epsilon

## Calculamos la matriz fundamental (F)

def ransac(puntos_clave_l, puntos_clave_d, iter, t):
    F_estimada = None
    C_est = []
    C = []
    max_inliers = 0
    #print(len(puntos_clave_l))
    #print(len(puntos_clave_d))
    puntos_normalizados_l, T1 = normalizar_puntos(puntos_clave_l)
    puntos_normalizados_d, T2 = normalizar_puntos(puntos_clave_d)
    #print(len(puntos_normalizados_l))
    #print(len(puntos_normalizados_d))
    #print("Coordenadas puntos_normalizados_l:", puntos_normalizados_l[:5])  
    #print("Coordenadas puntos_normalizados_d:", puntos_normalizados_d[:5])
    print("Empezamos RANSAC")
    for _ in range(iter):
        idx = random.sample(range(len(puntos_normalizados_d)), 8)
        sample_l = puntos_normalizados_l[idx]
        #print("8 puntos L")
        #print(sample_l)
        sample_d = puntos_normalizados_d[idx]
        #print("8 puntos D")
        #print(sample_d)
        #min_puntos = min(len(puntos_normalizados_l), len(puntos_normalizados_d))
        #puntos_normalizados_l = puntos_normalizados_l[:min_puntos]
        #puntos_normalizados_d = puntos_normalizados_d[:min_puntos]
        F = eight_point_algorithm(sample_l, sample_d, T1, T2) #poner aqui dentro la normalizacion de puntos
        inliers = 0
        for i in range(len(puntos_clave_l)):
            error = estima_error(sample_l, sample_d, F)
            #print(error)
            if error < t:
                inliers += 1
                C.append((sample_l, sample_d))

        if len(C) > len(C_est):
            C_est = np.array(C)
            C_est_np = np.array(C_est, dtype=object)
            F_est = F
            C = []
            if inliers > max_inliers:
                max_inliers = inliers
    print("C_est =")
    print(C_est_np)
    print("F_est =")
    print(F_est)
    #print("Matriz de transformación T1:", T1)
    #print("Matriz de transformación T2:", T2)
    print("Terminamos RANSAC")
    return F_est, C_est_np
        

def dibujar_matching(img_l, img_d, puntos_l, puntos_d, F):
    """
    Dibuja líneas epipolares entre los puntos clave de img_l e img_d usando la matriz fundamental F.
    """
    # Convertimos imágenes a formato RGB
    img_l_rgb = np.flip(img_l, axis=-1)  # Invertir el canal de color si es necesario
    img_d_rgb = np.flip(img_d, axis=-1)

    # Concatenamos ambas imágenes horizontalmente
    altura_max = max(img_l.shape[0], img_d.shape[0])
    ancho_total = img_l.shape[1] + img_d.shape[1]

    fig, ax = plt.subplots(figsize=(12, 6))
    img_combinada = np.zeros((altura_max, ancho_total, 3), dtype=np.uint8)
    img_combinada[:img_l.shape[0], :img_l.shape[1]] = img_l_rgb
    img_combinada[:img_d.shape[0], img_l.shape[1]:] = img_d_rgb

    ax.imshow(img_combinada)

    # Dibujar líneas epipolares
    for punto_l, punto_d in zip(puntos_l, puntos_d):
        # Convertimos los puntos a coordenadas homogéneas
        punto_l_h = np.append(punto_l, 1)
        punto_d_h = np.append(punto_d, 1)

        # Calcular la línea epipolar en la segunda imagen
        linea_epipolar = F @ punto_l_h  # F * punto en la imagen izquierda

        # Calcular puntos extremos de la línea epipolar
        x = np.linspace(0, img_d.shape[1], num=2)
        y = -(linea_epipolar[0] * x + linea_epipolar[2]) / linea_epipolar[1]

        # Ajustar al ancho total (imagen derecha está desplazada)
        x += img_l.shape[1]

        ax.plot([punto_l[0], punto_d[0] + img_l.shape[1]], [punto_l[1], punto_d[1]], color="lime", linewidth=0.8)
        ax.plot(x, y, linestyle="--", color="red", linewidth=0.7)  # Línea epipolar roja

    # Configuración de la visualización
    ax.set_title("Líneas epipolares y correspondencias")
    ax.axis("off")
    plt.show()



def main():
    img_l = cv2.imread('im_i.jpeg', cv2.IMREAD_COLOR)
    img_d = cv2.imread('im_d.jpeg', cv2.IMREAD_COLOR)
    krt_descomposition(P)
    puntos_clave_l, puntos_clave_d = correspendencias(img_l,img_d)
    r = 150
    t = 700
    F, puntos = ransac(puntos_clave_l, puntos_clave_d, r, t)
    print(f"Puntos: {puntos[:40]}")
    # Verificar si F_est satisface la ecuación epipolar

    # Separar todos los puntos en dos listas distintas
    puntos_l_list, puntos_d_list = zip(*puntos)

    puntos_l = np.vstack(puntos_l_list)  # Apila todos los puntos en una sola estructura
    puntos_d = np.vstack(puntos_d_list)  # Hace lo mismo para los puntos de la segunda imagen

    print(f"Puntos L: {puntos_l[:9]}")
    print(f"Puntos D: {puntos_d[:9]}")

    puntos_l_h = np.hstack((puntos_l, np.ones((puntos_l.shape[0], 1))))
    puntos_d_h = np.hstack((puntos_d, np.ones((puntos_d.shape[0], 1))))

    #print("Forma de puntos_l_h:", puntos_l_h.shape)
    #print("Forma de puntos_d_h:", puntos_d_h.shape)


    # Imprimir las primeras 5 filas de cada uno
    print("Primeras 5 filas de puntos_l:", puntos_l[:5])
    print("Primeras 5 filas de puntos_d:", puntos_d[:5])

    errores_epipolares = [
    np.abs(p_d_h.T @ F @ p_l_h) for p_l_h, p_d_h in zip(puntos_l_h[:50], puntos_d_h[:50])
    ]
    print("Error epipolar promedio:", np.mean(errores_epipolares))

    dibujar_matching(img_l, img_d, puntos_l, puntos_d, F)


if __name__ == '__main__':
    main()