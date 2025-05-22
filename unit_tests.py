import numpy as np
import cv2 
import matplotlib.pyplot as plt
import sys
import random

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

    # Vector de traslación
    t = s*U*P[:3,-1]

    ## Calculamos los parámetros intrínsecos de la cámara
    # Matriz K 
    K = np.linalg.inv(U/U[2,2])
    return K, R, t

def reconstruir_P(K, R, t):
    return K @ np.hstack((R, t[:,2].reshape(-1, 1)))

def correspendencias(img_l,img_d):
    # Creamos un SIFT detector
    sift = cv2.SIFT_create()

    # Detectamos los puntos clave
    puntos_clave_cv_l, descriptores_l = sift.detectAndCompute(img_l, None)
    puntos_clave_cv_d, descriptores_d = sift.detectAndCompute(img_d, None)

    puntos_clave_l = np.array([kp.pt for kp in puntos_clave_cv_l], dtype=np.float32)
    puntos_clave_d = np.array([kp.pt for kp in puntos_clave_cv_d], dtype=np.float32)
    
    # Dibujamos los puntos clave en la imagen
    img_puntos_clave_l = cv2.drawKeypoints(
        img_l, puntos_clave_cv_l, None,
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img_puntos_clave_d = cv2.drawKeypoints(
        img_d, puntos_clave_cv_d, None,
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return puntos_clave_l, puntos_clave_d, img_puntos_clave_l, img_puntos_clave_d

def plot_correspondencias(img_puntos_clave_l, img_puntos_clave_d):
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

def estima_error(puntos_l, puntos_d, M):
    # Convertimos los puntos a coordenadas homogéneas
    puntos_l_h = np.hstack((puntos_l, np.ones((puntos_l.shape[0], 1))))
    puntos_d_h = np.hstack((puntos_d, np.ones((puntos_d.shape[0], 1))))

    # Multiplicamos correctamente con la matriz fundamental
    recta_l = M.T @ puntos_d_h.T  # (3x3) @ (3xN) → (3xN)
    recta_d = M @ puntos_l_h.T    # (3x3) @ (3xN) → (3xN)

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

def ransac(puntos_clave_l, puntos_clave_d, iter, t):

    def eight_point_algorithm(points_l, points_d, T1, T2):

        def construir_matriz_A(points_l, points_d):
            A = []
            for (x1, y1), (x2, y2) in zip(points_l, points_d):
                A.append([
                    x1 * x2, y1 * x2, x2,
                    x1 * y2, y1 * y2, y2,
                    x1, y1, 1
                ])
            return np.array(A)

        A = construir_matriz_A(points_l, points_d)
        U,S,Vt = np.linalg.svd(A)
        V = Vt.T
        z = V[-1]
        F_vec = z
        F = np.reshape(F_vec, (3, 3)) 
        #Normalizamos F
        F = F / F[2,2]

        Uf, Sf, Vtf = np.linalg.svd(F)

        Sf[-1] = 0  # anular el menor valor singular
        F_rank2 = Uf @ np.diagflat(Sf) @ Vtf
        F_denorm = T2.T @ F_rank2 @ T1    
        F_denorm = F_denorm/F_denorm[2,2] # PREGUNTAR EN CLASE por que denormalizar y normalizar.
        #F_denorm = F_rank2
        
        return F_denorm

    C_est = []
    C = []
    max_inliers = 0
    best_error = np.inf

    puntos_normalizados_l, T1 = normalizar_puntos(puntos_clave_l)
    puntos_normalizados_d, T2 = normalizar_puntos(puntos_clave_d)

    print("Empezamos RANSAC")
    for _ in range(iter):
        if len(puntos_clave_d) > len(puntos_clave_l):
            idx = random.sample(range(len(puntos_clave_l)), 8)
        else:
            idx = random.sample(range(len(puntos_clave_d)), 8)
        sample_l = puntos_clave_l[idx]
        sample_d = puntos_clave_d[idx]

        sample_l, T_1 = normalizar_puntos(sample_l)
        sample_d, T_2 = normalizar_puntos(sample_d)

        F = eight_point_algorithm(sample_l, sample_d, T_1, T_2) #poner aqui dentro la normalizacion de puntos
        inliers = 0
        C = []
        
        for i, (pl_n, pd_n) in enumerate(zip(puntos_normalizados_l, puntos_normalizados_d)):
            error = estima_error(np.array([pl_n]), np.array([pd_n]), F)
            if error < t:
                inliers += 1
                #if (tuple(puntos_clave_l[i]), tuple(puntos_clave_d[i])) not in C:
                #   C.append((tuple(puntos_clave_l[i]), tuple(puntos_clave_d[i])))
                pl = tuple(map(float, puntos_clave_l[i]))
                pd = tuple(map(float, puntos_clave_d[i]))
                C.append((pl, pd))

        # if len(C) > len(C_est) and best_error > error:
        if len(C) > len(C_est):
            print("Mejor error hasta el momento:", error)
            best_error = error
            #C_est = np.array(C)
            #C_est_np = (np.array(C_est, dtype=object)).copy()
            C_est_np = (np.array(C)).copy()
            F_est = F
            if inliers > max_inliers:
                max_inliers = inliers
    #Filtrar luego para quedarse con las rectas con la orientación mas similar / común
    print("C_est =")
    print(C_est_np)
    print("F_est =")
    print(F_est)
    print("Terminamos RANSAC")
    return F_est, C_est_np

def angle_bin(v, num_bins=36):
        angle = np.arctan2(v[1], v[0])
        bin_index = int(((angle + np.pi) / (2 * np.pi)) * num_bins)
        return bin_index

def dibujar_puntos_y_lineas(img_l, img_d, puntos_l, puntos_d, num):

    img_l_rgb = img_l[:, :, ::-1].copy()
    img_d_rgb = img_d[:, :, ::-1].copy()

    altura = max(img_l.shape[0], img_d.shape[0])
    ancho_total = img_l.shape[1] + img_d.shape[1]
    img_combinada = np.zeros((altura, ancho_total, 3), dtype=np.uint8)
    img_combinada[:img_l.shape[0], :img_l.shape[1]] = img_l_rgb
    img_combinada[:img_d.shape[0], img_l.shape[1]:] = img_d_rgb

    total = len(puntos_l)
    mitad = num // 2
    centro = total // 2
    ini = max(0, centro - mitad)
    fin = min(total, centro + mitad)

    vectores = [np.array(p_d) - np.array(p_l) for p_l, p_d in zip(puntos_l[ini:fin], puntos_d[ini:fin])]
    bins = np.array([angle_bin(v) for v in vectores])
    valores, cuentas = np.unique(bins, return_counts=True)
    bin_comun = valores[np.argmax(cuentas)]

    plt.figure(figsize=(14, 6))
    plt.imshow(img_combinada)
    ax = plt.gca()

    for i, v in enumerate(vectores):
        if angle_bin(v) == bin_comun:
            x_l, y_l = puntos_l[ini + i]
            x_d, y_d = puntos_d[ini + i]
            ax.plot(x_l, y_l, 'go', markersize=4)
            ax.plot(x_d + img_l.shape[1], y_d, 'bo', markersize=4)
            ax.plot([x_l, x_d + img_l.shape[1]], [y_l, y_d], 'y-', linewidth=0.7)

    ax.set_title(f"{fin - ini} correspondencias centradas (orientación más común)")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def encontrar_mejor_punto(puntos_l, puntos_d):
    total = len(puntos_l)
    num = len(puntos_l)
    mitad = num // 2
    centro = total // 2
    ini = max(0, centro - mitad)
    fin = min(total, centro + mitad)
    vectores = [np.array(p_d) - np.array(p_l) for p_l, p_d in zip(puntos_l[ini:fin], puntos_d[ini:fin])]
    bins = np.array([angle_bin(v) for v in vectores])
    valores, cuentas = np.unique(bins, return_counts=True)
    bin_comun = valores[np.argmax(cuentas)]
    almacen_l = np.empty((0, 3))  # Matriz vacía con 3 columnas
    almacen_d = np.empty((0, 3))
    for i, v in enumerate(vectores):
        if angle_bin(v) == bin_comun:
            x_l, y_l = puntos_l[ini + i]
            x_d, y_d = puntos_d[ini + i]
            almacen_l = np.vstack([almacen_l, np.array([x_l, y_l, 1])])
            almacen_d = np.vstack([almacen_d, np.array([x_d, y_d, 1])])
    return almacen_l, almacen_d

def calcular_matriz_E(F,K):
    K_np = np.array(K)
    K_trans = K_np.T
    E = K_trans @ F @ K
    return E

def matching_puntos_E(puntos_clave_l, puntos_clave_d, E, t):
    C_est = []
    C = []
    puntos_normalizados_l, T1_e = normalizar_puntos(puntos_clave_l)
    puntos_normalizados_d, T2_e = normalizar_puntos(puntos_clave_d)

    for i, (pl_n, pd_n) in enumerate(zip(puntos_normalizados_l, puntos_normalizados_d)):
            error = estima_error(np.array([pl_n]), np.array([pd_n]), E)
            if error < t:
                if (tuple(puntos_clave_l[i]), tuple(puntos_clave_d[i])) not in C:
                    C.append((tuple(puntos_clave_l[i]), tuple(puntos_clave_d[i])))

    if len(C) > len(C_est):
        print("Mejor error hasta el momento:", error)
        best_error = error
        C_est = np.array(C)
        C_est_np = np.array(C_est, dtype=object)
        E_est = E
    return E_est, C_est_np

def visualizar_lineas_epipolares(img_d, img_i, l_izq, l_der, punto_izq, punto_der):
    """
    Visualiza las líneas epipolares en ambas imágenes a partir de los vectores de línea y puntos dados.
    """

    def dibujar_linea(ax, line, shape, color='green'):
        a, b, c = line
        height, width = shape[:2]

        if abs(b) > 1e-6:
            x_vals = np.array([0, width])
            y_vals = -(a * x_vals + c) / b
        else:
            y_vals = np.array([0, height])
            x_vals = -(b * y_vals + c) / a

        ax.plot(x_vals, y_vals, color=color, linestyle='--', linewidth=1)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invertir eje Y

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(img_i, cmap='gray')
    axs[0].scatter(*punto_izq[:2], color='red', label='Punto Izquierda')
    dibujar_linea(axs[0], l_izq, img_i.shape)
    axs[0].set_title('Imagen Izquierda')
    axs[0].axis('off')
    axs[0].legend()

    axs[1].imshow(img_d, cmap='gray')
    axs[1].scatter(*punto_der[:2], color='blue', label='Punto Derecha')
    dibujar_linea(axs[1], l_der, img_d.shape)
    axs[1].set_title('Imagen Derecha')
    axs[1].axis('off')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def visualizar_lineas_epipolares_listas(img_d, img_i, lineas_izq, lineas_der, puntos_izq, puntos_der):
    """
    Visualiza múltiples líneas epipolares en ambas imágenes a partir de listas de líneas y puntos.

    Parámetros:
    - img_d: Imagen derecha.
    - img_i: Imagen izquierda.
    - lineas_izq: Lista de líneas epipolares en la imagen izquierda.
    - lineas_der: Lista de líneas epipolares en la imagen derecha.
    - puntos_izq: Lista de puntos (x, y, 1) en la imagen izquierda.
    - puntos_der: Lista de puntos (x, y, 1) en la imagen derecha.
    """

    def dibujar_linea(ax, line, shape, color='green'):
        a, b, c = line
        height, width = shape[:2]

        if abs(b) > 1e-6:
            x_vals = np.array([0, width])
            y_vals = -(a * x_vals + c) / b
        else:
            y_vals = np.array([0, height])
            x_vals = -(b * y_vals + c) / a

        ax.plot(x_vals, y_vals, color=color, linestyle='--', linewidth=1)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Invertir eje Y

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(img_i, cmap='gray')
    axs[0].set_title('Imagen Izquierda')
    axs[0].axis('off')

    axs[1].imshow(img_d, cmap='gray')
    axs[1].set_title('Imagen Derecha')
    axs[1].axis('off')

    colores = ['red', 'green', 'blue', 'magenta', 'cyan', 'orange']

    for i in range(len(lineas_izq)):
        color = colores[i % len(colores)]

        punto_i = puntos_izq[i][:2]
        punto_d = puntos_der[i][:2]

        axs[0].scatter(*punto_i, color=color, label=f'Punto Izq #{i}')
        dibujar_linea(axs[0], lineas_izq[i], img_i.shape, color=color)

        axs[1].scatter(*punto_d, color=color, label=f'Punto Der #{i}')
        dibujar_linea(axs[1], lineas_der[i], img_d.shape, color=color)

    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def visualizar_puntos(img_i, img_d, punto_i, punto_d):
    """
    Dibuja dos gráficos simultáneos con los puntos en sus respectivas imágenes.
            
    Parámetros:
    - img_i: Imagen izquierda.
    - img_d: Imagen derecha.
    - punto_i: Coordenadas del punto clave en la imagen izquierda [x, y].
    - punto_d: Coordenadas del punto clave en la imagen derecha [x, y].
    """
                
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                
    # Imagen Izquierda
    axs[0].imshow(img_i, cmap='gray')
    axs[0].scatter(punto_i[0], punto_i[1], color='red', marker='o', label="Punto en Imagen Izquierda")
    axs[0].set_title("Imagen Izquierda")
    axs[0].axis("off")
    axs[0].legend()
                
    # Imagen Derecha
    axs[1].imshow(img_d, cmap='gray')
    axs[1].scatter(punto_d[0], punto_d[1], color='blue', marker='o', label="Punto en Imagen Derecha")
    axs[1].set_title("Imagen Derecha")
    axs[1].axis("off")
    axs[1].legend()
                
    plt.tight_layout()
    plt.show()


def main():
    img_l = cv2.imread('im_i.jpg', cv2.IMREAD_COLOR)
    img_d = cv2.imread('im_d.jpg', cv2.IMREAD_COLOR)
    flag = True

    # placeholders para usar en funciones
    K = R = tras = None
    puntos_clave_d = puntos_clave_l = None
    img_puntos_clave_d = img_puntos_clave_l = None
    F = E = puntos = puntos_e = None
    t = 50
    puntos_l = puntos_d = None
    
    while(flag):
        P = np.array([
        [1541.24, 0, 993.53, 0],  
        [0, 1538.17, 757.98, 0],  
        [0, 0, 1, 0]   
        ]) #Matriz Calibracion Jaime

        '''
        Pn = np.array([
        [1476.14, -1.19, 571.43, 0],  
        [0, 1483.97, 1009.60, 0],  
        [0, 0, 1, 0]   
        ])
        '''
        def caso_0():
            nonlocal flag
            flag = False

        def caso_1():
            nonlocal K, R, tras
            K, R, tras = krt_descomposition(P)
            return f"\nK =\n{K}\nR =\n{R}\nt =\n{tras}"

        def caso_2():
            return f"P reconstruida =\n{reconstruir_P(K, R, t)}\n P original = \n{P}"

        def caso_3():
            nonlocal img_puntos_clave_d, img_puntos_clave_l, puntos_clave_l, puntos_clave_d
            puntos_clave_l, puntos_clave_d, img_puntos_clave_l, img_puntos_clave_d = correspendencias(img_l, img_d) 
            return f"Correspondencias primeros 9 puntos imagen izquierda =\n {puntos_clave_l[:9]}\n Correspondencias primeros 9 puntos imagen derecha =\n {puntos_clave_d[:9]}"
        
        def  caso_4():
            nonlocal img_puntos_clave_d, img_puntos_clave_l
            plot_correspondencias(img_puntos_clave_l, img_puntos_clave_d)
            return f"Plot de los puntos claves realizado"

        def caso_5():
            nonlocal F, puntos, t
            r = 1000
            F, puntos = ransac(puntos_clave_l, puntos_clave_d, r, t)
            return f"Mejor matriz F = \n {F}, con primeras 4 correspondencias = \n{puntos[:4]}"
        
        def caso_6():
            nonlocal puntos_d, puntos_l
            puntos_l_list, puntos_d_list = zip(*puntos)
            puntos_l = np.vstack(puntos_l_list)  # Apila todos los puntos en una sola estructura
            puntos_d = np.vstack(puntos_d_list)  # Hace lo mismo para los puntos de la segunda imagen
            dibujar_puntos_y_lineas(img_l,img_d, puntos_l, puntos_d, len(puntos_l))
            return f"Plot de los matching realizado con matriz Fundamental (F)"

        def caso_7():
            nonlocal E                     
            E = calcular_matriz_E(F,K)
            return f"Matriz Esencial =\n {E}"
        
        def caso_8():
            nonlocal puntos_e, E
            '''
            E = np.array([
                [-37.41, 40.13, -43.14],  
                [87.40, -163.25, 22.45],  
                [-17.74, 46.35, 10.33]   
                ])
            '''
            punt_l = puntos_clave_l[:9]
            print(f"puntos clave l --> {punt_l}")
            print(E)
            E_norm = E/E[2,2] #normalizamos la E
            print(E_norm)
            E, puntos_e = matching_puntos_E(puntos_clave_l, puntos_clave_d, E_norm, t)
            return f"Primeros 9 puntos matching con matriz E = {puntos_e[:9]}"

        def caso_9():
            nonlocal puntos_d, puntos_l
            puntos_l_list, puntos_d_list = zip(*puntos_e)
            puntos_l = np.vstack(puntos_l_list)  # Apila todos los puntos en una sola estructura
            puntos_d = np.vstack(puntos_d_list)  # Hace lo mismo para los puntos de la segunda imagen
            dibujar_puntos_y_lineas(img_l,img_d, puntos_l, puntos_d, len(puntos_l))
            return f"Plot de los matching realizado con matriz Esencial (E)"
        
        def caso_10():
            nonlocal E
            #E = np.array([
            #    [-37.41, 40.13, -43.14],  
            #    [87.40, -163.25, 22.45],  
            #    [-17.74, 46.35, 10.33]   
            #    ])
            E_norm = E/E[2,2]
            det_E = np.linalg.det(E_norm)
            if det_E == 0:
                print("Determinate diferente de 0")
            else:
                print(f"Determinante igual a 0")
            #!!!!PREGUNTAR MAÑANA A PAU
            u, s, vh = np.linalg.svd(E_norm)
            s[-1] = 0  # Forzamos el tercer valor singular a cero
            E_correcta = u @ np.diag(s) @ vh
            E_ET = E_correcta @ E_correcta.T  # Multiplicación de E por su transpuesta
            valores_singulares = np.linalg.svd(E_ET, compute_uv=False)
            print(f"Valores singulares de EE^T después de corrección: {valores_singulares}")
            #!!!!PREGUNTAR MAÑANA A PAU (2 valortes singulares iguales y uno a 0)

        def caso_12():
            punto_1 = np.array([39.327152252197266, 1085.8031005859375, 1])
            punto_2 = np.array([754.025390625, 697.622802734375, 1])
            E = np.matrix([
                [502.68, -483.45, -170.30],  
                [-350.33, 334.57, 650.49],  
                [-422.59, 406.55, 114.42]   
                ])
            # Ecuacion Lounguet-Higgins, estamos comprobando si el punto x1 llace sobre la linea epipolar asociada a x2
            # para que E represente la correctamente la relacion geometrica (Rot, transl) entre las dos camaras (Kl = Kr)
            LH = punto_2.T @ E @ punto_1
            print(f"Valor LH = {LH}")
            if np.abs(LH) <= 0.001:
                strin1 = f"Se cumple Longuet-Higgins, (x'T*E*x = 0)"
            else:
                strin1 = f"No se cumple Longuet-Higgins, (x'T*E*x = 0)"
            
            l_2 = E @ punto_2
            print(f"linea epipolar 2 = {l_2}\n")
            print(f"punto epipolar 2 = {punto_2}\n")

            l_1 = E.T @ punto_1
            print(f"linea epipolar 1 = {l_1}\n")
            print(f"punto epipolar 1 = {punto_1}\n")

            #L_e2 = punto_2.T * l_2
            #L_e1 = punto_1.T * l_1
            
            L_e2 = np.dot(punto_2, l_2.A1)  # .A1 convierte de matrix a array (1D)
            L_e1 = np.dot(punto_1, l_1.A1)  

            print(f"L_e2 = {L_e2}\n")
            print(f"L_e1 = {L_e1}\n")



            if L_e1 <= 0.001:
                strin2 = " x1 está contenido en la linea epipolar l1"
            else:
                strin2 = " x1 NO está contenido en la linea epipolar l1"

            if L_e2 <= 0.001:
                strin3 = " x2 está contenido en la linea epipolar l2"
            else:
                strin3 = " x2 NO está contenido en la linea epipolar l2"
            return strin1 + strin2 + strin3

        def caso_11():
            nonlocal F, puntos_l, puntos_d
            puntos_1, puntos_2 = encontrar_mejor_punto(puntos_l, puntos_d)

            punto_1 = puntos_1[10]
            punto_2 = puntos_2[10]

            #punto_1 = np.append(punto_1, 1) ya están normalizados
            #punto_2 = np.append(punto_2, 1)

            print("punto_1 = ", punto_1)
            print("punto_2 = ", punto_2)
            print("F = ", F)


            '''
            punto_1 = np.array([39.327152252197266, 1085.8031005859375, 1])
            punto_2 = np.array([754.025390625, 697.622802734375, 1])
            
            F = np.matrix([
                [-0.0000688, -0.0000792, -0.242],  
                [-0.000837, 0.000916, -0.361],  
                [-0.0202, -0.0868, 1]
                ])
            '''
            # Ecuacion Lounguet-Higgins, estamos comprobando si el punto x1 llace sobre la linea epipolar asociada a x2
            # para que E represente la correctamente la relacion geometrica (Rot, transl) entre las dos camaras (Kl = Kr)
            LH = punto_2.T @ F @ punto_1
            print(f"Valor LH = {LH}")
            if np.abs(LH) <= 0.001:
                strin1 = f"Se cumple Longuet-Higgins, (x'T*F*x = 0)"
            else:
                strin1 = f"No se cumple Longuet-Higgins, (x'T*F*x = 0)"
            
            l_2 = F @ punto_2
            print(f"linea epipolar 2 = {l_2}\n")
            print(f"punto epipolar 2 = {punto_2}\n")

            l_1 = F.T @ punto_1
            print(f"linea epipolar 1 = {l_1}\n")
            print(f"punto epipolar 1 = {punto_1}\n")

            #L_e2 = punto_2.T * l_2
            #L_e1 = punto_1.T * l_1
            
            L_e2 = np.dot(punto_2, l_2)  # .A1 convierte de matrix a array (1D)
            L_e1 = np.dot(punto_1, l_1)  

            print(f"L_e2 = {L_e2}\n")
            print(f"L_e1 = {L_e1}\n")



            if np.abs(L_e1) <= 0.001:
                strin2 = " x1 está contenido en la linea epipolar l1"
            else:
                strin2 = " x1 NO está contenido en la linea epipolar l1"

            if np.abs(L_e2) <= 0.001:
                strin3 = " x2 está contenido en la linea epipolar l2"
            else:
                strin3 = " x2 NO está contenido en la linea epipolar l2"
            return strin1 + strin2 + strin3
        
        def caso_13():
            nonlocal F, puntos_l, puntos_d
            puntos_1, puntos_2 = encontrar_mejor_punto(puntos_l, puntos_d)

            punto_11 = puntos_1[10]
            punto_12 = puntos_2[10]
            l11 = F.T @ punto_12
            l12 = F @ punto_11
            print(f"Punto 1 = {punto_11}")
            print(f"Punto 2 = {punto_12}")

            print(f"Linea epipolar 1 = {l11}")
            print(f"Linea epipolar 2 = {l12}")

            punto_21 = puntos_1[20]
            punto_22 = puntos_2[20]

            l21 = F.T @ punto_22
            l22 = F @ punto_21

            punto_31 = puntos_1[30]
            punto_32 = puntos_2[30]
            l31 = F.T @ punto_32
            l32 = F @ punto_31

            punto_41 = puntos_1[40]
            punto_42 = puntos_2[40]
            l41 = F.T @ punto_42
            l42 = F @ punto_41

            visualizar_puntos(img_l, img_d, punto_11, punto_12)

            visualizar_lineas_epipolares(img_d, img_l, l11, l12, punto_11, punto_12)


        def caso_14():
            
            nonlocal E, puntos_l, puntos_d
            puntos_1, puntos_2 = encontrar_mejor_punto(puntos_l, puntos_d)

            punto_11 = puntos_1[10]
            punto_12 = puntos_2[10]
            l11 = E.T @ punto_12
            l12 = E @ punto_11
            print(f"Punto 1 = {punto_11}")
            print(f"Punto 2 = {punto_12}")

            print(f"Linea epipolar 1 = {l11}")
            print(f"Linea epipolar 2 = {l12}")

            punto_21 = puntos_1[50]
            punto_22 = puntos_2[50]

            punto_31 = puntos_1[100]
            punto_32 = puntos_2[100]


            punto_41 = puntos_1[150]
            punto_42 = puntos_2[150]

            puntos_izq = [punto_11, punto_21, punto_31, punto_41]
            puntos_der = [punto_12, punto_22, punto_32, punto_42]
            lineas_izq = [E.T @ p for p in puntos_der]  # líneas en imagen izquierda desde puntos derechos
            lineas_der = [E @ p for p in puntos_izq]    # líneas en imagen derecha desde puntos izquierdos



            visualizar_puntos(img_l, img_d, punto_11, punto_12)

            visualizar_lineas_epipolares(img_d, img_l, l11, l12, punto_11, punto_12)

            visualizar_lineas_epipolares_listas(img_d, img_l, lineas_izq, lineas_der, puntos_izq, puntos_der)
            

        switch = {
            "0": caso_0,
            "1": caso_1,
            "2": caso_2,
            "3": caso_3,
            "4": caso_4,
            "5": caso_5,
            "6": caso_6,
            "7": caso_7,
            "8": caso_8,
            "9": caso_9,
            "10": caso_10,
            "11": caso_11,
            "12": caso_12,
            "13": caso_13,
            "14": caso_14
        }

        opcion = input("Elige una opción (0-14): ")
        resultado = switch.get(opcion, lambda: "Opción no válida")()
        print(resultado)

if __name__ == '__main__':
    main()

