import numpy as np
import cv2 
import matplotlib.pyplot as plt
import sys
import random
from scipy.linalg import rq

def krt_descomposition(P):
    # Aplicamos la descomposicón QR
    Q, U = np.linalg.qr(np.linalg.inv(P[:3, :3]))
    #D = np.diag(np.sign(np.diag(U)) * np.array([-1, -1, 1]))
    #Q = Q * D
    #U = D * U
    # Obtenemos s para forzar que la matriz de rotación R resultante 
    # tenga determinante positivo, ya que es una forma de garantizar que la rotación sea pura
    s = np.linalg.det(Q)
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
    return K @ np.hstack((R, t[:,-1].reshape(-1, 1)))

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
    centroide = np.mean(puntos, axis=0)
    puntos_desplazados = puntos - centroide
    distancia_media = np.mean(np.sqrt(np.sum(puntos_desplazados**2, axis=1)))
    factor_escala = np.sqrt(2) / distancia_media
    T = np.array([[factor_escala, 0, -factor_escala * centroide[0]],
                [0, factor_escala, -factor_escala * centroide[1]],
                [0, 0, 1]])
    puntos_homogeneos = np.hstack((puntos, np.ones((puntos.shape[0], 1))))
    puntos_normalizados = (T @ puntos_homogeneos.T).T
    return puntos_normalizados[:, :2], T

def sampson_error(F, x1, x2):
    x1 = x1.T  # (3, N)
    x2 = x2.T  # (3, N)
    Fx1 = F @ x1
    Ftx2 = F.T @ x2
    x2tFx1 = np.sum(x2 * (F @ x1), axis=0)
    denom = Fx1[0, :]**2 + Fx1[1, :]**2 + Ftx2[0, :]**2 + Ftx2[1, :]**2
    return x2tFx1**2 / denom

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
        F = F / F[2,2]
        Uf, Sf, Vtf = np.linalg.svd(F)
        Sf[-1] = 0
        F_rank2 = Uf @ np.diagflat(Sf) @ Vtf
        F_denorm = T2.T @ F_rank2 @ T1    
        F_norm = F_denorm/F_denorm[2,2] 
        return F_norm
    C_est = []
    C = []
    max_inliers = 0
    best_error = np.inf
    puntos_normalizados_l, T1 = normalizar_puntos(puntos_clave_l)
    puntos_normalizados_d, T2 = normalizar_puntos(puntos_clave_d)
    print("Empezamos RANSAC")
    for _ in range(iter):
        N = min(len(puntos_clave_l), len(puntos_clave_d))
        assert N >= 8, "No hay suficientes puntos para RANSAC"
        idx = random.sample(range(N), 8)
        sample_l = puntos_clave_l[idx]
        sample_d = puntos_clave_d[idx]
        sample_l, T_1 = normalizar_puntos(sample_l)
        sample_d, T_2 = normalizar_puntos(sample_d)
        F = eight_point_algorithm(sample_l, sample_d, T_1, T_2)
        inliers = 0
        C = []
        for i, (pl_n, pd_n) in enumerate(zip(puntos_normalizados_l, puntos_normalizados_d)):
            pl_n_h = np.append(pl_n, 1)
            pd_n_h = np.append(pd_n, 1)
            error = sampson_error(F, np.array([pl_n_h]), np.array([pd_n_h]))[0]
            if error < t:
                inliers += 1
                pl = tuple(map(float, puntos_clave_l[i]))
                pd = tuple(map(float, puntos_clave_d[i]))
                C.append((pl, pd))
        if len(C) > len(C_est) and best_error > error:
            print("Mejor error hasta el momento:", error)
            best_error = error
            C_est_np = (np.array(C)).copy()
            F_est = F
            if inliers > max_inliers:
                max_inliers = inliers
                print(f"max_inliers = {max_inliers}")
    print("C_est =")
    print(C_est_np)
    print("F_est =")
    print(F_est)
    print("Terminamos RANSAC")
    if len(C_est_np) >= 8:
        puntos_l_inliers = np.array([p[0] for p in C_est_np])
        puntos_d_inliers = np.array([p[1] for p in C_est_np])
        puntos_l_norm, T1 = normalizar_puntos(puntos_l_inliers)
        puntos_d_norm, T2 = normalizar_puntos(puntos_d_inliers)
        F_final = eight_point_algorithm(puntos_l_norm, puntos_d_norm, T1, T2)
        return F_final, C_est_np
    else:
        return F_est, C_est_np 

def krt_decomposition_fusiello(P):
    """
    Descompone la matriz de proyección P en K, R, t según Fusiello (Alg. 4.3).
    Input:
        P: Matriz de proyección 3x4
    Output:
        K: Matriz de parámetros intrínsecos (3x3)
        R: Matriz de rotación (3x3)
        t: Vector de traslación (3x1)
    """
    # Extraer M (las primeras 3x3 de P)
    M = P[:, :3]
    # RQ descomposición (no QR)
    K, R = rq(M)
    # Normalizar K para que K[2,2] = 1 y el determinante de R sea positivo
    if np.linalg.det(R) < 0:
        R = -R
        K = -K
    K = K / K[2,2]
    # Calcular t
    t = np.linalg.inv(K) @ P[:, 3]
    return K, R, t 

def dibujar_lineas_epipolares(img1, img2, F, pts1, pts2, num=10):
    """
    Dibuja líneas epipolares en ambas imágenes para un subconjunto de correspondencias.
    img1, img2: imágenes originales
    F: matriz fundamental
    pts1, pts2: puntos correspondientes (Nx2)
    num: número de correspondencias a mostrar
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    idx = random.sample(range(len(pts1)), min(num, len(pts1)))
    pts1 = pts1[idx]
    pts2 = pts2[idx]

    def draw_lines(img, lines, pts, color):
        '''Dibuja líneas epipolares y puntos'''
        r, c = img.shape[:2]
        img = img.copy()
        for r_line, pt in zip(lines, pts):
            a, b, c_ = r_line
            x0, y0 = 0, int(-c_/b) if b != 0 else 0
            x1, y1 = img.shape[1], int(-(c_ + a*img.shape[1])/b) if b != 0 else 0
            img = cv2.line(img, (x0, y0), (x1, y1), color, 2)  # grosor 2
            img = cv2.circle(img, tuple(np.int32(pt)), 8, (0,0,255), -1)  # puntos rojos grandes
        return img

    # Calcula las líneas epipolares en la segunda imagen para los puntos de la primera
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    lines2 = (F @ pts1_h.T).T  # líneas en img2

    # Calcula las líneas epipolares en la primera imagen para los puntos de la segunda
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    lines1 = (F.T @ pts2_h.T).T  # líneas en img1

    img1_lines = draw_lines(img1, lines1, pts1, (0,255,0))   # verde
    img2_lines = draw_lines(img2, lines2, pts2, (255,0,0))   # azul

    plt.figure(figsize=(14,6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img1_lines, cv2.COLOR_BGR2RGB))
    plt.title('Líneas epipolares en imagen 1')
    plt.axis('off')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img2_lines, cv2.COLOR_BGR2RGB))
    plt.title('Líneas epipolares en imagen 2')
    plt.axis('off')
    plt.show()

    # Epipolos: núcleos derecho e izquierdo de F
    U, S, Vt = np.linalg.svd(F)
    epipolo_izq = Vt[-1]
    epipolo_izq = epipolo_izq/epipolo_izq[2]
    U, S, Vt = np.linalg.svd(F.T)
    epipolo_der = Vt[-1]
    epipolo_der = epipolo_der/epipolo_der[2]
    print("Epipolo izquierdo (en img1):", epipolo_izq)
    print("Epipolo derecho (en img2):", epipolo_der)

def rectificacion_calibrada(E, y1, y2):
    """
    Rectificación estereoscópica calibrada.
    E: matriz esencial
    y1, y2: puntos normalizados correspondientes (3,)
    Devuelve: R, t (rotación y traslación relativas)
    """
    # Descomposición de E en R y t (ver Hartley-Zisserman Alg. 9.13)
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U *= -1
    if np.linalg.det(Vt) < 0: Vt *= -1
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R = U @ W @ Vt
    t = U[:,2]
    # Asegura que el punto esté delante de ambas cámaras
    if np.dot(np.cross(R @ y1, y2), t) < 0:
        t = -t
    return R, t 

def homografias_rectificacion(Kl, Kr, R, t, y1, y2):
    """
    Calcula las homografías de rectificación para imágenes calibradas.
    Kl, Kr: matrices de calibración (3x3)
    R, t: rotación y traslación relativas (de la descomposición de E)
    y1, y2: puntos normalizados correspondientes (3,)
    Devuelve: Hl, Hr (homografías 3x3)
    """
    # Ejes de la nueva cámara
    r1 = t / np.linalg.norm(t)
    r2 = np.cross(y1, r1)
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    R_rect = np.stack((r1, r2, r3)).T  # 3x3

    # Homografía para la imagen izquierda
    Hl = Kl @ R_rect @ np.linalg.inv(Kl)
    # Homografía para la imagen derecha
    Hr = Kr @ R_rect @ R.T @ np.linalg.inv(Kr)
    return Hl, Hr 

def mostrar_rectificacion(img_l, img_r, Hl, Hr):
    h, w = img_l.shape[:2]
    img_l_rect = cv2.warpPerspective(img_l, Hl, (w, h))
    img_r_rect = cv2.warpPerspective(img_r, Hr, (w, h))
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Imagen izquierda rectificada")
    plt.imshow(cv2.cvtColor(img_l_rect, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Imagen derecha rectificada")
    plt.imshow(cv2.cvtColor(img_r_rect, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def homografia_rectificacion_epipolar(F, img_shape):
    """
    Calcula la homografía de rectificación para la segunda imagen (derecha)
    siguiendo la geometría epipolar (Hartley-Zisserman Alg. 11.10).
    F: matriz fundamental
    img_shape: shape de la imagen (h, w)
    Devuelve: H2 (homografía 3x3 para la imagen derecha)
    """
    h, w = img_shape[:2]
    # Epipolo derecho (en la segunda imagen)
    U, S, Vt = np.linalg.svd(F.T)
    e = Vt[-1]
    e = e / e[2]
    # Traslada el centro de la imagen al origen
    T = np.array([[1, 0, -w/2],
                  [0, 1, -h/2],
                  [0, 0, 1]])
    e_ = T @ e
    # Rotación para alinear el epipolo con el eje x
    alpha = np.arctan2(e_[1], e_[0])
    R = np.array([[np.cos(-alpha), -np.sin(-alpha), 0],
                  [np.sin(-alpha),  np.cos(-alpha), 0],
                  [0, 0, 1]])
    e_rot = R @ e_
    # Proyección para mandar el epipolo al infinito
    f = e_rot[0]
    G = np.eye(3)
    if np.abs(f) > 1e-6:
        G[2,0] = -1/f
    # Homografía total
    H = np.linalg.inv(T) @ G @ R @ T
    return H

def draw_epipolar_line(ax, l, img_shape, color='r'):
    """Dibuja la línea epipolar l en el eje ax, recortada a los bordes de la imagen."""
    h, w = img_shape[:2]
    a, b, c = l
    points = []
    # Intersección con los bordes izquierdo y derecho (x=0, x=w-1)
    for x in [0, w-1]:
        if abs(b) > 1e-6:
            y = -(a*x + c)/b
            if 0 <= y < h:
                points.append((x, int(round(y))))
    # Intersección con los bordes superior e inferior (y=0, y=h-1)
    for y in [0, h-1]:
        if abs(a) > 1e-6:
            x = -(b*y + c)/a
            if 0 <= x < w:
                points.append((int(round(x)), y))
    # Elimina duplicados
    points = list(dict.fromkeys(points))
    # Si hay al menos dos puntos dentro de la imagen, dibuja la línea
    if len(points) >= 2:
        (x0, y0), (x1, y1) = points[:2]
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2)

def click_epipolar(img1, img2, F, pts1, pts2, modo='F', lado='izq'):
    """
    Permite clicar en img1 y muestra la línea epipolar correspondiente en img2.
    lado: 'izq' (default) para click en img1 y línea en img2,
          'der' para click en img2 y línea en img1.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Haz click en un punto (imagen 1)" if lado=='izq' else "Haz click en un punto (imagen 2)")
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Línea epipolar en imagen 2" if lado=='izq' else "Línea epipolar en imagen 1")
    axs[0].axis('off')
    axs[1].axis('off')

    def onclick(event):
        if event.inaxes == axs[0]:
            x, y = event.xdata, event.ydata
            axs[0].clear()
            axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Haz click en un punto (imagen 1)" if lado=='izq' else "Haz click en un punto (imagen 2)")
            axs[0].axis('off')
            axs[0].plot(x, y, 'ro', markersize=10)

            # Busca la correspondencia más cercana (si hay matches)
            corr = None
            if pts1 is not None and pts2 is not None and len(pts1) > 0:
                dists = np.linalg.norm(pts1 - np.array([x, y]), axis=1) if lado=='izq' else np.linalg.norm(pts2 - np.array([x, y]), axis=1)
                idx = np.argmin(dists)
                if dists[idx] < 20:  # Solo si está cerca (umbral en píxeles)
                    corr = pts2[idx] if lado=='izq' else pts1[idx]

            axs[1].clear()
            axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            axs[1].set_title("Línea epipolar en imagen 2" if lado=='izq' else "Línea epipolar en imagen 1")
            axs[1].axis('off')
            if corr is not None:
                axs[1].plot(corr[0], corr[1], 'go', markersize=10)

            # Calcula la línea epipolar
            p = np.array([x, y, 1])
            if modo == 'F' or modo == 'E':
                l = F @ p if lado=='izq' else F.T @ p
            else:
                raise ValueError("modo debe ser 'F' o 'E'")

            # Dibuja la línea epipolar robustamente
            draw_epipolar_line(axs[1], l, img2.shape if lado=='izq' else img1.shape, color='r')
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def main_menu():
    # Carga imágenes
    img_l = cv2.imread('im_i.jpg')
    img_r = cv2.imread('im_d.jpg')
    if img_l is None or img_r is None:
        print("No se pudieron cargar las imágenes. Verifica las rutas y nombres.")
        return

    # Matriz de calibración de ejemplo (ajusta según tu cámara)
    K = np.array([
        [1541.24, 0, 993.53],
        [0, 1538.17, 757.98],
        [0, 0, 1]
    ])

    puntos_l = puntos_r = img_puntos_l = img_puntos_r = None
    F = E = None
    inliers = None
    puntos_l_inliers = puntos_r_inliers = None
    R = t = Hl = Hr = None

    while True:
        print("\n--- MENÚ ---")
        print("1. Mostrar correspondencias SIFT")
        print("2. Calcular y mostrar la matriz fundamental y líneas epipolares")
        print("3. Calcular y mostrar la matriz esencial")
        print("4. Rectificación estereoscópica y visualización")
        print("5. Clicar y mostrar línea epipolar (F)")
        print("6. Clicar y mostrar línea epipolar (E)")
        print("7. Salir")
        opcion = input("Elige una opción: ")

        if opcion == "1":
            puntos_l, puntos_r, img_puntos_l, img_puntos_r = correspendencias(img_l, img_r)
            plot_correspondencias(img_puntos_l, img_puntos_r)

        elif opcion == "2":
            if puntos_l is None or puntos_r is None:
                print("Primero debes obtener las correspondencias (opción 1).")
                continue
            F, inliers = ransac(puntos_l, puntos_r, iter=5000, t=1.0)
            print("Matriz Fundamental F:\n", F)
            puntos_l_inliers = np.array([p[0] for p in inliers])
            puntos_r_inliers = np.array([p[1] for p in inliers])
            dibujar_lineas_epipolares(img_l, img_r, F, puntos_l_inliers, puntos_r_inliers, num=10)

        elif opcion == "3":
            if F is None or puntos_l_inliers is None or puntos_r_inliers is None:
                print("Primero debes calcular la matriz fundamental (opción 2).")
                continue
            E = K.T @ F @ K
            print("Matriz Esencial E:\n", E)

        elif opcion == "4":
            if F is None or puntos_l_inliers is None or puntos_r_inliers is None:
                print("Primero debes calcular la matriz fundamental (opción 2).")
                continue
            # Calcula la homografía de rectificación epipolar para la imagen derecha
            H2 = homografia_rectificacion_epipolar(F, img_r.shape)
            H1 = np.eye(3)  # Para la izquierda, puedes usar la identidad
            mostrar_rectificacion(img_l, img_r, H1, H2)

        elif opcion == "5":
            if F is None or puntos_l_inliers is None or puntos_r_inliers is None:
                print("Primero debes calcular la matriz fundamental (opción 2).")
                continue
            click_epipolar(img_l, img_r, F, puntos_l_inliers, puntos_r_inliers, modo='F', lado='izq')

        elif opcion == "6":
            if F is None or puntos_l_inliers is None or puntos_r_inliers is None:
                print("Primero debes calcular la matriz fundamental (opción 2).")
                continue
            click_epipolar(img_r, img_l, F, puntos_r_inliers, puntos_l_inliers, modo='F', lado='der')

        elif opcion == "7":
            if E is None or puntos_l_inliers is None or puntos_r_inliers is None:
                print("Primero debes calcular la matriz esencial (opción 3).")
                continue
            click_epipolar(img_l, img_r, E, puntos_l_inliers, puntos_r_inliers, modo='E', lado='izq')

        elif opcion == "8":
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main_menu() 