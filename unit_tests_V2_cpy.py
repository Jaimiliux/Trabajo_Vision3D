import numpy as np
import cv2 
import matplotlib.pyplot as plt
import sys
import random

def rq_decomposition_numpy(A):
    """
    Realiza la descomposición RQ de una matriz 3x3 usando solo numpy.
    Devuelve R (triangular superior) y Q (ortogonal).
    """
    # Voltea la matriz y transpón
    A_flip = np.flipud(A).T
    Q, R = np.linalg.qr(A_flip)
    # Deshaz el flip y la transposición
    R = np.flipud(R.T)
    Q = Q.T[:, ::-1]
    # Asegura diagonal positiva en R
    for i in range(3):
        if R[i, i] < 0:
            R[:, i] *= -1
            Q[i, :] *= -1
    return R, Q

def krt_descomposition(P):
    """
    Descompone la matriz de proyección P en K, R, t usando solo numpy.
    P: matriz de proyección 3x4
    Devuelve:
        K: matriz intrínseca (3x3)
        R: matriz de rotación (3x3)
        t: vector de traslación (3x1)
    """
    M = P[:, :3]
    if np.allclose(M, np.triu(M)):
        K = M.copy()
        R = np.eye(3)
    else:
        K, R = rq_decomposition_numpy(M)
        if np.abs(K[2,2]) < 1e-8:
            raise ValueError("K[2,2] es cero tras la RQ. La matriz P puede ser degenerada.")
        K = K / K[2, 2]
        if np.linalg.det(R) < 0:
            R = -R
            K = -K
    t = np.linalg.inv(K) @ P[:, 3]
    return K, R, t.reshape(-1, 1)

def reconstruir_P(K, R, t):
    return K @ np.hstack((R, t))

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
'''
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
'''
def sampson_error(F, x1, x2):
    """
    Calcula el error de Sampson para cada correspondencia.
    F: matriz fundamental (3x3)
    x1, x2: puntos homogéneos (Nx3)
    Retorna: error de Sampson (N,)
    """
    x1 = x1.T  # (3, N)
    x2 = x2.T  # (3, N)
    Fx1 = F @ x1
    Ftx2 = F.T @ x2
    x2tFx1 = np.sum(x2 * (F @ x1), axis=0)
    denom = Fx1[0, :]**2 + Fx1[1, :]**2 + Ftx2[0, :]**2 + Ftx2[1, :]**2
    return x2tFx1**2 / denom

def plot_sift_matches(img_l, img_d, puntos_clave_l, puntos_clave_d):
    """
    Visualize all SIFT matches (before RANSAC)
    """
    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_d_rgb = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
    h1, w1 = img_l.shape[:2]
    h2, w2 = img_d.shape[:2]
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img_l_rgb
    out_img[:h2, w1:w1 + w2] = img_d_rgb
    plt.figure(figsize=(16, 8))
    plt.imshow(out_img)
    for (x1, y1), (x2, y2) in zip(puntos_clave_l, puntos_clave_d):
        plt.plot([x1, x2 + w1], [y1, y2], 'y-', linewidth=0.5)
        plt.plot(x1, y1, 'ro', markersize=2)
        plt.plot(x2 + w1, y2, 'bo', markersize=2)
    plt.title('All SIFT matches (before RANSAC)')
    plt.axis('off')
    plt.show()

def plot_inlier_matches(img_l, img_d, inliers):
    """
    Visualize inlier matches (after RANSAC)
    """
    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_d_rgb = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
    h1, w1 = img_l.shape[:2]
    h2, w2 = img_d.shape[:2]
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img_l_rgb
    out_img[:h2, w1:w1 + w2] = img_d_rgb
    plt.figure(figsize=(16, 8))
    plt.imshow(out_img)
    for (p1, p2) in inliers:
        x1, y1 = p1
        x2, y2 = p2
        plt.plot([x1, x2 + w1], [y1, y2], 'g-', linewidth=0.7)
        plt.plot(x1, y1, 'ro', markersize=3)
        plt.plot(x2 + w1, y2, 'bo', markersize=3)
    plt.title('Inlier matches (after RANSAC)')
    plt.axis('off')
    plt.show()

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

        Uf, Sf, Vtf = np.linalg.svd(F) #!!!Normalizar valores singulares
        #tan_v = Sf[1]/Sf[0]
        #cos_v = 1 / np.sqrt(1 + tan_v**2)
        #sin_v = tan_v * cos_v

        #Sf[0] = cos_v
        #Sf[1] = sin_v

        Sf[-1] = 0  # anular el menor valor singular
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

        F = eight_point_algorithm(sample_l, sample_d, T_1, T_2) #poner aqui dentro la normalizacion de puntos
        inliers = 0
        C = []
        
        for i, (pl_n, pd_n) in enumerate(zip(puntos_normalizados_l, puntos_normalizados_d)):
            # Convertir a homogéneo (añadir 1)
            pl_n_h = np.append(pl_n, 1)
            pd_n_h = np.append(pd_n, 1)
            # El error de Sampson espera arrays de Nx3
            error = sampson_error(F, np.array([pl_n_h]), np.array([pd_n_h]))[0]
            if error < t:
                inliers += 1
                pl = tuple(map(float, puntos_clave_l[i]))
                pd = tuple(map(float, puntos_clave_d[i]))
                C.append((pl, pd))

        if len(C) > len(C_est) and best_error > error:
        #if len(C) > len(C_est):
            print("Mejor error hasta el momento:", error)
            best_error = error
            #C_est = np.array(C)
            #C_est_np = (np.array(C_est, dtype=object)).copy()
            C_est_np = (np.array(C)).copy()
            F_est = F
            if inliers > max_inliers:
                max_inliers = inliers
                print(f"max_inliers = {max_inliers}")
    print(f"Total inliers found: {max_inliers} (threshold t={t})")
    #Filtrar luego para quedarse con las rectas con la orientación mas similar / común
    print("C_est =")
    print(C_est_np)
    print("F_est =")
    print(F_est)
    print("Terminamos RANSAC")
    if len(C_est_np) >= 8:
        # Extrae los puntos inliers correctamente
        puntos_l_inliers = np.array([p[0] for p in C_est_np])  # (x1, y1)
        puntos_d_inliers = np.array([p[1] for p in C_est_np])  # (x2, y2)
        # Normaliza
        puntos_l_norm, T1 = normalizar_puntos(puntos_l_inliers)
        puntos_d_norm, T2 = normalizar_puntos(puntos_d_inliers)
        # Calcula F con todos los inliers
        F_final = eight_point_algorithm(puntos_l_norm, puntos_d_norm, T1, T2)
        puntos_l_list, puntos_d_list = zip(*C_est_np)
        puntos_l = np.vstack(puntos_l_list)
        puntos_d = np.vstack(puntos_d_list)
        return F_final, C_est_np
    else:
        return F_est, C_est_np

def angle_bin(v, num_bins=36):
        angle = np.arctan2(v[1], v[0])
        bin_index = int(((angle + np.pi) / (2 * np.pi)) * num_bins)
        return bin_index
def calcular_matriz_E(F,K):
    K_np = np.array(K)
    K_trans = K_np.T
    E = K_trans @ F @ K
    return E

def check_matrix_properties(F, E, K, puntos_l, puntos_d):
    """
    Diagnostic function to check properties of matrices and points
    """
    print("\n=== Matrix Properties Check ===")
    
    # Check F
    print("\n1. Fundamental Matrix (F) Analysis:")
    U, S, Vt = np.linalg.svd(F)
    print(f"Singular values of F: {S}")
    print(f"Rank of F: {np.linalg.matrix_rank(F)}")
    print(f"det(F): {np.linalg.det(F)}")
    
    # Check E
    print("\n2. Essential Matrix (E) Analysis:")
    U, S, Vt = np.linalg.svd(E)
    print(f"Singular values of E: {S}")
    print(f"Rank of E: {np.linalg.matrix_rank(E)}")
    print(f"det(E): {np.linalg.det(E)}")
    
    # Check K
    print("\n3. Intrinsic Matrix (K) Analysis:")
    print(f"Focal lengths: fx={K[0,0]}, fy={K[1,1]}")
    print(f"Principal point: cx={K[0,2]}, cy={K[1,2]}")
    
    # Check points
    print("\n4. Points Analysis:")
    print(f"Number of points: {len(puntos_l)}")
    print(f"Points left image - Mean: {np.mean(puntos_l, axis=0)}, Std: {np.std(puntos_l, axis=0)}")
    print(f"Points right image - Mean: {np.mean(puntos_d, axis=0)}, Std: {np.std(puntos_d, axis=0)}")
    
    # Check epipolar constraint
    print("\n5. Epipolar Constraint Check:")
    errors = []
    for i in range(min(5, len(puntos_l))):  # Check first 5 points
        x1 = np.append(puntos_l[i], 1)
        x2 = np.append(puntos_d[i], 1)
        error = x2.T @ F @ x1
        errors.append(error)
    print(f"Epipolar constraint errors (first 5 points): {errors}")
    print(f"Mean epipolar error: {np.mean(np.abs(errors))}")

def visualizar_epipolar_validation(img_l, img_d, F, puntos_l, puntos_d, E=None, K=None, num_points=5):
    """
    Visualize epipolar lines for validation (both F and E if provided)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_d_rgb = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_l_rgb)
    ax2.imshow(img_d_rgb)
    for i in range(min(num_points, len(puntos_l))):
        x1, y1 = puntos_l[i]
        x2, y2 = puntos_d[i]
        p1 = np.array([x1, y1, 1])
        p2 = np.array([x2, y2, 1])
        # F-based lines
        l2 = F @ p1
        l1 = F.T @ p2
        # Plot points
        ax1.plot(x1, y1, 'ro', markersize=5)
        ax2.plot(x2, y2, 'bo', markersize=5)
        h, w = img_l.shape[:2]
        # F lines
        if abs(l1[1]) > 1e-6:
            y1_line = np.array([0, h])
            x1_line = -(l1[1] * y1_line + l1[2]) / l1[0]
        else:
            x1_line = np.array([0, w])
            y1_line = -(l1[0] * x1_line + l1[2]) / l1[1]
        ax1.plot(x1_line, y1_line, 'g-', linewidth=1, label='F')
        if abs(l2[1]) > 1e-6:
            y2_line = np.array([0, h])
            x2_line = -(l2[1] * y2_line + l2[2]) / l2[0]
        else:
            x2_line = np.array([0, w])
            y2_line = -(l2[0] * x2_line + l2[2]) / l2[1]
        ax2.plot(x2_line, y2_line, 'g-', linewidth=1, label='F')
        # E-based lines (if E and K provided)
        if E is not None and K is not None:
            # Left image: line from right point
            p2_norm = np.linalg.inv(K) @ p2
            l1_E = E.T @ p2_norm
            l1_E_pix = np.linalg.inv(K).T @ l1_E
            if abs(l1_E_pix[1]) > 1e-6:
                y1e = np.array([0, h])
                x1e = -(l1_E_pix[1] * y1e + l1_E_pix[2]) / l1_E_pix[0]
            else:
                x1e = np.array([0, w])
                y1e = -(l1_E_pix[0] * x1e + l1_E_pix[2]) / l1_E_pix[1]
            ax1.plot(x1e, y1e, 'm--', linewidth=1, label='E')
            # Right image: line from left point
            p1_norm = np.linalg.inv(K) @ p1
            l2_E = E @ p1_norm
            l2_E_pix = np.linalg.inv(K).T @ l2_E
            if abs(l2_E_pix[1]) > 1e-6:
                y2e = np.array([0, h])
                x2e = -(l2_E_pix[1] * y2e + l2_E_pix[2]) / l2_E_pix[0]
            else:
                x2e = np.array([0, w])
                y2e = -(l2_E_pix[0] * x2e + l2_E_pix[2]) / l2_E_pix[1]
            ax2.plot(x2e, y2e, 'm--', linewidth=1, label='E')
    ax1.set_title('Left Image with Epipolar Lines')
    ax2.set_title('Right Image with Epipolar Lines')
    plt.tight_layout()
    plt.show()

def robust_sift_matching(img_l, img_d, ratio_thresh=0.75):
    """
    SIFT matching with Lowe's ratio test and cross-checking.
    Returns filtered keypoints and matches.
    """
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_l, None)
    kp2, des2 = sift.detectAndCompute(img_d, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    puntos_l = []
    puntos_d = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            # Cross-check: verify that the match is mutual best
            queryIdx = m.queryIdx
            trainIdx = m.trainIdx
            # Find the best match for trainIdx in des1
            matches2 = bf.knnMatch(des2, des1, k=2)
            if matches2[trainIdx][0].trainIdx == queryIdx:
                puntos_l.append(kp1[queryIdx].pt)
                puntos_d.append(kp2[trainIdx].pt)
                good.append(m)
    puntos_l = np.array(puntos_l, dtype=np.float32)
    puntos_d = np.array(puntos_d, dtype=np.float32)
    # Print first 8 correspondences for manual entry
    print("\nFirst 8 correspondences (copy these for manual entry in case 20):")
    for i in range(min(8, len(puntos_l))):
        l = puntos_l[i]
        d = puntos_d[i]
        print(f"Left: {l[0]:.2f} {l[1]:.2f}    Right: {d[0]:.2f} {d[1]:.2f}")
    return puntos_l, puntos_d, good, kp1, kp2

def manual_correspondences():
    """
    Allow user to manually input a small set of correspondences for testing.
    Returns two lists of points.
    """
    print("Manual correspondences mode. Enter coordinates as x y (e.g., 100 200). Type 'done' to finish.")
    puntos_l = []
    puntos_d = []
    while True:
        s = input("Left image point (or 'done'): ")
        if s.strip().lower() == 'done':
            break
        x, y = map(float, s.strip().split())
        puntos_l.append([x, y])
        s2 = input("Right image point: ")
        x2, y2 = map(float, s2.strip().split())
        puntos_d.append([x2, y2])
    return np.array(puntos_l, dtype=np.float32), np.array(puntos_d, dtype=np.float32)

# Add a global RANSAC threshold variable
RANSAC_THRESHOLD = 10

def interactive_epipolar_view(img_l, img_d, F):
    """
    Interactive function: user clicks a point in the left or right image,
    and the corresponding epipolar line is drawn in the other image.
    """
    def draw_line(ax, line, shape, color='g'):
        a, b, c = line
        h, w = shape[:2]
        if abs(b) > 1e-6:
            x_vals = np.array([0, w])
            y_vals = -(a * x_vals + c) / b
        else:
            y_vals = np.array([0, h])
            x_vals = -(b * y_vals + c) / a
        ax.plot(x_vals, y_vals, color=color, linestyle='-', linewidth=2)

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_d_rgb = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
    axs[0].imshow(img_l_rgb)
    axs[0].set_title('Left Image (click here or right)')
    axs[1].imshow(img_d_rgb)
    axs[1].set_title('Right Image (click here or left)')
    plt.tight_layout()
    print("Click a point in either image (left or right). Close the window to exit.")
    while True:
        pts = plt.ginput(1, timeout=0)
        if not pts:
            break
        x, y = pts[0]
        # Determine which image was clicked
        w = img_l.shape[1]
        if x < w:
            # Clicked in left image
            x_img, y_img = x, y
            axs[0].scatter(x_img, y_img, color='red', s=60)
            p = np.array([x_img, y_img, 1])
            l = F @ p  # Epipolar line in right image
            draw_line(axs[1], l, img_d.shape, color='g')
            axs[1].set_title('Right Image (epipolar line shown)')
        else:
            # Clicked in right image
            x_img, y_img = x - w, y
            axs[1].scatter(x_img, y_img, color='blue', s=60)
            p = np.array([x_img, y_img, 1])
            l = F.T @ p  # Epipolar line in left image
            draw_line(axs[0], l, img_l.shape, color='g')
            axs[0].set_title('Left Image (epipolar line shown)')
        plt.draw()
    plt.close(fig)
    print("Interactive epipolar view closed.")

def interactive_epipolar_view_E(img_l, img_d, E, K):
    """
    Interactive function: user clicks a point in the left or right image,
    and the corresponding epipolar line is drawn in the other image using the essential matrix E.
    """
    def draw_line(ax, line, shape, color='m'):
        a, b, c = line
        h, w = shape[:2]
        if abs(b) > 1e-6:
            x_vals = np.array([0, w])
            y_vals = -(a * x_vals + c) / b
        else:
            y_vals = np.array([0, h])
            x_vals = -(b * y_vals + c) / a
        ax.plot(x_vals, y_vals, color=color, linestyle='-', linewidth=2)

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_d_rgb = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
    axs[0].imshow(img_l_rgb)
    axs[0].set_title('Left Image (click here or right)')
    axs[1].imshow(img_d_rgb)
    axs[1].set_title('Right Image (click here or left)')
    plt.tight_layout()
    print("Click a point in either image (left or right). Close the window to exit.")
    while True:
        pts = plt.ginput(1, timeout=0)
        if not pts:
            break
        x, y = pts[0]
        w = img_l.shape[1]
        if x < w:
            # Clicked in left image
            x_img, y_img = x, y
            axs[0].scatter(x_img, y_img, color='red', s=60)
            p = np.array([x_img, y_img, 1])
            # Normalize point
            p_norm = np.linalg.inv(K) @ p
            l = E @ p_norm  # Epipolar line in right normalized coords
            # Convert line to pixel coords: l' = K^-T l
            l_pix = np.linalg.inv(K).T @ l
            draw_line(axs[1], l_pix, img_d.shape, color='m')
            axs[1].set_title('Right Image (epipolar line from E)')
        else:
            # Clicked in right image
            x_img, y_img = x - w, y
            axs[1].scatter(x_img, y_img, color='blue', s=60)
            p = np.array([x_img, y_img, 1])
            p_norm = np.linalg.inv(K) @ p
            l = E.T @ p_norm  # Epipolar line in left normalized coords
            l_pix = np.linalg.inv(K).T @ l
            draw_line(axs[0], l_pix, img_l.shape, color='m')
            axs[0].set_title('Left Image (epipolar line from E)')
        plt.draw()
    plt.close(fig)
    print("Interactive epipolar view (E) closed.")

def block_matching(left, right, max_disparity=64, kernel_size=5, use_subpixel=True):
    """
    Block matching algorithm for stereo disparity estimation.
    
    Args:
        left: Left image (grayscale)
        right: Right image (grayscale)
        max_disparity: Maximum disparity to search for
        kernel_size: Size of the matching window
        use_subpixel: Whether to use subpixel refinement
    
    Returns:
        disparity_map: Computed disparity map
    """
    # Get image dimensions
    height, width = left.shape
    
    # Initialize disparity map
    disparity_map = np.zeros((height, width), dtype=np.float32)
    
    # Calculate kernel half size
    kernel_half = kernel_size // 2
    
    # Convert images to float for better precision
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    print("block matching")
    # For each pixel in the image (except borders)
    for i in range(kernel_half, height - kernel_half):
        for j in range(max_disparity, width - kernel_half):
            # Initialize variables for best match
            best_offset = -1
            min_error = float('inf')
            errors = np.zeros(max_disparity)
            
            # For each possible disparity
            for offset in range(max_disparity):
                error = 0.0
                
                # Compute SSD over the window
                for x in range(-kernel_half, kernel_half + 1):
                    for y in range(-kernel_half, kernel_half + 1):
                        # Get pixel values from both images
                        left_val = left[i + x, j + y]
                        right_val = right[i + x, j + y - offset]
                        
                        # Compute squared difference
                        diff = left_val - right_val
                        error += diff * diff
                
                # Store error for this disparity
                errors[offset] = error
                
                # Update best match if this is better
                if error < min_error:
                    min_error = error
                    best_offset = offset
            
            # Subpixel refinement
            subpixel_offset = 0.0
            if use_subpixel and best_offset > 0 and best_offset < max_disparity - 1:
                error_left = errors[best_offset - 1]
                error_right = errors[best_offset + 1]
                
                # Avoid division by zero
                denominator = error_left - 2 * min_error + error_right
                if abs(denominator) > 1e-6:
                    subpixel_offset = 0.5 * (error_left - error_right) / denominator
            
            # Store final disparity
            disparity_map[i, j] = best_offset + subpixel_offset
    print("block matching finalizado")
    
    return disparity_map

def compute_disparity_map(left_img, right_img, max_disparity=100, kernel_size=5, use_subpixel=True):
    """ 
    Wrapper function to compute disparity map from RGB images.
    
    Args:
        left_img: Left RGB image
        right_img: Right RGB image
        max_disparity: Maximum disparity to search for
        kernel_size: Size of the matching window
        use_subpixel: Whether to use subpixel refinement
    
    Returns:
        disparity_map: Computed disparity map
    """
    print("compute_disparity_map")
    # Convert RGB to grayscale
    def rgb2gray(rgb):
        print("rgb2gray")
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    # Convert images to grayscale
    left_gray = rgb2gray(left_img)
    right_gray = rgb2gray(right_img)
    
    # Compute disparity map
    disparity_map = block_matching(
        left_gray, 
        right_gray,
        max_disparity=max_disparity,
        kernel_size=kernel_size,
        use_subpixel=use_subpixel
    )
    
    return disparity_map

def visualize_disparity(disparity_map, title="Disparity Map"):
    """
    Visualize the disparity map.
    
    Args:
        disparity_map: Computed disparity map
        title: Title for the plot
    """
    print("visualize_disparity")
    plt.figure(figsize=(10, 8))
    plt.imshow(disparity_map, cmap='plasma')
    plt.colorbar(label='Disparity')
    plt.title(title)
    plt.axis('off')
    plt.show()

def reconstruct_3d(disparity_map, K, baseline, left_img=None):
    """
    Reconstruct 3D points from disparity map.
    Args:
        disparity_map: (H, W) array of disparities
        K: Camera intrinsic matrix (3x3)
        baseline: Distance between cameras (in same units as K)
        left_img: Optional color image for coloring the point cloud
    Returns:
        points_3d: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (if left_img provided)
    """
    f = K[0, 0]  # Focal length in pixels
    cx = K[0, 2]
    cy = K[1, 2]
    h, w = disparity_map.shape
    
    # Create reprojection matrix
    Q = np.float32([[1, 0, 0, -cx],
                    [0, 1, 0, -cy],
                    [0, 0, 0, f],
                    [0, 0, -1/baseline, 0]])
    
    # Reproject to 3D
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    
    # Create mask for valid disparities
    mask = disparity_map > 0
    
    # Get valid 3D points
    points_3d = points_3d[mask]
    
    if left_img is not None:
        # Get colors for valid points
        colors = left_img[mask]
        return points_3d, colors
    return points_3d

def save_point_cloud(points_3d, colors, filename):
    """
    Save point cloud to PLY file.
    Args:
        points_3d: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors
        filename: Output PLY file name
    """
    # Create header
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points_3d)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    
    # Write to file
    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\n')
        for point, color in zip(points_3d, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

def visualize_point_cloud(points_3d, colors=None):
    """
    Visualize point cloud using matplotlib.
    Args:
        points_3d: (N, 3) array of 3D points
        colors: Optional (N, 3) array of RGB colors
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        # Normalize colors to [0,1]
        colors = colors / 255.0
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c=colors, s=1, alpha=0.5)
    else:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  s=1, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Point Cloud')
    plt.show()

def caso_20():
    """
    Reconstruct 3D scene from stereo images.
    """
    global img_l, img_d, K
    
    # Check if we have the camera matrix
    if K is None:
        print("ERROR: Camera matrix K is not available. Run camera calibration first.")
        return "Aborted: No camera matrix available."
    
    # Compute disparity map
    print("Computing disparity map...")
    disparity_map = compute_disparity_map(
        img_l, 
        img_d,
        max_disparity=100,
        kernel_size=5,
        use_subpixel=True
    )
    
    # Get baseline from user
    baseline = float(input("Enter baseline distance (in mm): "))
    
    # Reconstruct 3D points
    print("Reconstructing 3D points...")
    points_3d, colors = reconstruct_3d(disparity_map, K, baseline, img_l)
    
    # Ask user what to do with the point cloud
    print("\nWhat would you like to do with the point cloud?")
    print("1. Visualize")
    print("2. Save to PLY file")
    print("3. Both")
    choice = input("Enter your choice (1-3): ")
    
    if choice in ['1', '3']:
        print("Visualizing point cloud...")
        visualize_point_cloud(points_3d, colors)
    
    if choice in ['2', '3']:
        filename = input("Enter filename for PLY file (e.g., pointcloud.ply): ")
        print(f"Saving point cloud to {filename}...")
        save_point_cloud(points_3d, colors, filename)
    
    return "3D reconstruction completed."

def encontrar_mejor_punto(puntos_l, puntos_d):
    """
    Helper function to find the best points for rectification.
    This is a placeholder - you may need to implement your specific logic.
    """
    # Convert to homogeneous coordinates if needed
    if puntos_l.shape[1] == 2:
        puntos_l_h = np.hstack([puntos_l, np.ones((len(puntos_l), 1))])
    else:
        puntos_l_h = puntos_l
        
    if puntos_d.shape[1] == 2:
        puntos_d_h = np.hstack([puntos_d, np.ones((len(puntos_d), 1))])
    else:
        puntos_d_h = puntos_d
        
    return puntos_l_h, puntos_d_h

def apply_homographies_and_visualize(img_l, img_d, HL, HD):
    """
    Apply homographies to rectify stereo images and visualize the results.
    """
    # Get image dimensions
    h, w = img_l.shape[:2]
    
    # Apply homographies to rectify images
    img_l_rectified = cv2.warpPerspective(img_l, HL, (w, h))
    img_d_rectified = cv2.warpPerspective(img_d, HD, (w, h))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Left Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Original Right Image')
    axes[0, 1].axis('off')
    
    # Rectified images
    axes[1, 0].imshow(cv2.cvtColor(img_l_rectified, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Rectified Left Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(img_d_rectified, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Rectified Right Image')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show side-by-side comparison with horizontal lines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(cv2.cvtColor(img_l_rectified, cv2.COLOR_BGR2RGB))
    ax1.set_title('Rectified Left Image')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(img_d_rectified, cv2.COLOR_BGR2RGB))
    ax2.set_title('Rectified Right Image')
    ax2.axis('off')
    
    # Draw horizontal lines to verify rectification
    for y in range(50, h, 100):
        ax1.axhline(y=y, color='red', linewidth=1, alpha=0.7)
        ax2.axhline(y=y, color='red', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return img_l_rectified, img_d_rectified

def main():
    global RANSAC_THRESHOLD
    img_l = cv2.imread('cones/disp6.png', cv2.IMREAD_COLOR)
    img_d = cv2.imread('cones/disp2.png', cv2.IMREAD_COLOR)
    flag = True

    img_l_rect = None
    img_d_rect = None   
    
    # placeholders para usar en funciones
    K = R = tras = None
    puntos_clave_d = puntos_clave_l = None
    img_puntos_clave_d = img_puntos_clave_l = None
    F = E = puntos = puntos_e = None
    t = 10
    puntos_l = puntos_d = None
    robust_sift_ran = False
    ransac_ran = False
    E_computed = False
    HL = HD = None  # Add homography matrices for rectification

    while(flag):
        P = np.array([
        [1541.24, 0, 993.53, 0],  
        [0, 1538.17, 757.98, 0],  
        [0, 0, 1, 0]   
        ]) #Matriz Calibracion Jaime

        def caso_0():
            nonlocal flag
            flag = False

        def caso_1():
            nonlocal K, R, tras
            K, R, tras = krt_descomposition(P)
            return f"\nK =\n{K}\nR =\n{R}\nt =\n{tras}"

        def caso_2():
            return f"P reconstruida =\n{reconstruir_P(K, R, tras)}\n P original = \n{P}"
        
        def caso_3():
            nonlocal img_l, img_d, puntos_clave_l, puntos_clave_d, robust_sift_ran
            puntos_clave_l, puntos_clave_d, good, kp1, kp2 = robust_sift_matching(img_l, img_d)
            robust_sift_ran = True
            print(f"Robust SIFT matches found: {len(good)}")
            plot_sift_matches(img_l, img_d, puntos_clave_l, puntos_clave_d)
            return "Plotted robust SIFT matches (Lowe's ratio + cross-check)."

        def caso_4():
            nonlocal puntos_clave_l, puntos_clave_d
            puntos_clave_l, puntos_clave_d = manual_correspondences()
            print(f"Manual correspondences loaded: {len(puntos_clave_l)} pairs.")
            return "Manual correspondences loaded."

        def caso_5():
            global RANSAC_THRESHOLD
            t = float(input(f"Current RANSAC threshold is {RANSAC_THRESHOLD}. Enter new threshold: "))
            RANSAC_THRESHOLD = t
            print(f"RANSAC threshold set to {RANSAC_THRESHOLD}")
            return f"RANSAC threshold set to {RANSAC_THRESHOLD}"

        def caso_6():
            nonlocal F, puntos, robust_sift_ran, ransac_ran, puntos_l, puntos_d
            if not robust_sift_ran:
                print("WARNING: You should run option 19 (robust SIFT) before running RANSAC!")
                return "Aborted: Run option 19 first."
            r = 10000
            F, puntos = ransac(puntos_clave_l, puntos_clave_d, r, RANSAC_THRESHOLD)
            ransac_ran = True
            # Extract points from the RANSAC results
            if len(puntos) > 0:
                puntos_l = np.array([p[0] for p in puntos])
                puntos_d = np.array([p[1] for p in puntos])
            print(f"Rerun RANSAC with threshold {RANSAC_THRESHOLD}. Inliers: {len(puntos)}")
            print("Now run option 7 to compute the essential matrix E.")
            return f"RANSAC rerun with threshold {RANSAC_THRESHOLD}."

        def caso_7():
            nonlocal E, F, K, E_computed
            if F is None:
                print("WARNING: You should run RANSAC (option 22) before computing E!")
                return "Aborted: Run option 22 first."
            E = calcular_matriz_E(F,K)
            E_computed = True
            print("Now run option 16 to check diagnostics and visualize epipolar lines.")
            return f"Matriz Esencial =\n {E}"

        def caso_8():
            nonlocal img_l, img_d, F
            if F is None:
                print("You must run RANSAC (option 22) first.")
                return "Aborted: F not available."
            interactive_epipolar_view(img_l, img_d, F)
            return "Interactive epipolar view finished."

        def caso_9():
            nonlocal img_l, img_d, E, K
            if E is None or K is None:
                print("You must compute E (option 7) and have K available.")
                return "Aborted: E or K not available."
            interactive_epipolar_view_E(img_l, img_d, E, K)
            return "Interactive epipolar view with E finished."

        def caso_10():
            nonlocal img_l, img_d
            # Compute disparity map
            disparity_map = compute_disparity_map(
                img_l, 
                img_d,
                max_disparity=100,  # Adjust based on your needs
                kernel_size=5,      # Adjust based on your needs
                use_subpixel=True
            )
            
            # Visualize results
            visualize_disparity(disparity_map, "Computed Disparity Map")
            
            return "Disparity map computation completed."

        def caso_11():
            nonlocal F, E, K, puntos_l, puntos_d, E_computed
            if not E_computed:
                print("WARNING: You should compute E (option 7) before running diagnostics!")
                return "Aborted: Run option 7 first."
            check_matrix_properties(F, E, K, puntos_l, puntos_d)
            visualizar_epipolar_validation(img_l, img_d, F, puntos_l, puntos_d, E=E, K=K)
            return "Diagnostic check completed"

        def caso_12():
            nonlocal img_l, img_d, puntos_clave_l, puntos_clave_d
            plot_sift_matches(img_l, img_d, puntos_clave_l, puntos_clave_d)
            return "Plotted all SIFT matches."

        def caso_13():
            nonlocal img_l, img_d, puntos
            plot_inlier_matches(img_l, img_d, puntos)
            return "Plotted inlier matches after RANSAC."

        def caso_16():
            nonlocal puntos_clave_l, puntos_clave_d, F, HL, HD
            if puntos_clave_l is None or puntos_clave_d is None:
                print("ERROR: No keypoints available. Run SIFT matching first.")
                return "Aborted: No keypoints available."
            if F is None:
                print("ERROR: Fundamental matrix not available. Run RANSAC first.")
                return "Aborted: No fundamental matrix available."
                
            puntos_clave_l, puntos_clave_d = encontrar_mejor_punto(puntos_clave_l, puntos_clave_d)
            puntos_clave_l = puntos_clave_l[:, :2]
            puntos_clave_d = puntos_clave_d[:, :2]
            
            if len(puntos_clave_l) < 11:
                print("ERROR: Not enough points for rectification. Need at least 11 points.")
                return "Aborted: Not enough points."
                
            y_o = puntos_clave_l[10]
            y_o = [y_o[0] + 5, y_o[1] - 5]

            _, _, Vt = np.linalg.svd(F)
            eL = Vt[-1]
            eL = eL / eL[2]

            def skew(vec):
                return np.array([
                    [0, -vec[2], vec[1]],
                    [vec[2], 0, -vec[0]],
                    [-vec[1], vec[0], 0]
                ])

            v = np.ones(3)
            M = skew(eL) @ F + np.outer(eL, v)
            print("Matriz M = ", M)

            Ttrans = np.array([
                [1, 0, -y_o[0]],
                [0, 1, -y_o[1]],
                [0, 0, 1]
            ])

            eL_ = Ttrans @ eL

            theta = np.arctan2(eL_[1], eL_[0])
            Trot = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])

            eL_hat = Trot @ eL_
            H_inf = np.matrix([
                [1, 0 ,0],
                [0, 1, 0],
                [-1/eL_hat[0], 0, 1]
            ])
            HL = H_inf @ Trot @ Ttrans

            # Convertimos a homogéneos
            p_l_h = np.hstack([puntos_clave_l, np.ones((len(puntos_clave_l), 1))])
            p_r_h = np.hstack([puntos_clave_d, np.ones((len(puntos_clave_d), 1))])

            # Transformamos puntos
            yL_tilde = (HL @ p_l_h.T).T
            yL_tilde /= yL_tilde[:, 2:3]

            yR_tilde = (HL @ (M @ p_l_h.T)).T
            yR_tilde /= yR_tilde[:, 2:3]

            YL = yL_tilde
            uR = yR_tilde[:, 0]  # coordenadas u

            YR = yR_tilde
            uL = yL_tilde[:,0]
            Y_1 = YR.T @ YR    
            print("Matriz Y1 = ", Y_1)
            Y_1 = np.linalg.inv(Y_1)

            #a_vec = np.linalg.lstsq(YL, uR, rcond=None)[0]
            print(YR)
            print(f"\n")
            print(uL)
            a_vec = Y_1 @ (YR.T * uL) #(3x1)

            # Fix deprecation warnings by properly extracting scalar values
            a = float(a_vec[0])
            b = float(a_vec[1])
            c = float(a_vec[2])

            A = np.matrix([
                [a, b, c],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=float)
            HD = A @ HL @ M
            print(f"Matriz HL = {HL}")
            print(f"Matriz HD = {HD}")
            return f"Matriz HL = {HL},\n Matriz HD = {HD}"

        def caso_17():
            """
            Visualize the results of stereo rectification using computed homographies.
            """
            nonlocal img_l, img_d, HL, HD
            if HL is None or HD is None:
                print("ERROR: Homography matrices not available. Run option 16 first.")
                return "Aborted: No homography matrices available."
            
            print("Applying homographies and visualizing rectified images...")
            img_l_rect, img_d_rect = apply_homographies_and_visualize(img_l, img_d, HL, HD)
            
            # Ask if user wants to save rectified images
            save_choice = input("Do you want to save the rectified images? (y/n): ")
            if save_choice.lower() == 'y':
                cv2.imwrite('rectified_left.jpg', img_l_rect)
                cv2.imwrite('rectified_right.jpg', img_d_rect)
                print("Rectified images saved as 'rectified_left.jpg' and 'rectified_right.jpg'")
            
            return "Stereo rectification visualization completed."

        def caso_20():
            """
            Reconstruct 3D scene from stereo images.
            """
            nonlocal img_l_rect, img_d_rect, K
            
            # Check if we have the camera matrix
            if K is None:
                print("ERROR: Camera matrix K is not available. Run camera calibration first.")
                return "Aborted: No camera matrix available."
            
            # Compute disparity map
            print("Computing disparity map...")
            disparity_map = compute_disparity_map(
                img_l, 
                img_d,
                max_disparity=64,
                kernel_size=5,
                use_subpixel=True
            )
            
            # Get baseline from user
            baseline = float(input("Enter baseline distance (in mm): "))
            
            # Reconstruct 3D points
            print("Reconstructing 3D points...")
            points_3d, colors = reconstruct_3d(disparity_map, K, baseline, img_l)
            
            # Ask user what to do with the point cloud
            print("\nWhat would you like to do with the point cloud?")
            print("1. Visualize")
            print("2. Save to PLY file")
            print("3. Both")
            choice = input("Enter your choice (1-3): ")
            
            if choice in ['1', '3']:
                print("Visualizing point cloud...")
                visualize_point_cloud(points_3d, colors)
            
            if choice in ['2', '3']:
                filename = input("Enter filename for PLY file (e.g., pointcloud.ply): ")
                print(f"Saving point cloud to {filename}...")
                save_point_cloud(points_3d, colors, filename)
            
            return "3D reconstruction completed."

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
            "16": caso_16,
            "17": caso_17,
            "20": caso_20
        }

        opcion = input("Elige una opción (0-20): ")
        resultado = switch.get(opcion, lambda: "Opción no válida")()
        print(resultado)

if __name__ == '__main__':
    main()
