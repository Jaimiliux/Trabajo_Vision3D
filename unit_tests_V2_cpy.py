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
    
    # Show side-by-side comparison with horizontal lines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB))
    ax1.set_title('Rectified Left Image')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))
    ax2.set_title('Rectified Right Image')
    ax2.axis('off')
    
    # Draw horizontal lines to verify rectification
    for y in range(50, h, 100):
        ax1.axhline(y=y, color='red', linewidth=1, alpha=0.7)
        ax2.axhline(y=y, color='red', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return img_l, img_d

def robust_sift_matching(img_l, img_d, ratio_thresh=0.75):

    
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
            print(f"[F] Epipolar line in RIGHT image for point ({x_img:.2f}, {y_img:.2f}) in LEFT: a={l[0]:.6f}, b={l[1]:.6f}, c={l[2]:.6f}")
            draw_line(axs[1], l, img_d.shape, color='g')
            axs[1].set_title('Right Image (epipolar line shown)')
        else:
            # Clicked in right image
            x_img, y_img = x - w, y
            axs[1].scatter(x_img, y_img, color='blue', s=60)
            p = np.array([x_img, y_img, 1])
            l = F.T @ p  # Epipolar line in left image
            print(f"[F] Epipolar line in LEFT image for point ({x_img:.2f}, {y_img:.2f}) in RIGHT: a={l[0]:.6f}, b={l[1]:.6f}, c={l[2]:.6f}")
            draw_line(axs[0], l, img_l.shape, color='g')
            axs[0].set_title('Left Image (epipolar line shown)')
        plt.draw()
    plt.close(fig)
    print("Interactive epipolar view closed.")
    return "Interactive epipolar view finished."

def interactive_epipolar_view_E(img_l, img_d, E, K):


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
            p_norm = np.linalg.inv(K) @ p
            l = E @ p_norm  # Epipolar line in right normalized coords
            l_pix = np.linalg.inv(K).T @ l
            print(f"[E] Epipolar line in RIGHT image (pixels) for point ({x_img:.2f}, {y_img:.2f}) in LEFT: a={l_pix[0]:.6f}, b={l_pix[1]:.6f}, c={l_pix[2]:.6f}")
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
            print(f"[E] Epipolar line in LEFT image (pixels) for point ({x_img:.2f}, {y_img:.2f}) in RIGHT: a={l_pix[0]:.6f}, b={l_pix[1]:.6f}, c={l_pix[2]:.6f}")
            draw_line(axs[0], l_pix, img_l.shape, color='m')
            axs[0].set_title('Left Image (epipolar line from E)')
        plt.draw()
    plt.close(fig)
    print("Interactive epipolar view (E) closed.")
    return "Interactive epipolar view with E finished."

def compute_disparity_map_opencv(img_l, img_d, method='SGBM', **kwargs):
    """
    Compute disparity map using OpenCV's optimized stereo matchers.
    
    Args:
        img_l: Left image
        img_d: Right image  
        method: 'BM' for StereoBM or 'SGBM' for StereoSGBM
        **kwargs: Additional parameters for the stereo matcher
    
    Returns:
        disparity_map: Computed disparity map
    """
    # Convert to grayscale if needed
    if len(img_l.shape) == 3:
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY)
    else:
        gray_l = img_l.copy()
        gray_d = img_d.copy()
    
    if method == 'BM':
        # StereoBM parameters
        block_size = kwargs.get('block_size', 15)
        num_disparities = kwargs.get('num_disparities', 64)
        
        # Create StereoBM object
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        
        # Additional BM parameters
        stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        stereo.setPreFilterSize(kwargs.get('prefilter_size', 9))
        stereo.setPreFilterCap(kwargs.get('prefilter_cap', 31))
        stereo.setTextureThreshold(kwargs.get('texture_threshold', 10))
        stereo.setUniquenessRatio(kwargs.get('uniqueness_ratio', 15))
        stereo.setSpeckleRange(kwargs.get('speckle_range', 32))
        stereo.setSpeckleWindowSize(kwargs.get('speckle_window_size', 100))
        
    elif method == 'SGBM':
        # StereoSGBM parameters
        block_size = kwargs.get('block_size', 5)
        num_disparities = kwargs.get('num_disparities', 64)
        
        # Create StereoSGBM object
        stereo = cv2.StereoSGBM_create(
            minDisparity=kwargs.get('min_disparity', 0),
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=kwargs.get('P1', 8 * 3 * block_size**2),
            P2=kwargs.get('P2', 32 * 3 * block_size**2),
            disp12MaxDiff=kwargs.get('disp12_max_diff', 1),
            uniquenessRatio=kwargs.get('uniqueness_ratio', 10),
            speckleWindowSize=kwargs.get('speckle_window_size', 100),
            speckleRange=kwargs.get('speckle_range', 32),
            preFilterCap=kwargs.get('prefilter_cap', 63),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    else:
        raise ValueError("Method must be 'BM' or 'SGBM'")
    
    print(f"Computing disparity map using {method}...")
    
    # Compute disparity map
    disparity = stereo.compute(gray_l, gray_d).astype(np.float32) / 16.0
    
    # Filter out invalid disparities
    disparity[disparity <= 0] = 0
    
    print(f"Disparity map computed. Range: {disparity.min():.2f} to {disparity.max():.2f}")
    
    return disparity

def reconstruct_3d_opencv(disparity_map, Q_matrix, mask_disparity=True):
    """
    Reconstruct 3D points from disparity map using OpenCV's reprojectImageTo3D.
    
    Args:
        disparity_map: Disparity map from stereo matching
        Q_matrix: 4x4 reprojection matrix from stereo rectification
        mask_disparity: Whether to mask out invalid disparities
    
    Returns:
        points_3d: 3D points array (H, W, 3)
    """
    print("Reconstructing 3D points using OpenCV...")
    
    # Use OpenCV's optimized 3D reprojection
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q_matrix)
    
    if mask_disparity:
        # Mask out points with invalid disparity
        mask = disparity_map > 0
        points_3d[~mask] = 0
    
    print("3D reconstruction completed using OpenCV.")
    
    return points_3d

def create_reprojection_matrix(K, baseline):
    """
    Create the reprojection matrix Q for stereo reconstruction.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        baseline: Distance between camera centers
    
    Returns:
        Q: 4x4 reprojection matrix
    """
    fx = K[0, 0]
    fy = K[1, 1] 
    cx = K[0, 2]
    cy = K[1, 2]
    
    Q = np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy], 
        [0, 0, 0, fx],
        [0, 0, -1/baseline, 0]
    ])
    
    return Q

def save_point_cloud(points_3d, colors, filename):
    """
    Save point cloud to PLY file.
    
    Args:
        points_3d: Array of 3D points (N, 3)
        colors: Array of colors (N, 3) or None
        filename: Output filename
    """
    # Create header
    if colors is not None:
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
    else:
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points_3d)}",
            "property float x",
            "property float y",
            "property float z", 
            "end_header"
        ]
    
    # Write to file
    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\n')
        if colors is not None:
            for point, color in zip(points_3d, colors):
                f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        else:
            for point in points_3d:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

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

def visualize_disparity(disparity_map, title="Disparity Map"):
    """
    Visualize disparity map.
    """
    print("visualize_disparity")
    plt.figure(figsize=(10, 8))
    plt.imshow(disparity_map, cmap='plasma')
    plt.colorbar(label='Disparity')
    plt.title(title)
    plt.axis('off')
    plt.show()

def extract_valid_points_and_colors(points_3d, left_img, disparity_map, depth_threshold=1000):
    """
    Extract valid 3D points and their corresponding colors from the left image.
    
    Args:
        points_3d: 3D points array (H, W, 3)
        left_img: Left image for color extraction
        disparity_map: Disparity map for validity checking
        depth_threshold: Maximum depth to consider valid
    
    Returns:
        valid_points: Valid 3D points (N, 3)
        colors: Corresponding colors (N, 3)
    """
    h, w = disparity_map.shape
    
    # Create validity mask
    valid_mask = (disparity_map > 0) & (np.abs(points_3d[:, :, 2]) < depth_threshold)
    
    # Extract valid points
    valid_points = points_3d[valid_mask]
    
    # Extract corresponding colors
    if len(left_img.shape) == 3:
        colors = left_img[valid_mask]
    else:
        # If grayscale, replicate to RGB
        gray_colors = left_img[valid_mask]
        colors = np.column_stack([gray_colors, gray_colors, gray_colors])
    
    print(f"Extracted {len(valid_points)} valid 3D points")
    
    return valid_points, colors

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

def get_stereo_params(method='SGBM', quality='medium'):
    """
    Get optimized stereo matching parameters for different scenarios.
    
    Args:
        method: 'BM' or 'SGBM'
        quality: 'fast', 'medium', 'high'
    
    Returns:
        dict: Parameter dictionary for stereo matcher
    """
    if method == 'BM':
        if quality == 'fast':
            return {
                'num_disparities': 48,
                'block_size': 21,
                'prefilter_size': 9,
                'prefilter_cap': 31,
                'texture_threshold': 10,
                'uniqueness_ratio': 15,
                'speckle_range': 32,
                'speckle_window_size': 100
            }
        elif quality == 'medium':
            return {
                'num_disparities': 64,
                'block_size': 15,
                'prefilter_size': 9,
                'prefilter_cap': 31,
                'texture_threshold': 10,
                'uniqueness_ratio': 10,
                'speckle_range': 32,
                'speckle_window_size': 150
            }
        else:  # high quality
            return {
                'num_disparities': 96,
                'block_size': 11,
                'prefilter_size': 9,
                'prefilter_cap': 31,
                'texture_threshold': 10,
                'uniqueness_ratio': 5,
                'speckle_range': 16,
                'speckle_window_size': 200
            }
    else:  # SGBM
        if quality == 'fast':
            return {
                'num_disparities': 48,
                'block_size': 7,
                'P1': 8 * 3 * 7**2,
                'P2': 32 * 3 * 7**2,
                'disp12_max_diff': 2,
                'uniqueness_ratio': 15,
                'speckle_window_size': 50,
                'speckle_range': 16,
                'prefilter_cap': 63
            }
        elif quality == 'medium':
            return {
                'num_disparities': 64,
                'block_size': 5,
                'P1': 8 * 3 * 5**2,
                'P2': 32 * 3 * 5**2,
                'disp12_max_diff': 1,
                'uniqueness_ratio': 10,
                'speckle_window_size': 100,
                'speckle_range': 32,
                'prefilter_cap': 63
            }
        else:  # high quality
            return {
                'num_disparities': 96,
                'block_size': 3,
                'P1': 8 * 3 * 3**2,
                'P2': 32 * 3 * 3**2,
                'disp12_max_diff': 1,
                'uniqueness_ratio': 5,
                'speckle_window_size': 150,
                'speckle_range': 16,
                'prefilter_cap': 63
            }

def main():
    global RANSAC_THRESHOLD
    img_l = cv2.imread('im_i.png', cv2.IMREAD_COLOR)
    img_d = cv2.imread('im_d.png', cv2.IMREAD_COLOR)
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
                print("You must run RANSAC (option 6) first.")
                return "Aborted: F not available."
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
                w = img_l.shape[1]
                if x < w:
                    # Clicked in left image
                    x_img, y_img = x, y
                    axs[0].scatter(x_img, y_img, color='red', s=60)
                    p = np.array([x_img, y_img, 1])
                    l = F @ p  # Epipolar line in right image
                    print(f"[F] Epipolar line in RIGHT image for point ({x_img:.2f}, {y_img:.2f}) in LEFT: a={l[0]:.6f}, b={l[1]:.6f}, c={l[2]:.6f}")
                    draw_line(axs[1], l, img_d.shape, color='g')
                    axs[1].set_title('Right Image (epipolar line shown)')
                else:
                    # Clicked in right image
                    x_img, y_img = x - w, y
                    axs[1].scatter(x_img, y_img, color='blue', s=60)
                    p = np.array([x_img, y_img, 1])
                    l = F.T @ p  # Epipolar line in left image
                    print(f"[F] Epipolar line in LEFT image for point ({x_img:.2f}, {y_img:.2f}) in RIGHT: a={l[0]:.6f}, b={l[1]:.6f}, c={l[2]:.6f}")
                    draw_line(axs[0], l, img_l.shape, color='g')
                    axs[0].set_title('Left Image (epipolar line shown)')
                plt.draw()
            plt.close(fig)
            print("Interactive epipolar view closed.")
            return "Interactive epipolar view finished."

        def caso_9():
            nonlocal img_l, img_d, E, K
            if E is None or K is None:
                print("You must compute E (option 7) and have K available.")
                return "Aborted: E or K not available."
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
                    p_norm = np.linalg.inv(K) @ p
                    l = E @ p_norm  # Epipolar line in right normalized coords
                    l_pix = np.linalg.inv(K).T @ l
                    print(f"[E] Epipolar line in RIGHT image (pixels) for point ({x_img:.2f}, {y_img:.2f}) in LEFT: a={l_pix[0]:.6f}, b={l_pix[1]:.6f}, c={l_pix[2]:.6f}")
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
                    print(f"[E] Epipolar line in LEFT image (pixels) for point ({x_img:.2f}, {y_img:.2f}) in RIGHT: a={l_pix[0]:.6f}, b={l_pix[1]:.6f}, c={l_pix[2]:.6f}")
                    draw_line(axs[0], l_pix, img_l.shape, color='m')
                    axs[0].set_title('Left Image (epipolar line from E)')
                plt.draw()
            plt.close(fig)
            print("Interactive epipolar view (E) closed.")
            return "Interactive epipolar view with E finished."

        def caso_10():
            nonlocal img_l, img_d
            print("Choose disparity computation method:")
            print("1. StereoBM (faster, good for textured scenes)")
            print("2. StereoSGBM (slower, better quality)")
            method_choice = input("Enter choice (1-2): ")
            
            print("Choose quality preset:")
            print("1. Fast (lower quality, faster)")
            print("2. Medium (balanced)")
            print("3. High (better quality, slower)")
            quality_choice = input("Enter choice (1-3): ")
            
            method = 'BM' if method_choice == "1" else 'SGBM'
            quality_map = {'1': 'fast', '2': 'medium', '3': 'high'}
            quality = quality_map.get(quality_choice, 'medium')
            
            print(f"Using {method} with {quality} quality preset...")
            
            # Get optimized parameters
            params = get_stereo_params(method, quality)
            
            # Compute disparity map using OpenCV
            disparity_map = compute_disparity_map_opencv(img_l, img_d, method=method, **params)
            
            # Visualize results
            visualize_disparity(disparity_map, f"Disparity Map ({method} - {quality} quality)")
            
            return f"Disparity map computation completed using OpenCV {method} ({quality} quality)."

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

        def caso_14():
            """
            Visualize epipolar lines in the left image
            """
            nonlocal img_l, img_d, F, puntos_l, puntos_d
            if F is None or puntos_l is None or puntos_d is None or len(puntos_l) < 8:
                print("You must run RANSAC (option 6) first and have enough correspondences.")
                return "Aborted: F or correspondences not available."
            
            print("Plotting epipolar lines in the left image...")
            
            # Select a subset of points to avoid clutter
            num_lines = min(10, len(puntos_d))
            indices = np.random.choice(len(puntos_d), num_lines, replace=False)
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
            ax.imshow(img_l_rgb)
            ax.set_title('Left Image with Epipolar Lines')
            
            h, w = img_l.shape[:2]
            colors = plt.cm.rainbow(np.linspace(0, 1, num_lines))
            
            for i, idx in enumerate(indices):
                # Point in right image
                x2, y2 = puntos_d[idx]
                # Corresponding point in left image
                x1, y1 = puntos_l[idx]
                
                # Compute epipolar line in left image
                p2 = np.array([x2, y2, 1])
                l1 = F.T @ p2  # Epipolar line in left image
                
                # Draw the epipolar line
                if abs(l1[1]) > 1e-6:
                    x_line = np.array([0, w])
                    y_line = -(l1[0] * x_line + l1[2]) / l1[1]
                else:
                    y_line = np.array([0, h])
                    x_line = -(l1[1] * y_line + l1[2]) / l1[0]
                
                # Plot line and point
                ax.plot(x_line, y_line, color=colors[i], linewidth=2, alpha=0.7, 
                       label=f'Line {i+1}' if i < 5 else '')
                ax.plot(x1, y1, 'o', color=colors[i], markersize=8, markeredgecolor='white', 
                       markeredgewidth=2)
                
                # Add text annotation for the first few points
                if i < 5:
                    ax.annotate(f'P{i+1}({x1:.0f},{y1:.0f})', 
                              (x1, y1), xytext=(5, 5), textcoords='offset points',
                              fontsize=8, color='white', weight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))
            
            ax.legend(loc='upper right')
            ax.axis('off')
            plt.tight_layout()
            plt.show()
            
            print(f"Displayed {num_lines} epipolar lines in the left image.")
            print("Each line corresponds to a point in the right image.")
            print("The colored dots show the actual corresponding points in the left image.")
            
            return "Epipolar lines visualization in left image completed."

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
            Reconstruct 3D scene from stereo images using OpenCV.
            """
            nonlocal img_l, img_d, K
            
            # Check if we have the camera matrix
            if K is None:
                print("ERROR: Camera matrix K is not available. Run camera calibration first.")
                return "Aborted: No camera matrix available."
            
            print("Choose disparity computation method:")
            print("1. StereoBM (faster, good for textured scenes)")
            print("2. StereoSGBM (slower, better quality)")
            method_choice = input("Enter choice (1-2): ")
            
            print("Choose quality preset:")
            print("1. Fast (lower quality, faster)")
            print("2. Medium (balanced)")
            print("3. High (better quality, slower)")
            quality_choice = input("Enter choice (1-3): ")
            
            method = 'BM' if method_choice == "1" else 'SGBM'
            quality_map = {'1': 'fast', '2': 'medium', '3': 'high'}
            quality = quality_map.get(quality_choice, 'medium')
            
            print(f"Using {method} with {quality} quality preset...")
            
            # Get optimized parameters
            params = get_stereo_params(method, quality)
            
            # Compute disparity map using OpenCV
            print("Computing disparity map...")
            disparity_map = compute_disparity_map_opencv(img_l, img_d, method=method, **params)
            
            # Get baseline from user
            baseline = float(input("Enter baseline distance (in mm): "))
            
            # Create reprojection matrix
            Q_matrix = create_reprojection_matrix(K, baseline)
            
            # Reconstruct 3D points using OpenCV
            print("Reconstructing 3D points...")
            points_3d_full = reconstruct_3d_opencv(disparity_map, Q_matrix)
            
            # Extract valid points and colors
            valid_points, colors = extract_valid_points_and_colors(
                points_3d_full, img_l, disparity_map, depth_threshold=2000
            )
            
            # Ask user what to do with the point cloud
            print("\nWhat would you like to do with the point cloud?")
            print("1. Visualize")
            print("2. Save to PLY file")
            print("3. Both")
            print("4. Show disparity map first")
            choice = input("Enter your choice (1-4): ")
            
            if choice in ['4']:
                visualize_disparity(disparity_map, f"Disparity Map ({method} - {quality} quality)")
                choice = input("Now choose (1-3): ")
            
            if choice in ['1', '3']:
                print("Visualizing point cloud...")
                visualize_point_cloud(valid_points, colors)
            
            if choice in ['2', '3']:
                filename = input("Enter filename for PLY file (e.g., pointcloud.ply): ")
                print(f"Saving point cloud to {filename}...")
                save_point_cloud(valid_points, colors, filename)
            
            return f"3D reconstruction completed using OpenCV {method} ({quality} quality)."

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
            "14": caso_14,
            "16": caso_16,
            "17": caso_17,
            "20": caso_20
        }

        opcion = input("Elige una opción (0-20): ")
        resultado = switch.get(opcion, lambda: "Opción no válida")()
        print(resultado)

if __name__ == '__main__':
    main()
