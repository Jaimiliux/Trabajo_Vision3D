import numpy as np
import cv2 
import matplotlib.pyplot as plt
import sys
import random
import glob




def calibracion_camara_chessboard(image_files, chessboard_size=(7,7), square_size=2.4, show_corners=True):
    """
    Calibrates the camera using a set of chessboard images.
    Args:
        image_files: List of file paths to chessboard images.
        chessboard_size: Number of inner corners per chessboard row and column (cols, rows).
                        For an 8x8 chessboard, use (7,7) as it has 7x7 inner corners.
        square_size: Size of a square in your defined unit (e.g., millimeters).
        show_corners: If True, shows detected corners for each image.
    Returns:
        ret: RMS re-projection error.
        mtx: Camera matrix (intrinsics).
        dist: Distortion coefficients.
        rvecs: Rotation vectors.
        tvecs: Translation vectors.
    """
    # Prepare object points (0,0,0), (1,0,0), ..., (8,5,0) if chessboard_size=(9,6)
    objp = np.zeros((chessboard_size[1]*chessboard_size[0], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objp *= square_size

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            if show_corners:
                cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv2.imshow('Corners', img)
                cv2.waitKey(500)
        else:
            print(f"Chessboard not found in {fname}")

    if show_corners:
        cv2.destroyAllWindows()

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print("RMS re-projection error:", ret)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    return ret, mtx, dist, rvecs, tvecs

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

def normalize_points(pts):
    centroid = np.mean(pts, axis=0)
    pts_shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(pts_shifted**2, axis=1)))
    scale = np.sqrt(2) / mean_dist
    T = np.array([
        [scale, 0, -scale*centroid[0]],
        [0, scale, -scale*centroid[1]],
        [0, 0, 1]
    ])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T

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

    puntos_normalizados_l, T1 = normalize_points(puntos_clave_l)
    puntos_normalizados_d, T2 = normalize_points(puntos_clave_d)

    print("Empezamos RANSAC")
    for _ in range(iter):

        N = min(len(puntos_clave_l), len(puntos_clave_d))
        assert N >= 8, "No hay suficientes puntos para RANSAC"
        idx = random.sample(range(N), 8)

        sample_l = puntos_clave_l[idx]
        sample_d = puntos_clave_d[idx]

        sample_l, T_1 = normalize_points(sample_l)
        sample_d, T_2 = normalize_points(sample_d)

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
        puntos_l_inliers = np.array([p[0] for p in C_est_np])
        puntos_d_inliers = np.array([p[1] for p in C_est_np])
        puntos_l_norm, T1 = normalize_points(puntos_l_inliers)
        puntos_d_norm, T2 = normalize_points(puntos_d_inliers)
        F_final = eight_point_algorithm(puntos_l_norm, puntos_d_norm, T1, T2)
        puntos_l_list, puntos_d_list = zip(*C_est_np)
        puntos_l = np.vstack(puntos_l_list)
        puntos_d = np.vstack(puntos_d_list)
        return F_final, C_est_np
    else:
        return F_est, C_est_np

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

def robust_numpy_matching(des1, des2, ratio_thresh=0.75):
    """
    Robust SIFT matching using only NumPy:
    - Lowe's ratio test
    - Cross-check (mutual best)
    - Remove duplicates
    Returns: idx_l, idx_d (indices of matched keypoints in img_l and img_d)
    """
    # Forward matching: des1 -> des2
    matches_l = []
    matches_d = []
    for i, d1 in enumerate(des1):
        dists = np.linalg.norm(des2 - d1, axis=1)
        if len(dists) < 2:
            continue
        idx = np.argsort(dists)
        best, second_best = dists[idx[0]], dists[idx[1]]
        if best < ratio_thresh * second_best:
            matches_l.append(i)
            matches_d.append(idx[0])

    # Backward matching: des2 -> des1
    matches_l2 = []
    matches_d2 = []
    for j, d2 in enumerate(des2):
        dists = np.linalg.norm(des1 - d2, axis=1)
        if len(dists) < 2:
            continue
        idx = np.argsort(dists)
        best, second_best = dists[idx[0]], dists[idx[1]]
        if best < ratio_thresh * second_best:
            matches_l2.append(idx[0])
            matches_d2.append(j)

    # Cross-check: keep only mutual matches
    set_forward = set(zip(matches_l, matches_d))
    set_backward = set(zip(matches_l2, matches_d2))
    mutual_matches = list(set_forward & set_backward)

    # Remove duplicates (keep only one match per keypoint in img_l)
    seen_l = set()
    filtered_matches = []
    for l, d in mutual_matches:
        if l not in seen_l:
            filtered_matches.append((l, d))
            seen_l.add(l)

    idx_l = np.array([l for l, d in filtered_matches])
    idx_d = np.array([d for l, d in filtered_matches])
    return idx_l, idx_d

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

def block_matching(left, right, max_disparity=100, kernel_size=5, use_subpixel=True):
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

def filter_horizontal_matches(puntos_l, puntos_d, max_angle_deg=20):
    """
    Keep only matches where the correspondence vector is close to horizontal.
    max_angle_deg: maximum allowed angle (in degrees) from the x-axis.
    Returns filtered puntos_l, puntos_d.
    """
    vectors = puntos_d - puntos_l
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # in radians
    angles_deg = np.degrees(np.abs(angles))
    mask = angles_deg < max_angle_deg
    return puntos_l[mask], puntos_d[mask]

def plot_rectified_points(img_l, img_d, puntos_clave_l, puntos_clave_d, HL, HD):
    """
    Plots original and rectified correspondences.
    """
    # Convert to homogeneous
    p_l_h = np.hstack([puntos_clave_l, np.ones((len(puntos_clave_l), 1))])
    p_d_h = np.hstack([puntos_clave_d, np.ones((len(puntos_clave_d), 1))])

    # Apply rectification
    p_l_rect = (HL @ p_l_h.T).T
    p_l_rect /= p_l_rect[:, 2:3]
    p_d_rect = (HD @ p_d_h.T).T
    p_d_rect /= p_d_rect[:, 2:3]

    # Plot original points
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB))
    plt.scatter(puntos_clave_l[:, 0], puntos_clave_l[:, 1], c='r', label='Original Left')
    plt.title('Original Left Points')
    plt.axis('equal')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))
    plt.scatter(puntos_clave_d[:, 0], puntos_clave_d[:, 1], c='b', label='Original Right')
    plt.title('Original Right Points')
    plt.axis('equal')
    plt.legend()
    plt.show()

    # Plot rectified points
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(p_l_rect[:, 0], p_l_rect[:, 1], c='r', label='Rectified Left')
    plt.title('Rectified Left Points')
    plt.axis('equal')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(p_d_rect[:, 0], p_d_rect[:, 1], c='b', label='Rectified Right')
    plt.title('Rectified Right Points')
    plt.axis('equal')
    plt.legend()
    plt.show()

    # Optionally, plot lines between rectified correspondences
    plt.figure(figsize=(12, 6))
    for i in range(len(p_l_rect)):
        plt.plot([p_l_rect[i, 0], p_d_rect[i, 0]], [p_l_rect[i, 1], p_d_rect[i, 1]], 'g-')
    plt.scatter(p_l_rect[:, 0], p_l_rect[:, 1], c='r', label='Rectified Left')
    plt.scatter(p_d_rect[:, 0], p_d_rect[:, 1], c='b', label='Rectified Right')
    plt.title('Rectified Correspondences')
    plt.axis('equal')
    plt.legend()
    plt.show()

def draw_epipolar_lines(img1, img2, F, pts1, pts2, num_lines=10):
    """
    Draw epipolar lines on img1 and img2 for the given point correspondences.
    Only uses NumPy and matplotlib.
    """
    img1 = img1.copy()
    img2 = img2.copy()
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(img1)
    ax2.imshow(img2)

    # Pick a subset if too many
    idxs = np.random.choice(len(pts1), min(num_lines, len(pts1)), replace=False)

    for i in idxs:
        p1 = np.array([*pts1[i], 1.0])
        p2 = np.array([*pts2[i], 1.0])

        # Epipolar line in right image for p1
        l2 = F @ p1
        # Epipolar line in left image for p2
        l1 = F.T @ p2

        # Draw point in left image
        ax1.plot(p1[0], p1[1], 'ro')
        # Draw point in right image
        ax2.plot(p2[0], p2[1], 'bo')

        # Draw epipolar line in right image
        x = np.linspace(0, w2, 100)
        y = -(l2[0] * x + l2[2]) / (l2[1] + 1e-12)
        ax2.plot(x, y, 'm-')

        # Draw epipolar line in left image
        x = np.linspace(0, w1, 100)
        y = -(l1[0] * x + l1[2]) / (l1[1] + 1e-12)
        ax1.plot(x, y, 'g-')

    ax1.set_title('Left Image with Epipolar Lines')
    ax2.set_title('Right Image with Epipolar Lines')
    plt.tight_layout()
    plt.show()

def mean_epipolar_error(F, pts1, pts2):
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    errors = []
    for p1, p2 in zip(pts1_h, pts2_h):
        err = np.abs(p2 @ F @ p1)
        errors.append(err)
    return np.mean(errors)

def analyze_epipolar_geometry(img_l, img_d, F, E, K, puntos_l, puntos_d, num_points=10):
    """
    Visualize epipolar geometry, epipoles, and epipolar lines using F and E.
    Draws mean epipolar error for each.
    """
    import matplotlib.pyplot as plt
    # --- Compute epipoles ---
    # Left epipole (F)
    _, _, Vt = np.linalg.svd(F)
    eL = Vt[-1]
    eL = eL / eL[2]
    # Right epipole (F)
    _, _, Vt = np.linalg.svd(F.T)
    eR = Vt[-1]
    eR = eR / eR[2]
    # Left epipole (E)
    _, _, Vt = np.linalg.svd(E)
    eL_E = Vt[-1]
    eL_E = eL_E / eL_E[2]
    # Right epipole (E)
    _, _, Vt = np.linalg.svd(E.T)
    eR_E = Vt[-1]
    eR_E = eR_E / eR_E[2]

    # --- Plot epipolar lines and points ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_d_rgb = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
    axs[0,0].imshow(img_l_rgb)
    axs[0,1].imshow(img_d_rgb)
    axs[1,0].imshow(img_l_rgb)
    axs[1,1].imshow(img_d_rgb)
    axs[0,0].set_title('Left Image (F)')
    axs[0,1].set_title('Right Image (F)')
    axs[1,0].set_title('Left Image (E)')
    axs[1,1].set_title('Right Image (E)')

    # Draw points and epipoles
    for ax, pts, epi, color, label in [
        (axs[0,0], puntos_l, eL, 'r', 'Epipole F'),
        (axs[0,1], puntos_d, eR, 'b', 'Epipole F'),
        (axs[1,0], puntos_l, eL_E, 'm', 'Epipole E'),
        (axs[1,1], puntos_d, eR_E, 'c', 'Epipole E')]:
        ax.scatter(pts[:,0], pts[:,1], c=color, s=30, label='Points')
        if np.isfinite(epi[0]) and np.isfinite(epi[1]):
            ax.plot(epi[0], epi[1], marker='*', color='y', markersize=15, label=label)
        ax.legend()

    # Draw epipolar lines for a subset of points
    idxs = np.random.choice(len(puntos_l), min(num_points, len(puntos_l)), replace=False)
    h, w = img_l.shape[:2]
    for i in idxs:
        # --- F ---
        p1 = np.array([*puntos_l[i], 1.0])
        p2 = np.array([*puntos_d[i], 1.0])
        l2 = F @ p1  # Epipolar line in right image
        l1 = F.T @ p2  # Epipolar line in left image
        # Draw on left
        x = np.linspace(0, w, 100)
        y = -(l1[0] * x + l1[2]) / (l1[1] + 1e-12)
        axs[0,0].plot(x, y, 'g-', alpha=0.7)
        # Draw on right
        x = np.linspace(0, w, 100)
        y = -(l2[0] * x + l2[2]) / (l2[1] + 1e-12)
        axs[0,1].plot(x, y, 'g-', alpha=0.7)
        # --- E ---
        # Normalize points
        p1n = np.linalg.inv(K) @ p1
        p2n = np.linalg.inv(K) @ p2
        l2_E = E @ p1n
        l1_E = E.T @ p2n
        # Convert lines to pixel coords
        l2_E_pix = np.linalg.inv(K).T @ l2_E
        l1_E_pix = np.linalg.inv(K).T @ l1_E
        # Draw on left
        x = np.linspace(0, w, 100)
        y = -(l1_E_pix[0] * x + l1_E_pix[2]) / (l1_E_pix[1] + 1e-12)
        axs[1,0].plot(x, y, 'y-', alpha=0.7)
        # Draw on right
        x = np.linspace(0, w, 100)
        y = -(l2_E_pix[0] * x + l2_E_pix[2]) / (l2_E_pix[1] + 1e-12)
        axs[1,1].plot(x, y, 'y-', alpha=0.7)

    # --- Compute mean epipolar error for F and E ---
    # For F
    pts1_h = np.hstack([puntos_l, np.ones((puntos_l.shape[0], 1))])
    pts2_h = np.hstack([puntos_d, np.ones((puntos_d.shape[0], 1))])
    errors_F = np.abs(np.sum(pts2_h * (F @ pts1_h.T).T, axis=1))
    mean_err_F = np.mean(errors_F)
    # For E (normalize points)
    pts1n = (np.linalg.inv(K) @ pts1_h.T).T
    pts2n = (np.linalg.inv(K) @ pts2_h.T).T
    errors_E = np.abs(np.sum(pts2n * (E @ pts1n.T).T, axis=1))
    mean_err_E = np.mean(errors_E)

    # Show mean errors
    axs[0,0].set_xlabel(f"Mean epipolar error (F): {mean_err_F:.4f}")
    axs[1,0].set_xlabel(f"Mean epipolar error (E): {mean_err_E:.4f}")
    plt.tight_layout()
    plt.show()

    print(f"Mean epipolar error (F): {mean_err_F:.4f}")
    print(f"Mean epipolar error (E): {mean_err_E:.4f}")

def plot_combined_epipolar_lines(img_l, img_d, F, E, K, puntos_l, puntos_d, num_points=10):
    """
    For each correspondence, compute the epipolar line in the right image using F and E,
    then plot both and their average.
    """
    img_d_rgb = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
    h, w = img_d.shape[:2]
    plt.figure(figsize=(10, 8))
    plt.imshow(img_d_rgb)
    idxs = np.random.choice(len(puntos_l), min(num_points, len(puntos_l)), replace=False)
    for i in idxs:
        p1 = np.array([*puntos_l[i], 1.0])
        # --- Epipolar line from F ---
        l_F = F @ p1
        # --- Epipolar line from E ---
        p1n = np.linalg.inv(K) @ p1
        l_E = E @ p1n
        l_E_pix = np.linalg.inv(K).T @ l_E
        # --- Average line ---
        l_avg = (l_F + l_E_pix) / 2

        # Plot all three lines
        for l, color, label in zip([l_F, l_E_pix, l_avg], ['g', 'y', 'r'], ['F', 'E', 'Avg']):
            x = np.linspace(0, w, 100)
            y = -(l[0] * x + l[2]) / (l[1] + 1e-12)
            plt.plot(x, y, color+'-', alpha=0.7, label=label if i == idxs[0] else "")

        # Plot the corresponding point
        p2 = puntos_d[i]
        plt.plot(p2[0], p2[1], 'bo')

    # Plot the right epipole for F and E
    _, _, Vt = np.linalg.svd(F.T)
    eR = Vt[-1]
    eR = eR / eR[2]
    _, _, Vt = np.linalg.svd(E.T)
    eR_E = Vt[-1]
    eR_E = eR_E / eR_E[2]
    plt.plot(eR[0], eR[1], 'k*', markersize=15, label='Epipole F')
    plt.plot(eR_E[0], eR_E[1], 'm*', markersize=15, label='Epipole E')

    plt.title('Epipolar lines in right image (F: green, E: yellow, Avg: red)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_average_pose(rvecs, tvecs):
    """
    Compute average rotation and translation from calibration data.
    Args:
        rvecs: List of rotation vectors
        tvecs: List of translation vectors
    Returns:
        R_avg: Average rotation matrix
        t_avg: Average translation vector
    """
    # Convert rotation vectors to matrices
    R_mats = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
    
    # Compute average rotation using SVD
    R_avg = np.mean(R_mats, axis=0)
    U, _, Vt = np.linalg.svd(R_avg)
    R_avg = U @ Vt
    
    # Compute average translation
    t_avg = np.mean(tvecs, axis=0)
    
    return R_avg, t_avg

def main():
    global RANSAC_THRESHOLD
    img_l = cv2.imread('cones/disp6.png', cv2.IMREAD_COLOR)
    img_d = cv2.imread('cones/disp2.png', cv2.IMREAD_COLOR)
    flag = True

    # Calibration data
    rvecs = [
        np.array([[-0.09227338], [-0.36235900], [3.10268713]]),
        np.array([[-0.41866659], [-0.18428394], [1.19851205]]),
        np.array([[-0.26262245], [-0.11404684], [1.00970379]]),
        np.array([[-0.04503020], [-0.06757056], [-0.00637284]]),
        np.array([[0.10873015], [0.20834482], [1.56105980]]),
        np.array([[-0.34481281], [-0.11159564], [-0.04467941]]),
        np.array([[0.23869455], [-0.15919542], [0.03771881]]),
        np.array([[-0.09111583], [-0.35883380], [0.04179396]]),
        np.array([[-0.21390328], [0.07240139], [1.32544176]]),
        np.array([[-0.08502491], [0.07475942], [0.98736422]]),
        np.array([[-0.03280915], [-0.00129823], [1.54388441]]),
        np.array([[-0.29988674], [-0.06028035], [0.01604136]])
    ]
    
    tvecs = [
        np.array([[9.12563761], [3.95969424], [34.53141763]]),
        np.array([[2.87694656], [-10.65463504], [36.84418433]]),
        np.array([[1.39315532], [-12.12075440], [33.31305599]]),
        np.array([[-6.01539986], [-10.85007713], [38.01079557]]),
        np.array([[8.26934248], [-9.39338461], [30.52018179]]),
        np.array([[-6.39143705], [-8.38061939], [25.02447204]]),
        np.array([[-4.39138078], [-6.03976556], [24.39888876]]),
        np.array([[-7.44750747], [-7.50822184], [24.82066360]]),
        np.array([[4.05155829], [-6.06095640], [51.93734378]]),
        np.array([[2.85265334], [-11.40616315], [35.67331428]]),
        np.array([[6.69267849], [-9.69069973], [35.08191147]]),
        np.array([[-5.79303759], [-10.03286372], [33.01782364]])
    ]
    
    # Compute average pose
    R_avg, t_avg = compute_average_pose(rvecs, tvecs)
    
    # Updated camera matrix from calibration with average pose
    K = np.array([
        [1.55762650e+03, 0.00000000e+00, 1.00726807e+03],
        [0.00000000e+00, 1.55603801e+03, 7.45444599e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    
    P = np.hstack([K @ R_avg, K @ t_avg])
    
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
    
    while(flag):
        # Updated camera matrix from calibration
        P = np.array([
            [1.55762650e+03, 0.00000000e+00, 1.00726807e+03, -0.09227338],  
            [0.00000000e+00, 1.55603801e+03, 7.45444599e+02, -0.36235900],  
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.10268713]   
        ]) #Matriz Calibracion actualizada con rotación y traslación

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
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img_l, None)
            kp2, des2 = sift.detectAndCompute(img_d, None)
            idx_l, idx_d = robust_numpy_matching(des1, des2)
            puntos_clave_l = np.array([kp1[i].pt for i in idx_l], dtype=np.float32)
            puntos_clave_d = np.array([kp2[i].pt for i in idx_d], dtype=np.float32)
            # Filter by angle (horizontal matches only)
            puntos_clave_l, puntos_clave_d = filter_horizontal_matches(puntos_clave_l, puntos_clave_d, max_angle_deg=20)
            robust_sift_ran = True
            print(f"Robust SIFT matches found (horizontal only): {len(puntos_clave_l)}")
            plot_sift_matches(img_l, img_d, puntos_clave_l, puntos_clave_d)
            return "Plotted robust SIFT matches (NumPy matcher, cross-checked, horizontal only)."

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
                print("WARNING: You should run option 3 (robust SIFT) before running RANSAC!")
                return "Aborted: Run option 3 first."
            r = 50000
            F, puntos = ransac(puntos_clave_l, puntos_clave_d, r, RANSAC_THRESHOLD)
            ransac_ran = True
            # Extract points from the RANSAC results
            if len(puntos) > 7:
                puntos_l = np.array([p[0] for p in puntos])
                puntos_d = np.array([p[1] for p in puntos])
            else:
                print("No se han encontrado suficientes puntos para calcular la matriz fundamental")
                caso_6()
            print(f"Rerun RANSAC with threshold {RANSAC_THRESHOLD}. Inliers: {len(puntos)}")
            print("Now run option 7 to compute the essential matrix E.")
            return f"RANSAC rerun with threshold {RANSAC_THRESHOLD}."

        def caso_7():
            nonlocal E, F, K, E_computed
            if F is None:
                print("WARNING: You should run RANSAC (option 6) before computing E!")
                return "Aborted: Run option 6 first."
            E = calcular_matriz_E(F,K)
            E_computed = True
            print("Now run option 10 to check diagnostics and visualize epipolar lines.")
            return f"Matriz Esencial =\n {E}"

        def caso_8():
            nonlocal img_l, img_d, F
            if F is None:
                print("You must run RANSAC (option 6) first.")
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
            nonlocal F, E, K, puntos_l, puntos_d, E_computed
            if not E_computed:
                print("WARNING: You should compute E (option 7) before running diagnostics!")
                return "Aborted: Run option 7 first."
            check_matrix_properties(F, E, K, puntos_l, puntos_d)
            visualizar_epipolar_validation(img_l, img_d, F, puntos_l, puntos_d, E=E, K=K)
            print("Mean epipolar error:", mean_epipolar_error(F, puntos_l, puntos_d))
            return "Diagnostic check completed"
        
        def caso_11():
            nonlocal puntos_clave_l, puntos_clave_d, F, img_l, img_d

            if puntos_clave_l is None or puntos_clave_d is None or len(puntos_clave_l) < 11:
                print("Not enough points for rectification. Need at least 11 correspondences.")
                return "Aborted: Not enough correspondences."

            # Use only (x, y)
            pts1 = puntos_clave_l[:, :2]
            pts2 = puntos_clave_d[:, :2]

            # Compute left epipole
            _, _, Vt = np.linalg.svd(F)
            eL = Vt[-1]
            eL = eL / eL[2]

            # Compute translation to move a point to the origin
            y_o = pts1[10]
            Ttrans = np.array([
                [1, 0, -y_o[0]],
                [0, 1, -y_o[1]],
                [0, 0, 1]
            ])
            eL_ = Ttrans @ eL

            # Rotation to align epipole with x-axis
            theta = np.arctan2(eL_[1], eL_[0])
            Trot = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            eL_hat = Trot @ eL_
            if np.abs(eL_hat[0]) < 1e-6:
                print("Epipole is at infinity or too close to y-axis.")
                return "Aborted: Epipole problem."

            # Projective transform to send epipole to infinity
            H_inf = np.array([
                [1, 0 ,0],
                [0, 1, 0],
                [-1/eL_hat[0], 0, 1]
            ])
            HL = H_inf @ Trot @ Ttrans

            # Transform points
            p_l_h = np.hstack([pts1, np.ones((len(pts1), 1))])
            p_r_h = np.hstack([pts2, np.ones((len(pts2), 1))])
            yL_tilde = (HL @ p_l_h.T).T
            yL_tilde /= yL_tilde[:, 2:3]

            # Now, find a homography for the right image (HD) that aligns the y-coordinates
            # Use least-squares to fit a 1D affine transform: yL_tilde[:,1] ≈ a*yR_tilde[:,1] + b
            # For simplicity, use identity for HD (not optimal, but avoids explosion)
            HD = np.eye(3)

            # Plot
            plot_rectified_points(img_l, img_d, pts1, pts2, HL, HD)
            return f"Matriz HL = {HL},\n Matriz HD = {HD}"
            

        def caso_12():
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
            

        def caso_13():
            nonlocal img_l, img_d, puntos_clave_l, puntos_clave_d
            plot_sift_matches(img_l, img_d, puntos_clave_l, puntos_clave_d)
            return "Plotted all SIFT matches."

        def caso_14():
            nonlocal img_l, img_d, puntos
            plot_inlier_matches(img_l, img_d, puntos)
            return "Plotted inlier matches after RANSAC."

        def caso_15():
            draw_epipolar_lines(img_l, img_d, F, puntos_l, puntos_d)
            

        def caso_16():
            nonlocal img_l, img_d, F, E, K, puntos_l, puntos_d
            if puntos_l is None or puntos_d is None or len(puntos_l) < 8:
                print("You must run RANSAC (option 6) first to get good correspondences.")
                return "Aborted: Run RANSAC first."
            analyze_epipolar_geometry(img_l, img_d, F, E, K, puntos_l, puntos_d)
            return "Epipolar geometry analysis completed."

        def caso_17():
            nonlocal img_l, img_d, F, E, K, puntos_l, puntos_d
            if puntos_l is None or puntos_d is None or len(puntos_l) < 8:
                print("You must run RANSAC (option 6) first to get good correspondences.")
                return "Aborted: Run RANSAC first."
            plot_combined_epipolar_lines(img_l, img_d, F, E, K, puntos_l, puntos_d)
            return "Epipolar lines plotted."

        def caso_18():
            nonlocal img_l, img_d
            image_files = glob.glob('Muestra/*.jpeg')  # Path to your chessboard images
            ret, mtx, dist, rvecs, tvecs = calibracion_camara_chessboard(
                image_files, chessboard_size=(7,7), square_size=2.4, show_corners=True
            )
            print(f"RMS re-projection error: {ret}")
            print(f"Camera matrix:\n{mtx}")
            print(f"Distortion coefficients:\n{dist}")
            print(f"Rotation vectors:\n{rvecs}")
            print(f"Translation vectors:\n{tvecs}")
            return "Calibración de la cámara completada."

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
            "15": caso_15,
            "16": caso_16,
            "17": caso_17,
            "18": caso_18
        }

        opcion = input("Elige una opción (0-17): ")
        resultado = switch.get(opcion, lambda: "Opción no válida")()
        print(resultado)


if __name__ == '__main__':
    main()

