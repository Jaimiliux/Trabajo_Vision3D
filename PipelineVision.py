import numpy as np
import cv2 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import sys
import random
import open3d as o3d


def rq_decomposition_numpy(A):

    A_flip = np.flipud(A).T
    Q, R = np.linalg.qr(A_flip)
    R = np.flipud(R.T)
    Q = Q.T[:, ::-1]
    for i in range(3):
        if R[i, i] < 0:
            R[:, i] *= -1
            Q[i, :] *= -1
    return R, Q

def krt_descomposition(P):
    
    M = P[:, :3]
    
    #Comprobar si es triangular superior
    if np.allclose(M, np.triu(M)):
        K = M.copy()
        R = np.eye(3)
    else:
        #Descomposicion RQ
        K, R = rq_decomposition_numpy(M)
        
        #Comprobar caso degenerado
        if np.abs(K[2,2]) < 1e-8:
            raise ValueError("K[2,2] is zero after RQ decomposition. Matrix P may be degenerate.")
        
        #Normalizar
        K = K / K[2, 2]
        
        #Asegurar que R sea matriz de rotacion
        if np.linalg.det(R) < 0:
            R = -R
            K = -K
    
    #Extraer vector de traslacion
    t = np.linalg.inv(K) @ P[:, 3]
    return K, R, t


def construir_matriz_A(points_l, points_d):
    A = []
    for (x1, y1), (x2, y2) in zip(points_l, points_d):
        A.append([
            x1 * x2, y1 * x2, x2,
            x1 * y2, y1 * y2, y2,
            x1, y1, 1
        ])
    return np.array(A)

def normalizar_puntos(puntos):
     #Calcular el centroide
    centroide = np.mean(puntos, axis=0)
    #Desplazar los puntos al centroide
    puntos_desplazados = puntos - centroide
    #Calcular la distancia media desde el centroide
    distancia_media = np.mean(np.sqrt(np.sum(puntos_desplazados**2, axis=1)))
    #Escalar para que la distancia distancia_mediab sea sqrt(2)
    factor_escala = np.sqrt(2) / distancia_media
    #Matriz de transformación
    T = np.array([[factor_escala, 0, -factor_escala * centroide[0]],
                  [0, factor_escala, -factor_escala * centroide[1]],
                  [0, 0, 1]])
    #Aplicar la transformación a los puntos
    puntos_homogeneos = np.hstack((puntos, np.ones((puntos.shape[0], 1))))
    puntos_normalizados = (T @ puntos_homogeneos.T).T
    return puntos_normalizados[:, :2], T


def eight_point_algorithm(points_l, points_d, T1, T2):
   
    if T1 is None or T2 is None:
        points_l_norm, T1 = normalizar_puntos(points_l)
        points_d_norm, T2 = normalizar_puntos(points_d)
    else:
        points_l_norm = points_l
        points_d_norm = points_d

    A = construir_matriz_A(points_l_norm, points_d_norm)
    
    #Encontramos F
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    #Asegurar que F tenga rango 2
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Set smallest singular value to 0
    F = U @ np.diag(S) @ Vt
    
    #Desnormalizar
    F = T2.T @ F @ T1
    
    #Normalizar
    F = F / np.linalg.norm(F)
    
    return F

def verify_fundamental_matrix(F, puntos_l, puntos_d):
    #(Error Geometrico)
    #Comprobar rango de F
    rank = np.linalg.matrix_rank(F)
    #print(f"\nFundamental Matrix Rank: {rank}")
    
    #Comprobar determinante (debe ser cercano a 0)
    det = np.linalg.det(F)
    #print(f"Fundamental Matrix Determinant: {det}")
    
    #Comprobar restriccion epipolar
    puntos_l_h = np.hstack((puntos_l, np.ones((puntos_l.shape[0], 1))))
    puntos_d_h = np.hstack((puntos_d, np.ones((puntos_d.shape[0], 1))))

    #Encontrar lineas epipolares
    lines_l = (F.T @ puntos_d_h.T).T  # Lines in left image
    lines_r = (F @ puntos_l_h.T).T    # Lines in right image
    
    #Normalizar lineas
    norm_l = np.sqrt(np.sum(lines_l[:, :2]**2, axis=1, keepdims=True))
    norm_r = np.sqrt(np.sum(lines_r[:, :2]**2, axis=1, keepdims=True))
    
    #Evitar division por cero
    norm_l = np.where(norm_l > 1e-10, norm_l, 1.0)
    norm_r = np.where(norm_r > 1e-10, norm_r, 1.0)
    
    lines_l = lines_l / norm_l
    lines_r = lines_r / norm_r
    
    #Encontrar distancias a lineas epipolares
    dist_l = np.abs(np.sum(puntos_l_h * lines_l, axis=1))
    dist_r = np.abs(np.sum(puntos_d_h * lines_r, axis=1))
    
    #Encontrar error epipolar simetrico
    errors = (dist_l + dist_r) / 2
    '''
    print("\nEpipolar Error Statistics:")
    print(f"Mean error: {np.mean(errors):.3f}")
    print(f"Median error: {np.median(errors):.3f}")
    print(f"Max error: {np.max(errors):.3f}")
    print(f"Min error: {np.min(errors):.3f}")
    '''    
    return errors

def ransac(puntos_clave_l, puntos_clave_d, iter, t):
    F_estimada = None
    C_est = []
    C = []
    max_inliers = 0
    best_inlier_ratio = 0
    no_improvement_count = 0
    min_iterations = 2000
    
    print("Empezamos RANSAC")
    for i in range(iter):
        try:
            #Seleccionamos 8 puntos aleatorios
            idx = random.sample(range(len(puntos_clave_l)), 8)
            sample_l = puntos_clave_l[idx]
            sample_d = puntos_clave_d[idx]
            
            #Encontramos F
            F = eight_point_algorithm(sample_l, sample_d, None, None)
            
            #Comprobamos F
            errors = verify_fundamental_matrix(F, puntos_clave_l, puntos_clave_d)
            inliers = errors < t
            inlier_count = np.sum(inliers)
            
            #Actualizar mejor solucion
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                F_estimada = F
                C_est = [(tuple(p1), tuple(p2)) for p1, p2, valido in zip(puntos_clave_l, puntos_clave_d, inliers) if valido]
                best_inlier_ratio = inlier_count / len(puntos_clave_l)
                print(f"New best solution found with {max_inliers} inliers (ratio: {best_inlier_ratio:.3f})")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if i >= min_iterations:
                if best_inlier_ratio > 0.9:
                    print("Early stopping: Very good solution found (>90% inliers)")
                    break
                    
        except np.linalg.LinAlgError:
            print(f"Warning: Linear algebra error in iteration {i}")
            continue
    
    if F_estimada is None:
        print("Warning: RANSAC failed to find a good solution")
        return None, None
    
    return F_estimada, np.array(C_est)

def interactive_epipolar_view(img_l, img_d, F):
    
    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_d_rgb = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
    
    #Encontrar epipoles
    U, S, Vt = np.linalg.svd(F)
    e1 = Vt[-1, :]  # Right epipole
    e2 = U[:, -1]   # Left epipole
    
    #Convertir epipoles a coordenadas no homogeneas
    if abs(e1[2]) > 1e-10:
        e1 = e1[:2] / e1[2]
    else:
        print("Warning: Right epipole is at infinity")
        e1 = e1[:2] * 1e10
        
    if abs(e2[2]) > 1e-10:
        e2 = e2[:2] / e2[2]
    else:
        print("Warning: Left epipole is at infinity")
        e2 = e2[:2] * 1e10
    
    print(f"\nEpipoles (in image coordinates):")
    print(f"Left epipole: {e2}")
    print(f"Right epipole: {e1}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.imshow(img_l_rgb)
    ax1.set_title('Left Image - Click here')
    ax1.axis('off')
    
    ax2.imshow(img_d_rgb)
    ax2.set_title('Right Image - Epipolar Lines')
    ax2.axis('off')
    
    clicked_points = []
    colors = plt.cm.rainbow(np.linspace(0, 1, 100))  
    def onclick(event):
        if event.inaxes == ax1:  
            #Encontrar punto
            x, y = event.xdata, event.ydata
            
            #Dibujar punto
            color = colors[len(clicked_points) % len(colors)]
            ax1.plot(x, y, 'o', color=color, markersize=5)
            
            #Encontrar linea epipolar
            point = np.array([x, y, 1])
            epipolar_line = F @ point
            
            #calcular dimensiones de la imagen
            h, w = img_d.shape[:2]
            
            #Encontrar dos puntos en la linea epipolar
            if abs(epipolar_line[1]) > 1e-10:
                #Encontrar interseccion con bordes de la imagen
                x1, x2 = 0, w
                y1 = -(epipolar_line[0] * x1 + epipolar_line[2]) / epipolar_line[1]
                y2 = -(epipolar_line[0] * x2 + epipolar_line[2]) / epipolar_line[1]
                
                #Si linea no intersecta con bordes izquierdo/derecho, intentar top/bottom
                if y1 < 0 or y1 > h or y2 < 0 or y2 > h:
                    y1, y2 = 0, h
                    x1 = -(epipolar_line[1] * y1 + epipolar_line[2]) / epipolar_line[0]
                    x2 = -(epipolar_line[1] * y2 + epipolar_line[2]) / epipolar_line[0]
                
                #Dibujar linea epipolar
                ax2.plot([x1, x2], [y1, y2], '-', color=color, linewidth=2)
                
                #Guardar punto y color
                clicked_points.append((x, y, color))
            
            #Dibujar epipoles si estan dentro de los limites de la imagen
            if 0 <= e1[0] < w and 0 <= e1[1] < h:
                ax2.plot(e1[0], e1[1], 'k*', markersize=10, label='Right Epipole')
            if 0 <= e2[0] < w and 0 <= e2[1] < h:
                ax1.plot(e2[0], e2[1], 'k*', markersize=10, label='Left Epipole')

            if len(clicked_points) == 1:
                if ax1.get_legend_handles_labels()[1]: ax1.legend()
                if ax2.get_legend_handles_labels()[1]: ax2.legend()
            
            plt.draw()
    
    #on click
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()


def verify_camera_calibration(K, img_shape):
    h, w = img_shape[:2]
    
    #Comprobar distancia focal
    fx, fy = K[0,0], K[1,1]
    print("\nCamera Calibration Verification:")
    print(f"Focal lengths: fx = {fx:.2f}, fy = {fy:.2f}")
    print(f"Focal length ratio (fx/fy): {fx/fy:.3f}")
    
    #Comprobar punto principal
    cx, cy = K[0,2], K[1,2]
    print(f"Principal point: ({cx:.2f}, {cy:.2f})")
    print(f"Image center: ({w/2:.2f}, {h/2:.2f})")
    print(f"Principal point offset from center: ({cx-w/2:.2f}, {cy-h/2:.2f})")
    
    #Comprobar si el punto principal esta dentro de los limites de la imagen
    if 0 <= cx <= w and 0 <= cy <= h:
        print("Principal point is within image bounds")
    else:
        print("WARNING: Principal point is outside image bounds!")
    
    #Comprobar si la distancia focal es razonable

    if 1000 <= fx <= 3000 and 1000 <= fy <= 3000:
        print("Focal lengths are within typical range for phone cameras")
    else:
        print("WARNING: Focal lengths are outside typical range")
    
    if 0.95 <= fx/fy <= 1.05:
        print("Focal length ratio is close to 1")
    else:
        print("WARNING: Focal length ratio deviates significantly from 1")


def compute_essential_matrix(F, K):

    print(f"\nComputing Essential Matrix:")
    print(f"Fundamental Matrix F:\n{F}")
    print(f"Camera Matrix K:\n{K}")
    
    K_np = np.array(K)
    K_trans = K_np.T
    E = K_trans @ F @ K
        
    U, S, Vt = np.linalg.svd(E) 
    #Normalización usando la norma de Frobenius
    E /= np.linalg.norm(E, ord='fro')

    #Asegurar que E tiene rango 2 mediante SVD
    U, S, Vt = np.linalg.svd(E)
    print(f"SVD singular values after normalization: {S}")
    

    S_corrected = np.array([1.0, 1.0, 0.0])  #Valores ideales
    print(f"Corrected singular values: {S_corrected}")
    
    E = U @ np.diag(S_corrected) @ Vt
    
    #normalización
    E = E / np.linalg.norm(E, ord='fro')
    
    print(f"Final Essential Matrix E:\n{E}")
    U_final, S_final, Vt_final = np.linalg.svd(E)
    print(f"Final SVD singular values: {S_final}")
    print(f"Final E rank: {np.linalg.matrix_rank(E)}")
    print(f"Final E determinant: {np.linalg.det(E)}")

    return E

def interactive_essential_view(img_l, img_d, E, K):
   
    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_d_rgb = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)
    
    h, w = img_d.shape[:2]
    
    #Epipolos
    U, S, Vt = np.linalg.svd(E)
    e1 = Vt[-1, :]  # derecho
    e2 = U[:, -1]   # izquierdo
    
    print("Raw epipoles (in normalized coordinates):")
    print("Left epipole:", e2)
    print("Right epipole:", e1)
    
    #Epipolos a coordenadas no homogeneas
    if abs(e1[2]) > 1e-10:
        e1 = e1[:2] / e1[2]
    else:
        print("Warning: Right epipole is at infinity")
        e1 = e1[:2] * 1e10
        
    if abs(e2[2]) > 1e-10:
        e2 = e2[:2] / e2[2]
    else:
        print("Warning: Left epipole is at infinity")
        e2 = e2[:2] * 1e10
    
    #Epipolos a coordenadas de la imagen
    e1_img = K @ np.append(e1, 1)
    e2_img = K @ np.append(e2, 1)
    e1_img = e1_img[:2] / e1_img[2]
    e2_img = e2_img[:2] / e2_img[2]
    
    print("\nEpipoles in image coordinates:")
    print("Left epipole:", e2_img)
    print("Right epipole:", e1_img)
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    ax1.imshow(img_l_rgb)
    ax1.set_title('Left Image - Click here (Essential Matrix)')
    ax1.axis('off')
    
    ax2.imshow(img_d_rgb)
    ax2.set_title('Right Image - Epipolar Lines (Essential Matrix)')
    ax2.axis('off')
    

    clicked_points = []
    colors = plt.cm.rainbow(np.linspace(0, 1, 100)) 
    
    def onclick(event):
        if event.inaxes == ax1:

            x, y = event.xdata, event.ydata
            
            #Convertir a coordenadas normalizadas
            point = np.array([x, y, 1])
            point_norm = np.linalg.inv(K) @ point
            
            print(f"\nClicked point: ({x}, {y})")
            print("Normalized coordinates:", point_norm)
            
            #Dibujar punto en la imagen izquierda
            color = colors[len(clicked_points) % len(colors)]
            ax1.plot(x, y, 'o', color=color, markersize=5)
            
            #Calcular linea epipolar en coordenadas normalizadas
            epipolar_line = E @ point_norm
            
            print("Epipolar line in normalized coordinates:", epipolar_line)
            
            #Convertir linea epipolar a coordenadas de la imagen
            epipolar_line = np.linalg.inv(K).T @ epipolar_line
            
            #Normalizar coeficientes de la linea
            norm = np.sqrt(epipolar_line[0]**2 + epipolar_line[1]**2)
            if norm > 1e-10:
                epipolar_line = epipolar_line / norm
            
            print("Epipolar line in image coordinates:", epipolar_line)
            
            #Calcular puntos de interseccion con los bordes de la imagen
            a, b, c = epipolar_line
            
            #Encontrar intersecciones con los bordes de la imagen
            points = []
            
            #Interseccion con el borde izquierdo (x = 0)
            if abs(b) > 1e-10:
                y = -c / b
                if 0 <= y < h:
                    points.append((0, y))
            
            #Interseccion con el borde derecho (x = w)
            if abs(b) > 1e-10:
                y = -(a * w + c) / b
                if 0 <= y < h:
                    points.append((w, y))
            
            #Interseccion con el borde superior (y = 0)
            if abs(a) > 1e-10:
                x = -c / a
                if 0 <= x < w:
                    points.append((x, 0))
            
            #Interseccion con el borde inferior (y = h)
            if abs(a) > 1e-10:
                x = -(b * h + c) / a
                if 0 <= x < w:
                    points.append((x, h))
            
            
            if len(points) >= 2:
                # Ordenar puntos por coordenada x para asegurar dibujo consistente
                points.sort(key=lambda p: p[0])
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                print(f"Line endpoints: ({x1}, {y1}) to ({x2}, {y2})")
                
                #Dibujar linea epipolar
                ax2.plot([x1, x2], [y1, y2], '-', color=color, linewidth=2)
                
            
                clicked_points.append((x, y, color))
            
            # dibujar epipolos
            if 0 <= e1_img[0] < w and 0 <= e1_img[1] < h:
                ax2.plot(e1_img[0], e1_img[1], 'k*', markersize=10, label='Right Epipole')
            if 0 <= e2_img[0] < w and 0 <= e2_img[1] < h:
                ax1.plot(e2_img[0], e2_img[1], 'k*', markersize=10, label='Left Epipole')
            
            if len(clicked_points) == 1:
                if ax1.get_legend_handles_labels()[1]: ax1.legend()
                if ax2.get_legend_handles_labels()[1]: ax2.legend()

            plt.draw()   
    #on click
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

def sift_matching(img_l, img_d, ratio_thresh=0.75):
    #Creamos un detector SIFT de opencv   
    sift = cv2.SIFT_create()

    #Detectamos los puntos clave
    kp1, des1 = sift.detectAndCompute(img_l, None)
    kp2, des2 = sift.detectAndCompute(img_d, None)

    #Comprobamos si hay suficientes descriptores
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print("Warning: Could not find enough descriptors in one or both images.")
        return np.array([]), np.array([])

    #Creamos un matcher FLANN
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    
    #Hacemos matching de izquierda a derecha y aplicamos ratio test
    matches12 = flann.knnMatch(des1, des2, k=2)
    good12 = []
    for m, n in matches12:
        if m.distance < ratio_thresh * n.distance:
            good12.append(m)
            
    #Hacemos matching de derecha a izquierda y aplicamos ratio test
    matches21 = flann.knnMatch(des2, des1, k=2)
    good21 = []
    for m, n in matches21:
        if m.distance < ratio_thresh * n.distance:
            good21.append(m)
            
    #mantener solo matches que son consistentes en ambas direcciones
    symmetric_matches = []
    for m1 in good12:
        for m2 in good21:
            if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                symmetric_matches.append(m1)
                break
                
    if len(symmetric_matches) < 8:
        print(f"Warning: Only found {len(symmetric_matches)} symmetric matches. Estimation may be unstable.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in symmetric_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in symmetric_matches])
    
    return pts1, pts2


def resize_images_and_intrinsics(img_l, img_d, K, max_dimension=800):
    #Reescalamos las imagenes y la matriz de proyeccion para acelerar el calculo
    
    h, w = img_l.shape[:2]
    
    if h > w:
        scale = max_dimension / h
    else:
        scale = max_dimension / w

    new_h, new_w = int(h * scale), int(w * scale)

    img_l_resized = cv2.resize(img_l, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_d_resized = cv2.resize(img_d, (new_w, new_h), interpolation=cv2.INTER_AREA)

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale  # fx
    K_scaled[1, 1] *= scale  # fy
    K_scaled[0, 2] *= scale  # cx
    K_scaled[1, 2] *= scale  # cy
    
    
    return img_l_resized, img_d_resized, K_scaled


def uncalibrated_stereo_rectification(img_l, img_d, F, correspondences):

    h, w = img_l.shape[:2]

    #Comprobar y arreglar si la imagen esta al reves
    def _check_and_correct_flip(H, height):

        corners = np.array([
            [0, 0, 1],
            [w, 0, 1]
        ]).T
        
        transformed_corners = H @ corners
        transformed_corners = transformed_corners[:2, :] / transformed_corners[2, :]
        
        #Si la media de y de las esquinas superiores esta en la mitad inferior, la imagen esta al reves
        print("Applying vertical flip correction to homography.")
        flip_correction = np.array([[1, 0, 0], [0, -1, height], [0, 0, 1]])
        return flip_correction @ H
        return H

    def _compute_homography(epipole, width, height):
        #Mover el epipolo al infinito
        T = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]])
        e_c = T @ (epipole / epipole[2])
        
        angle = -np.arctan2(e_c[1], e_c[0])
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        e_r = R @ e_c

        f = e_r[0]
        G = np.array([[1, 0, 0], [0, 1, 0], [-1 / f if f != 0 else 0, 0, 1]])

        H = np.linalg.inv(T) @ G @ R @ T
        return H

    # Encontrar los epipolos
    _, _, Vt_f = np.linalg.svd(F)
    e_r = Vt_f[-1, :]
    _, _, Vt_ft = np.linalg.svd(F.T)
    e_l = Vt_ft[-1, :]

    #Computar las homografias iniciales
    H_r = _compute_homography(e_r, w, h)
    H_l_proj = _compute_homography(e_l, w, h)

    # Alinear la imagen izquierda a la derecha usando un modelo afín
    points_l_list = [c[0] for c in correspondences if len(c) == 2]
    points_r_list = [c[1] for c in correspondences if len(c) == 2]
    
    n_points = len(points_l_list)
    points_l_h = np.hstack([np.array(points_l_list), np.ones((n_points, 1))])
    points_r_h = np.hstack([np.array(points_r_list), np.ones((n_points, 1))])

    p_l_proj_h = (H_l_proj @ points_l_h.T).T
    p_r_rect_h = (H_r @ points_r_h.T).T
    
    p_l_proj = p_l_proj_h[:, :2] / p_l_proj_h[:, 2, np.newaxis]
    p_r_rect = p_r_rect_h[:, :2] / p_r_rect_h[:, 2, np.newaxis]

    #Resolver para un modelo afín simple (escala + traducción, no cizalla)
    # y_r ≈ ay * y_l + by
    y_l, y_r = p_l_proj[:, 1], p_r_rect[:, 1]
    M_y = np.vstack([y_l, np.ones(n_points)]).T
    (ay, by), _, _, _ = np.linalg.lstsq(M_y, y_r, rcond=None)

    # x_r ≈ ax * x_l + bx
    x_l, x_r = p_l_proj[:, 0], p_r_rect[:, 0]
    M_x = np.vstack([x_l, np.ones(n_points)]).T
    (ax, bx), _, _, _ = np.linalg.lstsq(M_x, x_r, rcond=None)

    Alignment = np.array([[ax, 0, bx], [0, ay, by], [0, 0, 1]])

    # Homografia final izquierda
    H_l = Alignment @ H_l_proj
    
    #Comprobar si hay volteos y corregirlos
    H_l = _check_and_correct_flip(H_l, h)
    H_r = _check_and_correct_flip(H_r, h)

    #Aplicar transformaciones a las imagenes
    img_l_rect = cv2.warpPerspective(img_l, H_l, (w, h))
    img_d_rect = cv2.warpPerspective(img_d, H_r, (w, h))

    print("Uncalibrated stereo rectification completed using custom Hartley's algorithm.")
    return H_l, H_r, img_l_rect, img_d_rect

def visualize_rectified_images(img_l_rect, img_d_rect):
    h, w = img_l_rect.shape[:2]
    combined = np.hstack((img_l_rect, img_d_rect))

    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(combined_rgb)

    for y in range(0, h, 50):
        plt.axhline(y=y, color='r', linestyle='--', alpha=0.5)
    
    plt.title('Rectified Stereo Pair with Epipolar Lines (Uncalibrated)')
    plt.axis('off')
    plt.show()

def mapa_disparidad(img_l_rect, img_d_rect, block_size=15, max_disparity=64):
  
    print(f"\nComputing disparity with simple Block Matching (Block: {block_size}, Max Disp: {max_disparity})...")

    left_gray = cv2.cvtColor(img_l_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right_gray = cv2.cvtColor(img_d_rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = left_gray.shape

    # Construir el volumen de costo (Suma de Diferencias Absolutas)
    print("  Construyendo volumen de costo (SAD)...")
    cost_volume = np.zeros((h, w, max_disparity), dtype=np.float32)
    
    for d in range(max_disparity):
        diff = np.zeros_like(left_gray)
        if d > 0:
            diff[:, d:] = np.abs(left_gray[:, d:] - right_gray[:, :-d])
        else:
            diff[:, :] = np.abs(left_gray - right_gray)
        
        # Sumar diferencias sobre una ventana usando un filtro de caja
        sad = cv2.boxFilter(diff, -1, (block_size, block_size), normalize=False)
        cost_volume[:, :, d] = sad

    # Seleccionar la mejor disparidad
    print("  Encontrando la mejor disparidad del volumen de costo...")
    int_disparity_map = np.argmin(cost_volume, axis=2).astype(np.float32)



    # Refinamiento sub-pixel
    print("  Aplicando refinamiento sub-pixel...")
    d_int = int_disparity_map.astype(int)
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # Asegurar que los indices esten dentro de los limites
    d_int_clipped = np.clip(d_int, 0, max_disparity - 1)
    
    C_min = cost_volume[y_coords, x_coords, d_int_clipped]
    
    d_prev = np.maximum(0, d_int_clipped - 1)
    C_prev = cost_volume[y_coords, x_coords, d_prev]

    d_next = np.minimum(max_disparity - 1, d_int_clipped + 1)
    C_next = cost_volume[y_coords, x_coords, d_next]
    
    numerator = C_prev - C_next
    denominator = 2 * (C_prev - 2*C_min + C_next)
    
    delta = np.zeros_like(numerator, dtype=np.float32)
    valid = np.abs(denominator) > 1e-6
    delta[valid] = numerator[valid] / denominator[valid]
    
    disparity_map_final = d_int.astype(np.float32) + np.clip(delta, -0.5, 0.5)
    
    # Invalida disparidades donde el refinamiento no es posible
    mask = (d_int_clipped == 0) | (d_int_clipped == max_disparity - 1)
    disparity_map_final[mask] = d_int_clipped[mask]
    
    return disparity_map_final

def visualize_disparity_map(img_l_rect, disparity_map):

    disparity_map[np.isinf(disparity_map)] = 0
    disparity_map[np.isnan(disparity_map)] = 0
    
    disp_vis = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    img_l_rgb = cv2.cvtColor(img_l_rect, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_l_rgb)
    plt.title('Rectified Left Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(disp_colored, cv2.COLOR_BGR2RGB))
    plt.title('Disparity Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()    

def verify_krt_decomposition(P, K, R, t):

    #Reconstruir la matriz de proyeccion
    P_reconstructed = K @ np.hstack((R, t.reshape(-1, 1)))
    
    #Calcular el error de reconstruccion
    reconstruction_error = np.linalg.norm(P - P_reconstructed, ord='fro')
    
    print(f"\nKRT Decomposition Verification:")
    print(f"Original P:\n{P}")
    print(f"Reconstructed P:\n{P_reconstructed}")
    print(f"Reconstruction error (Frobenius norm): {reconstruction_error}")
    
    #Comprobar si K es triangular superior
    K_lower = K - np.triu(K)
    is_upper_triangular = np.allclose(K_lower, 0, atol=1e-8)
    print(f"K is upper triangular: {is_upper_triangular}")
    
    #Comprobar si R es ortogonal
    R_orthogonal_error = np.linalg.norm(R @ R.T - np.eye(3), ord='fro')
    is_orthogonal = R_orthogonal_error < 1e-8
    print(f"R is orthogonal: {is_orthogonal} (error: {R_orthogonal_error})")
    
    #Comprobar si R tiene determinante 1 (rotacion propia)
    det_R = np.linalg.det(R)
    is_proper_rotation = np.abs(det_R - 1.0) < 1e-8
    print(f"R is proper rotation (det=1): {is_proper_rotation} (det: {det_R})")
    
    return reconstruction_error

def compare_epipolar_lines(point, F, E, K):

    print(f"\nComparing epipolar lines for point {point}:")
    
    #Usar la matriz fundamental directamente
    point_h = np.array([point[0], point[1], 1])
    epipolar_F = F @ point_h
    print(f"Epipolar line using F: {epipolar_F}")
    
    #Usar la matriz esencial
    point_norm = np.linalg.inv(K) @ point_h
    print(f"Punto en coordenadas normalizadas: {point_norm}")
    
    #Calcular linea epipolar en coordenadas normalizadas
    epipolar_E_norm = E @ point_norm
    print(f"Epipolar line in normalized coordinates (E): {epipolar_E_norm}")
    
    #Convertir de nuevo a coordenadas de la imagen
    epipolar_E = np.linalg.inv(K).T @ epipolar_E_norm
    print(f"Epipolar line using E (converted to image coords): {epipolar_E}")
    
    #Comparar los resultados
    diff = np.linalg.norm(epipolar_F - epipolar_E)
    print(f"Difference between F and E methods: {diff}")
    
    #Normalizar ambos
    epipolar_F_norm = epipolar_F / np.linalg.norm(epipolar_F[:2])
    epipolar_E_norm = epipolar_E / np.linalg.norm(epipolar_E[:2])
    
    diff_normalized = np.linalg.norm(epipolar_F_norm - epipolar_E_norm)
    print(f"Difference after normalization: {diff_normalized}")
    
    return epipolar_F, epipolar_E

def verify_rectification(correspondences, H_l, H_r):
    print("\nVerifying rectification quality:")
    
    if len(correspondences) == 0:
        print("No correspondences available for verification")
        return
    
    #Convertir correspondencias a numpy array
    points_l = []
    points_r = []
    
    for corr in correspondences:
        if len(corr) == 2:
            points_l.append(corr[0])
            points_r.append(corr[1])
    
    if len(points_l) == 0:
        print("No valid correspondences found")
        return
        
    points_l = np.array(points_l)
    points_r = np.array(points_r)
    
    #Convertir a coordenadas homogeneas
    points_l_h = np.hstack([points_l, np.ones((len(points_l), 1))])
    points_r_h = np.hstack([points_r, np.ones((len(points_r), 1))])
    
    #Aplicar transformaciones de rectificacion
    points_l_rect = (H_l @ points_l_h.T).T
    points_r_rect = (H_r @ points_r_h.T).T
    
    #Convertir de nuevo a coordenadas no homogeneas
    points_l_rect = points_l_rect[:, :2] / points_l_rect[:, 2:]
    points_r_rect = points_r_rect[:, :2] / points_r_rect[:, 2:]
    
    #Calcular diferencias en coordenadas y
    y_diffs = np.abs(points_l_rect[:, 1] - points_r_rect[:, 1])
    
    print(f"Y-coordinate differences after rectification:")
    print(f"Mean: {np.mean(y_diffs):.2f} pixels")
    print(f"Median: {np.median(y_diffs):.2f} pixels")
    print(f"Max: {np.max(y_diffs):.2f} pixels")
    print(f"Min: {np.min(y_diffs):.2f} pixels")
    
    # Contar puntos que estan bien alineados
    well_aligned = np.sum(y_diffs < 2.0)
    print(f"Points aligned within 2 pixels: {well_aligned}/{len(y_diffs)} ({100*well_aligned/len(y_diffs):.1f}%)")
    
    return y_diffs

def visualize_rectified_correspondences(img_l_rect, img_d_rect, correspondences, H_l, H_r):

    if len(correspondences) == 0:
        print("No correspondences to visualize")
        return
    
    #Convertir correspondencias a numpy array
    points_l = []
    points_r = []
    
    for corr in correspondences[:min(20, len(correspondences))]:  #enseñar 20 solo
        if len(corr) == 2:
            points_l.append(corr[0])
            points_r.append(corr[1])
    
    if len(points_l) == 0:
        print("No valid correspondences found")
        return
        
    points_l = np.array(points_l)
    points_r = np.array(points_r)
    
    #Convertir a coordenadas homogeneas
    points_l_h = np.hstack([points_l, np.ones((len(points_l), 1))])
    points_r_h = np.hstack([points_r, np.ones((len(points_r), 1))])
    
    #Aplicar transformaciones de rectificacion
    points_l_rect = (H_l @ points_l_h.T).T
    points_r_rect = (H_r @ points_r_h.T).T
    
    #Convertir de nuevo a coordenadas no homogeneas
    points_l_rect = points_l_rect[:, :2] / points_l_rect[:, 2:]
    points_r_rect = points_r_rect[:, :2] / points_r_rect[:, 2:]
    
    #Crear imagen combinada
    h, w = img_l_rect.shape[:2]
    combined = np.hstack((img_l_rect, img_d_rect))
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 7))
    plt.imshow(combined_rgb)
    
    #Dibujar correspondencias
    colors = plt.cm.rainbow(np.linspace(0, 1, len(points_l_rect)))
    
    for i, (pl, pr, color) in enumerate(zip(points_l_rect, points_r_rect, colors)):
        #Dibujar puntos
        plt.plot(pl[0], pl[1], 'o', color=color, markersize=8)
        plt.plot(pr[0] + w, pr[1], 'o', color=color, markersize=8)
        
        #Dibujar linea horizontal para mostrar la restriccion epipolar
        plt.plot([0, 2*w], [pl[1], pr[1]], '--', color=color, alpha=0.7, linewidth=1)
        
        #Dibujar linea conectando
        plt.plot([pl[0], pr[0] + w], [pl[1], pr[1]], '-', color=color, linewidth=2)
    
    #Dibujar lineas horizontales de la cuadricula
    for y in range(0, h, 50):
        plt.axhline(y=y, color='red', linestyle=':', alpha=0.3)
    
    plt.title('Rectified Correspondences')
    plt.axis('off')
    plt.show()

def recover_pose_from_essential(E, correspondences, K):
    
    #Extraer puntos de correspondencias
    points_l_list = [c[0] for c in correspondences if len(c) == 2]
    points_r_list = [c[1] for c in correspondences if len(c) == 2]

    if len(points_l_list) < 8:
        print("Error: Not enough correspondences for pose recovery.")
        return np.eye(3), np.zeros((3,1))
        
    points_l = np.array(points_l_list, dtype=np.float32)
    points_r = np.array(points_r_list, dtype=np.float32)

    #cv2.recoverPose para encontrar la correcta R y t
    points_in_front, R, t, mask = cv2.recoverPose(E, points_l, points_r, K)

    print(f"\nPose recovery found {points_in_front} points in front of both cameras.")
    print("Recovered Rotation (R):")
    print(R)
    print("Recovered Translation (t):")
    print(t)
    
    return R, t

def create_point_cloud(disparity_map, img_l_rect, K, t_stereo, H_l, H_r):
    print("\nGenerating 3D point cloud data...")
    
    baseline = np.linalg.norm(t_stereo)
    if baseline == 0:
        print("Error: Baseline is zero. Cannot create point cloud.")
        return np.array([]), np.array([])

    #Obtener parametros intrinsecos originales
    focal_length = K[0,0]
    cx_orig = K[0,2]
    cy_orig = K[1,2]

    #Calcular los nuevos puntos principales despues de aplicar la rectificacion
    p_orig_h = np.array([cx_orig, cy_orig, 1.0])
    
    p_l_rect_h = H_l @ p_orig_h
    cx_l_new = p_l_rect_h[0] / p_l_rect_h[2]
    cy_l_new = p_l_rect_h[1] / p_l_rect_h[2]

    p_r_rect_h = H_r @ p_orig_h
    cx_r_new = p_r_rect_h[0] / p_r_rect_h[2]
    
    #Construir la matriz Q con los puntos principales corregidos
    Q = np.float32([[1, 0, 0, -cx_l_new],
                    [0, 1, 0, -cy_l_new],
                    [0, 0, 0, focal_length],
                    [0, 0, 1.0/baseline, (cx_l_new - cx_r_new)/baseline]])

    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    colors_rgb = cv2.cvtColor(img_l_rect, cv2.COLOR_BGR2RGB)

    #Mascara de puntos invalidos (disparidad no positiva o profundidad infinita)
    mask = (disparity_map > 0) & np.isfinite(points_3d).all(axis=2)
    
    points = points_3d[mask]
    colors = colors_rgb[mask]
    
    if points.shape[0] == 0:
        print("Point cloud data generated with 0 valid points after initial filtering.")
        return np.array([]), np.array([])

    #Filtrado de puntos extremos usando el percentil 99
    depths = points[:, 2]
    max_depth = np.percentile(depths, 99.0) 
    
    percentile_mask = depths < max_depth
    points = points[percentile_mask]
    colors = colors[percentile_mask]

    #Girar los puntos para una orientacion intuitiva
    transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    points = points @ transform.T

    print(f"Point cloud data generated with {len(points)} points.")
    return points, colors

def visualize_point_cloud(points, colors):

    if points.shape[0] == 0:
        print("Point cloud is empty. Nothing to visualize.")
        return
        

    print("\nDisplaying 3D point cloud with Open3D.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    print("Rotate the view with the mouse. Press 'q' to close the window.")
    o3d.visualization.draw_geometries([pcd])

    # Matplotlib's scatter expects colors to be in the [0, 1] range
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255.0, s=1)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Point Cloud Visualization')
    
    # Set aspect ratio to be equal
    ax.set_aspect('auto') # 'equal' is not supported in 3D plots

    print("\nDisplaying 3D point cloud with Matplotlib. This may be slow and less interactive than Open3D.")
    print("Close the plot window to continue.")
    plt.show()

def main():
    
    #Cargar imagenes
    img_l_orig = cv2.imread('L5.jpeg', cv2.IMREAD_COLOR)
    img_d_orig = cv2.imread('R5.jpeg', cv2.IMREAD_COLOR)
    #cargar matriz de calibracion
    P = np.load('projection_matrix_portrait.npy')

    # Obtener la matriz de calibracion de la camara K desde la descomposicion KRT
    K_orig, R, t = krt_descomposition(P)

    # Reescalar imagenes y escalar K
    img_l, img_d, K = resize_images_and_intrinsics(img_l_orig, img_d_orig, K_orig)
    
    # Verificar la calibracion de la camara
    verify_camera_calibration(K, img_l.shape)
    
    # Verificar la descomposicion KRT (en la original, no la escalada)
    verify_krt_decomposition(P, K_orig, R, t)
    
    

    print("Finding SIFT matches...")
    puntos_clave_l, puntos_clave_d = sift_matching(img_l, img_d)
    print(f"Found {len(puntos_clave_l)} matches")
    
    # Iteraciones y threshold del ransac
    r = 10000  
    t = 1.0 
    print("Running RANSAC...")
    F, puntos = ransac(puntos_clave_l, puntos_clave_d, r, t)
    
    
    # Calcular la matriz esencial
    E = compute_essential_matrix(F, K)
   
    # Verificar la matriz fundamental y calcular los errores epipolares normalizados
    puntos_l_list, puntos_d_list = zip(*puntos)
    puntos_l = np.vstack(puntos_l_list)
    puntos_d = np.vstack(puntos_d_list)
    epipolar_errors = verify_fundamental_matrix(F, puntos_l, puntos_d)

    print("\nFundamental Matrix:")
    print(F)
    print("\nRank of F:", np.linalg.matrix_rank(F))
    print("Determinant of F:", np.linalg.det(F))
    
    print("\nEssential Matrix (normalized and constrained):")
    print(E)
    print("\nRank of E:", np.linalg.matrix_rank(E))
    print("Determinant of E:", np.linalg.det(E))
    
    #Comparacion de lineas epipolares
    test_point = [500, 400]  
    compare_epipolar_lines(test_point, F, E, K)
    
    #Visualizacion interactiva de lineas epipolares
    print("\nStarting interactive epipolar visualization (Fundamental Matrix)...")
    print("Click on the left image to see epipolar lines on the right image")
    interactive_epipolar_view(img_l, img_d, F)
    
    # Visualizacion interactiva de lineas epipolares
    print("\nStarting interactive epipolar visualization (Essential Matrix)...")
    print("Click on the left image to see epipolar lines on the right image")
    interactive_essential_view(img_l, img_d, E, K)
    
    # Recuperar la pose de la matriz esencial
    R_stereo, t_stereo = recover_pose_from_essential(E, puntos, K)
  
    print("\nPerforming Uncalibrated Stereo Rectification...")
    H_l, H_r, img_l_rect, img_d_rect = uncalibrated_stereo_rectification(img_l, img_d, F, puntos)

    # Visualizar imagenes rectificadas
    print("\nVisualizando par estereo rectificado...")
    visualize_rectified_images(img_l_rect, img_d_rect)
    

    print("\nComputing disparity map with custom Block Matching...")
    #parametros block matching
    block_size = 5
    max_disparity = 32
    disparity_map = mapa_disparidad(img_l_rect, img_d_rect, block_size, max_disparity)
    
    print("\nVisualizing disparity map...")
    visualize_disparity_map(img_l_rect, disparity_map)
    
    print("\nRectification homographies:")
    print("H_l:") 
    print(H_l)
    print("H_r:")
    print(H_r)

    # Verificar la calidad de la rectificacion
    verify_rectification(puntos, H_l, H_r)

    # Visualizar correspondencias rectificadas
    visualize_rectified_correspondences(img_l_rect, img_d_rect, puntos, H_l, H_r)

    # Crear nube de puntos
    points, colors = create_point_cloud(disparity_map, img_l_rect, K, t_stereo, H_l, H_r)

    # Visualizar nube de puntos
    visualize_point_cloud(points, colors)

if __name__ == '__main__':
    main()