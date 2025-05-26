import pytest
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import glob
from unit_tests_V2 import (
    krt_descomposition,
    reconstruir_P,
    normalize_points,
    sampson_error,
    ransac,
    calcular_matriz_E,
    robust_numpy_matching,
    block_matching,
    compute_disparity_map,
    filter_horizontal_matches,
    mean_epipolar_error,
    plot_sift_matches,
    plot_inlier_matches,
    draw_epipolar_lines,
    calibracion_camara_chessboard
)

# ------------------------------------------------------------------------
# Fixtures for synthetic data
# ------------------------------------------------------------------------

@pytest.fixture
def synthetic_projection_matrix():
    """Create a synthetic camera projection matrix."""
    return np.array([
        [1500, 0, 500, 0],
        [0, 1500, 500, 0],
        [0, 0, 1, 0]
    ])

@pytest.fixture
def synthetic_camera_params():
    """Create synthetic K, R, t camera parameters."""
    # Intrinsic parameters
    K = np.array([
        [1500, 0, 500],
        [0, 1500, 500],
        [0, 0, 1]
    ])
    
    # Rotation matrix (small rotation around y-axis)
    theta = np.radians(5)
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    # Translation vector
    t = np.array([[10], [5], [0]])
    
    return K, R, t

@pytest.fixture
def synthetic_stereo_pair():
    """Create synthetic stereo images with known correspondences."""
    # Create two blank images (640x480)
    height, width = 480, 640
    img_l = np.zeros((height, width, 3), dtype=np.uint8)
    img_d = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create synthetic points with known correspondences
    np.random.seed(42)  # For reproducibility
    n_points = 100
    left_points = []
    right_points = []
    
    # Create synthetic 3D points
    for i in range(n_points):
        # Generate random coordinates
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        size = np.random.randint(5, 15)
        color = tuple(np.random.randint(100, 255, 3).tolist())
        
        # Draw a circle in left image
        cv2.circle(img_l, (x, y), size, color, -1)
        left_points.append((x, y))
        
        # Add horizontal disparity for right image (simulating depth)
        disparity = np.random.randint(15, 35)
        x_right = max(0, x - disparity)
        
        # Draw the same circle in right image
        cv2.circle(img_d, (x_right, y), size, color, -1)
        right_points.append((x_right, y))
    
    # Convert to numpy arrays
    left_points = np.array(left_points, dtype=np.float32)
    right_points = np.array(right_points, dtype=np.float32)
    
    return img_l, img_d, left_points, right_points

@pytest.fixture
def synthetic_fundamental_matrix():
    """Create a synthetic fundamental matrix."""
    # Create a simple fundamental matrix (rank 2)
    F = np.array([
        [0, -0.0001, 0.01],
        [0.0001, 0, -0.01],
        [-0.01, 0.01, 0]
    ])
    
    # Ensure rank 2 by SVD
    u, s, vh = np.linalg.svd(F)
    s[2] = 0
    F_rank2 = u @ np.diag(s) @ vh
    
    # Normalize
    F_rank2 = F_rank2 / F_rank2[2, 2]
    
    return F_rank2

# ------------------------------------------------------------------------
# Tests for projection matrix decomposition and reconstruction
# ------------------------------------------------------------------------

def test_krt_decomposition(synthetic_projection_matrix):
    """Test KRT decomposition of a projection matrix."""
    P = synthetic_projection_matrix
    
    # Decompose the projection matrix
    K, R, t = krt_descomposition(P)
    
    # Check that K is upper triangular
    assert np.allclose(np.tril(K, -1), 0, atol=1e-8)
    
    # Check that K[2,2] is approximately 1
    assert np.isclose(K[2,2], 1.0)
    
    # Check that R is a valid rotation matrix
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
    
    # Test reconstruction
    P_reconstructed = reconstruir_P(K, R, t)
    assert np.allclose(P, P_reconstructed, atol=1e-8)

# ------------------------------------------------------------------------
# Tests for point normalization
# ------------------------------------------------------------------------

def test_normalize_points():
    """Test point normalization function."""
    # Create test points
    points = np.array([
        [100, 200],
        [150, 250],
        [200, 300],
        [250, 350]
    ])
    
    # Normalize points
    points_normalized, T = normalize_points(points)
    
    # Check that centroid is close to origin
    centroid = np.mean(points_normalized, axis=0)
    assert np.allclose(centroid, [0, 0], atol=1e-8)
    
    # Check that average distance from origin is close to sqrt(2)
    distances = np.sqrt(np.sum(points_normalized**2, axis=1))
    assert np.isclose(np.mean(distances), np.sqrt(2), atol=0.1)
    
    # Check transformation matrix
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (T @ points_homogeneous.T).T
    assert np.allclose(transformed_points[:, :2], points_normalized, atol=1e-8)

# ------------------------------------------------------------------------
# Tests for Sampson error
# ------------------------------------------------------------------------

def test_sampson_error(synthetic_fundamental_matrix):
    """Test Sampson error calculation."""
    F = synthetic_fundamental_matrix
    
    # Create test points (Nx3 format)
    x1 = np.array([[100, 200, 1], [150, 250, 1]], dtype=np.float32)
    x2 = np.array([[150, 200, 1], [200, 250, 1]], dtype=np.float32)
    
    # Calculate error
    error = sampson_error(F, x1, x2)
    
    # Error should be an array with one value per point
    assert isinstance(error, np.ndarray)
    assert error.shape == (2,)
    assert np.all(error >= 0)

# ------------------------------------------------------------------------
# Tests for RANSAC
# ------------------------------------------------------------------------

def test_ransac_synthetic(synthetic_stereo_pair):
    """Test RANSAC with synthetic data."""
    _, _, left_points, right_points = synthetic_stereo_pair
    
    # Run RANSAC
    F, inliers = ransac(left_points, right_points, 100, 5.0)
    
    # Check fundamental matrix properties
    assert F.shape == (3, 3)
    
    # Check rank 2 property
    u, s, vh = np.linalg.svd(F)
    assert s[2] / s[0] < 0.1  # Third singular value should be small
    
    # Check that we found inliers
    assert len(inliers) > 0
    assert len(inliers) <= len(left_points)

@pytest.fixture
def real_stereo_pair():
    left = cv2.imread('cones/im_i.jpg', cv2.IMREAD_COLOR)
    right = cv2.imread('cones/im_d.jpg', cv2.IMREAD_COLOR)
    if left is None or right is None:
        pytest.skip("Stereo images not found!")
    return left, right

def test_sift_and_ransac(real_stereo_pair):
    img_l, img_d = real_stereo_pair
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_l, None)
    kp2, des2 = sift.detectAndCompute(img_d, None)
    idx_l, idx_d = robust_numpy_matching(des1, des2)
    puntos_clave_l = np.array([kp1[i].pt for i in idx_l], dtype=np.float32)
    puntos_clave_d = np.array([kp2[i].pt for i in idx_d], dtype=np.float32)
    puntos_clave_l, puntos_clave_d = filter_horizontal_matches(puntos_clave_l, puntos_clave_d)
    if len(puntos_clave_l) < 8:
        pytest.skip("Not enough matches for RANSAC")
    F, inliers = ransac(puntos_clave_l, puntos_clave_d, 100, 5.0)
    assert F.shape == (3, 3)
    assert len(inliers) > 0

# ------------------------------------------------------------------------
# Tests for essential matrix
# ------------------------------------------------------------------------

def test_essential_matrix(synthetic_fundamental_matrix):
    """Test essential matrix calculation."""
    F = synthetic_fundamental_matrix
    K = np.array([
        [1000, 0, 500],
        [0, 1000, 500],
        [0, 0, 1]
    ])
    
    E = calcular_matriz_E(F, K)
    
    # Check essential matrix properties
    assert E.shape == (3, 3)
    
    # Check singular values
    u, s, vh = np.linalg.svd(E)
    s = s / s[0]  # Normalize
    assert abs(s[0] - s[1]) < 0.2  # First two should be similar
    assert s[2] < 0.1  # Third should be small

# ------------------------------------------------------------------------
# Tests for robust matching
# ------------------------------------------------------------------------

def test_robust_numpy_matching():
    """Test robust numpy matching function."""
    # Create synthetic descriptors
    np.random.seed(42)
    des1 = np.random.rand(100, 128)
    des2 = np.random.rand(100, 128)
    
    # Add some matching pairs
    for i in range(50):
        des2[i] = des1[i] + np.random.normal(0, 0.1, 128)
    
    # Run matching
    idx_l, idx_d = robust_numpy_matching(des1, des2)
    
    # Check results
    assert len(idx_l) > 0
    assert len(idx_l) == len(idx_d)
    assert len(idx_l) <= min(len(des1), len(des2))

# ------------------------------------------------------------------------
# Tests for block matching
# ------------------------------------------------------------------------

def test_block_matching():
    """Test block matching function."""
    # Create synthetic stereo pair
    height, width = 100, 100
    left = np.zeros((height, width), dtype=np.float32)
    right = np.zeros((height, width), dtype=np.float32)
    
    # Add some features
    for i in range(10, 90, 20):
        for j in range(10, 90, 20):
            left[i:i+10, j:j+10] = 255
            right[i:i+10, j-5:j+5] = 255  # Shifted by 5 pixels
    
    # Run block matching
    disparity = block_matching(left, right, max_disparity=20, kernel_size=5)
    
    # Check results
    assert disparity.shape == (height, width)
    assert np.all(disparity >= 0)
    assert np.all(disparity <= 20)

# ------------------------------------------------------------------------
# Tests for horizontal match filtering
# ------------------------------------------------------------------------

def test_filter_horizontal_matches():
    """Test horizontal match filtering."""
    # Create test points
    points_l = np.array([
        [100, 200],
        [150, 250],
        [200, 300]
    ])
    
    points_d = np.array([
        [120, 200],  # Horizontal
        [160, 260],  # Diagonal
        [180, 400]   # Vertical
    ])
    
    # Filter matches
    filtered_l, filtered_d = filter_horizontal_matches(points_l, points_d, max_angle_deg=20)
    
    # Check that only horizontal matches remain
    assert len(filtered_l) == 1
    assert len(filtered_d) == 1
    assert np.allclose(filtered_l[0], points_l[0])
    assert np.allclose(filtered_d[0], points_d[0])

# ------------------------------------------------------------------------
# Tests for mean epipolar error
# ------------------------------------------------------------------------

def test_mean_epipolar_error(synthetic_fundamental_matrix):
    """Test mean epipolar error calculation."""
    F = synthetic_fundamental_matrix
    
    # Create test points
    pts1 = np.array([
        [100, 200],
        [150, 250],
        [200, 300]
    ])
    
    pts2 = np.array([
        [120, 200],
        [160, 250],
        [220, 300]
    ])
    
    # Calculate error
    error = mean_epipolar_error(F, pts1, pts2)
    
    # Check result
    assert isinstance(error, float)
    assert error >= 0

# ------------------------------------------------------------------------
# Tests for camera calibration
# ------------------------------------------------------------------------

@pytest.fixture
def synthetic_chessboard_images():
    """Create synthetic chessboard images for calibration testing."""
    # Create a 7x7 chessboard pattern
    pattern_size = (7, 7)
    square_size = 2.4  # mm
    image_size = (640, 480)
    
    # Create object points (3D points in real world space)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Create synthetic camera parameters
    K = np.array([
        [1000, 0, image_size[0]/2],
        [0, 1000, image_size[1]/2],
        [0, 0, 1]
    ])
    
    # Create multiple views with different poses
    images = []
    objpoints = []
    imgpoints = []
    
    for i in range(5):
        # Create rotation and translation (ensure correct shape)
        rvec = np.array([0.1, 0.2, 0.3], dtype=np.float32).reshape(3, 1)
        tvec = np.array([0, 0, 1000 + i*100], dtype=np.float32).reshape(3, 1)
        
        # Project points
        imgp, _ = cv2.projectPoints(objp, rvec, tvec, K, None)
        imgp = imgp.reshape(-1, 2)
        
        # Create synthetic image
        img = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        for pt in imgp:
            cv2.circle(img, tuple(map(int, pt)), 3, 255, -1)
        
        images.append(img)
        objpoints.append(objp)
        imgpoints.append(imgp)
    
    return images, objpoints, imgpoints

def test_camera_calibration(synthetic_chessboard_images):
    """Test camera calibration function."""
    images, objpoints, imgpoints = synthetic_chessboard_images
    
    # Save images temporarily
    temp_dir = "temp_calib"
    os.makedirs(temp_dir, exist_ok=True)
    image_files = []
    for i, img in enumerate(images):
        filename = f"{temp_dir}/chessboard_{i}.jpg"
        cv2.imwrite(filename, img)
        image_files.append(filename)
    
    try:
        # Run calibration
        ret, mtx, dist, rvecs, tvecs = calibracion_camara_chessboard(
            image_files,
            chessboard_size=(7,7),
            square_size=2.4,
            show_corners=False
        )
        
        # Check results
        assert isinstance(ret, float)
        assert mtx.shape == (3, 3)
        assert dist.shape == (5,)
        assert len(rvecs) == len(images)
        assert len(tvecs) == len(images)
        
        # Check that camera matrix is reasonable
        assert mtx[0,0] > 0  # focal length x
        assert mtx[1,1] > 0  # focal length y
        assert mtx[2,2] == 1  # homogeneous coordinate
        
    finally:
        # Clean up temporary files
        for filename in image_files:
            os.remove(filename)
        os.rmdir(temp_dir)

def test_camera_calibration_real():
    """Test camera calibration with real chessboard images."""
    # Check if calibration images exist
    image_files = glob.glob('Muestra/*.jpeg')
    if not image_files:
        pytest.skip("No calibration images found in Muestra directory")
    
    # Run calibration
    ret, mtx, dist, rvecs, tvecs = calibracion_camara_chessboard(
        image_files,
        chessboard_size=(7,7),
        square_size=2.4,
        show_corners=False
    )
    
    # Check results
    assert isinstance(ret, float)
    assert mtx.shape == (3, 3)
    assert dist.size >= 5
    assert len(rvecs) == len(image_files)
    assert len(tvecs) == len(image_files)
    
    # Check that camera matrix is reasonable
    assert mtx[0,0] > 0  # focal length x
    assert mtx[1,1] > 0  # focal length y
    assert mtx[2,2] == 1  # homogeneous coordinate

# ------------------------------------------------------------------------
# Tests for visualization functions
# ------------------------------------------------------------------------

def test_plot_sift_matches(real_stereo_pair):
    """Test SIFT matches visualization."""
    img_l, img_d = real_stereo_pair
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_l, None)
    kp2, des2 = sift.detectAndCompute(img_d, None)
    idx_l, idx_d = robust_numpy_matching(des1, des2)
    puntos_clave_l = np.array([kp1[i].pt for i in idx_l], dtype=np.float32)
    puntos_clave_d = np.array([kp2[i].pt for i in idx_d], dtype=np.float32)
    puntos_clave_l, puntos_clave_d = filter_horizontal_matches(puntos_clave_l, puntos_clave_d)
    if len(puntos_clave_l) < 8:
        pytest.skip("Not enough matches for SIFT visualization")
    plot_sift_matches(img_l, img_d, puntos_clave_l, puntos_clave_d)

def test_plot_inlier_matches():
    """Test inlier matches visualization."""
    # Create synthetic images
    img_l = np.zeros((100, 100, 3), dtype=np.uint8)
    img_d = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Create inlier matches
    inliers = [((30, 30), (40, 30))]
    
    # Test visualization (should not raise any errors)
    plot_inlier_matches(img_l, img_d, inliers)

def test_draw_epipolar_lines(synthetic_fundamental_matrix):
    """Test epipolar line drawing."""
    # Create synthetic images
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Create test points
    pts1 = np.array([[30, 30], [50, 50]], dtype=np.float32)
    pts2 = np.array([[40, 30], [60, 50]], dtype=np.float32)
    
    # Test visualization (should not raise any errors)
    draw_epipolar_lines(img1, img2, synthetic_fundamental_matrix, pts1, pts2)

# ------------------------------------------------------------------------
# Tests for disparity computation
# ------------------------------------------------------------------------

def test_compute_disparity_map():
    """Test disparity map computation."""
    # Create synthetic stereo pair
    height, width = 100, 100
    left = np.zeros((height, width, 3), dtype=np.uint8)
    right = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some features
    for i in range(10, 90, 20):
        for j in range(10, 90, 20):
            cv2.circle(left, (j, i), 5, (255, 255, 255), -1)
            cv2.circle(right, (j-5, i), 5, (255, 255, 255), -1)
    
    # Compute disparity map
    disparity_map = compute_disparity_map(
        left,
        right,
        max_disparity=20,
        kernel_size=5,
        use_subpixel=True
    )
    
    # Check results
    assert disparity_map.shape == (height, width)
    assert np.all(disparity_map >= 0)
    assert np.all(disparity_map <= 20)
    
    # Check that we found the expected disparity
    # The circles are shifted by 5 pixels
    expected_disparity = 5
    mask = (left[:, :, 0] > 0) & (right[:, :, 0] > 0)
    if np.any(mask):
        mean_disparity = np.mean(disparity_map[mask])
        assert abs(mean_disparity - expected_disparity) < 2.0

def test_disparity_map_with_ground_truth(real_stereo_pair):
    left, right = real_stereo_pair
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    disparity_map = compute_disparity_map(left_gray, right_gray, max_disparity=20, kernel_size=5, use_subpixel=True)
    assert disparity_map.shape == left_gray.shape
    assert disparity_map.dtype == np.float32

if __name__ == "__main__":
    pytest.main(["-v", __file__])