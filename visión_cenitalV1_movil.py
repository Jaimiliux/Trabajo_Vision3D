#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import json
import threading
import time
import argparse

# ROS imports (optional - will work without ROS)
try:
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Bool
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    print("ROS not available - running in standalone mode")
    ROS_AVAILABLE = False

class Navegacion_Robot:
    def __init__(self, camera_source="mobile", mobile_ip=None, use_ros=False):
        """
        Initialize the robot navigation system
        
        Args:
            camera_source: "mobile", "usb", "file", or "ros"
            mobile_ip: IP address for mobile camera (e.g., "192.168.71.56:8080")
            use_ros: Whether to use ROS functionality
        """
        self.camera_source = camera_source
        self.mobile_ip = mobile_ip
        self.use_ros = use_ros and ROS_AVAILABLE
        
        # Initialize ROS if available and requested
        if self.use_ros:
            rospy.init_node("aruco_detector", anonymous=True)
            self.bridge = CvBridge()
        
        # Camera capture object
        self.cap = None
        self.camera_active = False
        
        # System parameters
        self.frame_size = (640, 480)
        
        # Debug mode
        self.debug_mode = False
        self.last_preprocessed = None
        self.last_detections = {}
        
        # Interactive parameter adjustment
        self.interactive_mode = False
        self.show_parameter_menu = False
        
        # Image enhancement parameters (adjustable in real-time)
        self.enhancement_params = {
            'brightness': 0,        # -100 to 100
            'contrast': 1.0,        # 0.5 to 3.0
            'saturation': 1.0,      # 0.0 to 2.0
            'gamma': 1.0,           # 0.5 to 2.0
            'blur_kernel': 3,       # 1, 3, 5, 7
            'bilateral_d': 9,       # 5, 9, 15
            'bilateral_sigma': 75,  # 25, 50, 75, 100
            'morph_kernel': 3,      # 1, 3, 5, 7
            'clahe_limit': 2.0,     # 1.0 to 8.0
            'clahe_grid': 8,        # 4, 8, 16
        }
        
        # ArUco detection parameters (adjustable in real-time)
        self.aruco_params = {
            'adaptiveThreshConstant': 7,           # 3 to 15
            'adaptiveThreshWinSizeMin': 3,         # 3 to 10
            'adaptiveThreshWinSizeMax': 23,        # 15 to 50
            'adaptiveThreshWinSizeStep': 10,       # 5 to 15
            'minMarkerPerimeterRate': 0.03,        # 0.01 to 0.1
            'maxMarkerPerimeterRate': 4.0,         # 2.0 to 8.0
            'polygonalApproxAccuracyRate': 0.05,   # 0.01 to 0.1
            'minCornerDistanceRate': 0.05,         # 0.01 to 0.1
            'minDistanceToBorder': 3,              # 1 to 10
            'cornerRefinementWinSize': 5,          # 3 to 15
            'cornerRefinementMaxIterations': 30,   # 10 to 100
            'cornerRefinementMinAccuracy': 0.1,    # 0.01 to 1.0
        }
        
        # Debugging parameters
        self.debug_params = {
            'show_preprocessing': True,
            'show_detection_steps': True,
            'show_parameter_effects': True,
            'show_timing_info': True,
            'show_rejection_analysis': True,
            'verbose_logging': True,
            'save_debug_frames': False,
            'frame_counter': 0
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'detection_times': [],
            'preprocessing_times': [],
            'best_methods': {},
            'parameter_effectiveness': {}
        }
        
        # Camera calibration parameters
        self.calibration_done = False
        
        # Default camera matrix (will be updated during calibration)
        self.camera_matrix = np.array([
            [500.0, 0, self.frame_size[0]/2],
            [0, 500.0, self.frame_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Default distortion coefficients
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # World points for homography calculation
        # These should correspond to a reasonable bird's eye view rectangle
        # Order: [bottom-left, top-left, bottom-right, top-right] for ArUco IDs [20, 21, 22, 23]
        self.world_pts = np.array([
            [  0,   0],   # ArUco ID 20: Bottom left
            [  0, 800],   # ArUco ID 21: Top left  
            [800,   0],   # ArUco ID 22: Bottom right
            [800, 800],   # ArUco ID 23: Top right
        ], dtype=np.float32)

        # Dictionary to store ArUco information by zone
        self.info_zonas = {
            i: {'arucos': [], 'centros': []} for i in range(1, 9)
        }
        
        # Reference ArUcos for homography (IDs 20-23)
        self.arucos_medio = {i: [] for i in range(20, 24)}
        self.centros_medio_IMG_Points = []
        
        # Current homography matrix
        self.current_homography = None

        # Initialize ROS topics if ROS is available
        if self.use_ros:
            # ROS subscriptions and publications
            self.sub = rospy.Subscriber("/csi_cam_0/image_raw", Image, self.ros_image_callback, queue_size=1)
            
            # Topics for robot communication
            self.mover_lata_pub = rospy.Publisher("/info_camara_mover_lata", String, queue_size=1)
            self.mover_a_lata_pub = rospy.Publisher("/info_camara_mover_a_lata", String, queue_size=1)
            
            # Gripper control topic
            self.gripper_pub = rospy.Publisher("/gripper_control", String, queue_size=1)
            
            # Subscriptions to robot messages
            self.mover_lata_sub = rospy.Subscriber("/info_robot_mover_lata", String, 
                                                lambda msg: self.process_robot_message(msg, "mover_lata"), 
                                                queue_size=1)
            self.mover_a_lata_sub = rospy.Subscriber("/info_robot_mover_a_lata", String, 
                                                    lambda msg: self.process_robot_message(msg, "mover_a_lata"), 
                                                    queue_size=1)

        # Robot state - ENHANCED WITH ORIENTATION
        self.robot_state = {
            'lata_actual': None,
            'almacen_origen': None,
            'almacen_destino': None,
            'accion': "Parar",
            'modo': None,
            'navegando': False,
            'gripper_state': 'closed',
            'approaching_can': False,
            'can_picked': False,
            'position': None,           # (x, y) current position
            'orientation': None,        # Current heading in degrees (0-360)
            'orientation_vector': None, # (dx, dy) unit vector of robot facing direction
            'last_valid_orientation': 0.0  # Backup orientation value
        }

        self.robot_id = 25
        
        # Distance thresholds for gripper control
        self.gripper_open_threshold = 80   # Distance to open gripper (pixels)
        self.gripper_close_threshold = 30  # Distance to close gripper (pixels)
        self.destination_threshold = 50    # Distance to consider arrived at destination
        
        # Orientation calculation parameters
        self.orientation_smoothing = 0.7   # Smoothing factor for orientation (0-1)
        self.min_orientation_change = 5.0  # Minimum change in degrees to update orientation
        
        # Calibration variables
        self.calibration_frames = []
        self.calibration_count = 0
        self.max_calibration_frames = 10
        
        # Thread safety
        self.processing_lock = threading.Lock()
        
        # Load previous calibration if available
        self.load_calibration()
        
        # Initialize camera
        self.initialize_camera()
        
        print("Robot Navigation System initialized with orientation tracking")
        print("Camera source: {}".format(camera_source))
        print("Robot ArUco ID set to: {}".format(self.robot_id))

    def initialize_camera(self):
        """
        Initialize camera based on the selected source
        """
        try:
            if self.camera_source == "mobile":
                if not self.mobile_ip:
                    print("ERROR: Mobile IP not provided!")
                    print("Please provide mobile IP address (e.g., '192.168.1.100:8080')")
                    return False
                
                # Try different mobile camera URL formats
                mobile_urls = [
                    "http://{}/video".format(self.mobile_ip),
                    "http://{}/cam.mjpg".format(self.mobile_ip),
                    "http://{}/video.mjpg".format(self.mobile_ip),
                    "http://{}/mjpeg".format(self.mobile_ip),
                ]
                
                for url in mobile_urls:
                    print("Trying mobile camera URL: {}".format(url))
                    self.cap = cv2.VideoCapture(url)
                    if self.cap.isOpened():
                        print("✅ Successfully connected to mobile camera: {}".format(url))
                        self.camera_active = True
                        break
                    else:
                        self.cap.release()
                
                if not self.camera_active:
                    print("❌ Failed to connect to mobile camera")
                    print("Make sure:")
                    print("1. Your phone and computer are on the same WiFi network")
                    print("2. Camera app is running on your phone")
                    print("3. IP address is correct: {}".format(self.mobile_ip))
                    return False
                    
            elif self.camera_source == "usb":
                # Try USB cameras (0, 1, 2...)
                for i in range(3):
                    print("Trying USB camera {}...".format(i))
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        print("✅ Successfully connected to USB camera {}".format(i))
                        self.camera_active = True
                        break
                    else:
                        self.cap.release()
                
                if not self.camera_active:
                    print("❌ No USB camera found")
                    return False
                    
            elif self.camera_source == "file":
                video_file = input("Enter video file path: ")
                if os.path.exists(video_file):
                    self.cap = cv2.VideoCapture(video_file)
                    if self.cap.isOpened():
                        print("✅ Successfully opened video file: {}".format(video_file))
                        self.camera_active = True
                    else:
                        print("❌ Failed to open video file")
                        return False
                else:
                    print("❌ Video file not found: {}".format(video_file))
                    return False
                    
            elif self.camera_source == "ros":
                if not self.use_ros:
                    print("❌ ROS not available")
                    return False
                print("✅ Using ROS camera topic")
                self.camera_active = True
                
            # Set camera properties for better performance
            if self.cap and self.camera_active and self.camera_source != "ros":
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
            return self.camera_active
            
        except Exception as e:
            print("Error initializing camera: {}".format(str(e)))
            return False

    def get_frame(self):
        """
        Get a frame from the camera source
        """
        try:
            if not self.camera_active:
                return None
                
            if self.camera_source == "ros":
                # For ROS, frames are handled in ros_image_callback
                return None
                
            if self.cap is None:
                return None
                
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                print("Failed to read frame from camera")
                return None
                
        except Exception as e:
            print("Error getting frame: {}".format(str(e)))
            return None

    def ros_image_callback(self, msg):
        """
        ROS image callback (only used when camera_source is "ros")
        """
        try:
            if not self.use_ros:
                return
                
            # Convert ROS image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_frame(frame)
            
        except Exception as e:
            if self.use_ros:
                rospy.logerr("Error in ROS image callback: %s", str(e))
            else:
                print("Error in ROS image callback: {}".format(str(e)))

    def preprocess_frame(self, frame):
        """
        Enhanced preprocessing with adjustable parameters and extensive debugging
        """
        import time
        start_time = time.time()
        
        try:
            if frame is None:
                self.log_debug("❌ Input frame is None")
                return None
                
            self.debug_params['frame_counter'] += 1
            frame_id = self.debug_params['frame_counter']
            
            if self.debug_params['verbose_logging']:
                self.log_debug("🔄 Processing frame %d, shape: %s", frame_id, frame.shape)
                
            # Resize if necessary
            original_shape = frame.shape[:2]
            if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
                frame = cv2.resize(frame, self.frame_size)
                if self.debug_params['verbose_logging']:
                    self.log_debug("📐 Resized from %s to %s", original_shape, frame.shape[:2])
            
            # Apply undistortion if camera is calibrated
            if self.calibration_done:
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
                if self.debug_params['verbose_logging']:
                    self.log_debug("🔧 Applied camera undistortion")
            
            # Apply image enhancements with adjustable parameters
            enhanced_frame = self.apply_image_enhancements(frame)
            
            # Convert to grayscale
            if len(enhanced_frame.shape) == 3:
                gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = enhanced_frame.copy()
            
            if self.debug_params['verbose_logging']:
                self.log_debug("🎨 Converted to grayscale, range: [%d, %d]", gray.min(), gray.max())
            
            # Apply multiple preprocessing techniques with current parameters
            preprocessing_results = {}
            
            # Method 1: Original
            preprocessing_results['original'] = gray.copy()
            
            # Method 2: Enhanced with CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=self.enhancement_params['clahe_limit'], 
                tileGridSize=(self.enhancement_params['clahe_grid'], self.enhancement_params['clahe_grid'])
            )
            preprocessing_results['enhanced'] = clahe.apply(gray)
            
            # Method 3: Gaussian blur
            kernel_size = self.enhancement_params['blur_kernel']
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            preprocessing_results['blurred'] = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            
            # Method 4: Bilateral filter
            preprocessing_results['bilateral'] = cv2.bilateralFilter(
                gray, 
                self.enhancement_params['bilateral_d'], 
                self.enhancement_params['bilateral_sigma'], 
                self.enhancement_params['bilateral_sigma']
            )
            
            # Method 5: Morphological operations
            morph_kernel_size = self.enhancement_params['morph_kernel']
            if morph_kernel_size % 2 == 0:
                morph_kernel_size += 1
            kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
            preprocessing_results['morphological'] = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Method 6: Adaptive histogram equalization
            preprocessing_results['adaptive'] = cv2.equalizeHist(gray)
            
            # Method 7: Combined approach
            combined = clahe.apply(gray)
            combined = cv2.bilateralFilter(combined, 5, 50, 50)
            preprocessing_results['combined'] = combined
            
            # Calculate preprocessing statistics
            preprocessing_time = time.time() - start_time
            self.performance_stats['preprocessing_times'].append(preprocessing_time)
            
            if self.debug_params['show_preprocessing']:
                self.log_debug("⏱️  Preprocessing took %.3f ms", preprocessing_time * 1000)
                for method, result in preprocessing_results.items():
                    self.log_debug("📊 %s: range [%d, %d], mean=%.1f, std=%.1f", 
                                  method, result.min(), result.max(), result.mean(), result.std())
            
            # Save debug frames if requested
            if self.debug_params['save_debug_frames'] and frame_id % 10 == 0:
                for method, result in preprocessing_results.items():
                    filename = "debug_frame_{:04d}_{}.jpg".format(frame_id, method)
                    cv2.imwrite(filename, result)
                self.log_debug("💾 Saved debug frames for frame %d", frame_id)
            
            return preprocessing_results
            
        except Exception as e:
            self.log_error("❌ Error in preprocess_frame: %s", str(e))
            import traceback
            self.log_error("🔍 Traceback: %s", traceback.format_exc())
            return None

    def apply_image_enhancements(self, frame):
        """
        Apply adjustable image enhancements (brightness, contrast, saturation, gamma)
        """
        try:
            enhanced = frame.copy().astype(np.float32)
            
            # Apply brightness adjustment
            if self.enhancement_params['brightness'] != 0:
                enhanced = enhanced + self.enhancement_params['brightness']
                if self.debug_params['verbose_logging']:
                    self.log_debug("🔆 Applied brightness: %+d", self.enhancement_params['brightness'])
            
            # Apply contrast adjustment
            if self.enhancement_params['contrast'] != 1.0:
                enhanced = enhanced * self.enhancement_params['contrast']
                if self.debug_params['verbose_logging']:
                    self.log_debug("🎭 Applied contrast: %.2f", self.enhancement_params['contrast'])
            
            # Apply gamma correction
            if self.enhancement_params['gamma'] != 1.0:
                enhanced = enhanced / 255.0
                enhanced = np.power(enhanced, self.enhancement_params['gamma'])
                enhanced = enhanced * 255.0
                if self.debug_params['verbose_logging']:
                    self.log_debug("🌟 Applied gamma: %.2f", self.enhancement_params['gamma'])
            
            # Clip values to valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            # Apply saturation adjustment (only for color images)
            if len(frame.shape) == 3 and self.enhancement_params['saturation'] != 1.0:
                hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = hsv[:, :, 1] * self.enhancement_params['saturation']
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                if self.debug_params['verbose_logging']:
                    self.log_debug("🎨 Applied saturation: %.2f", self.enhancement_params['saturation'])
            
            return enhanced
            
        except Exception as e:
            self.log_error("❌ Error in apply_image_enhancements: %s", str(e))
            return frame

    def detect_arucos(self, frame):
        """
        Enhanced ArUco detection with multiple attempts and better parameters
        """
        try:
            # Clear previous detections
            for zona_id in self.info_zonas:
                self.info_zonas[zona_id]['arucos'] = []
                self.info_zonas[zona_id]['centros'] = []
            
            for aruco_id in self.arucos_medio:
                self.arucos_medio[aruco_id] = []
            
            # Get preprocessed frames
            preprocessed = self.preprocess_frame(frame)
            if preprocessed is None:
                return 0
            
            # ArUco dictionary and detection parameters
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            
            # Try multiple parameter sets for better detection
            parameter_sets = []
            
            # Parameter set 1: Standard detection
            params1 = cv2.aruco.DetectorParameters_create()
            params1.adaptiveThreshWinSizeMin = 3
            params1.adaptiveThreshWinSizeMax = 23
            params1.adaptiveThreshWinSizeStep = 10
            params1.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            params1.cornerRefinementWinSize = 5
            params1.cornerRefinementMaxIterations = 30
            params1.cornerRefinementMinAccuracy = 0.1
            parameter_sets.append(("Standard", params1))
            
            # Parameter set 2: High sensitivity
            params2 = cv2.aruco.DetectorParameters_create()
            params2.adaptiveThreshWinSizeMin = 3
            params2.adaptiveThreshWinSizeMax = 53
            params2.adaptiveThreshWinSizeStep = 4
            params2.adaptiveThreshConstant = 7
            params2.minMarkerPerimeterRate = 0.03
            params2.maxMarkerPerimeterRate = 4.0
            params2.polygonalApproxAccuracyRate = 0.05
            params2.minCornerDistanceRate = 0.05
            params2.minDistanceToBorder = 3
            params2.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            parameter_sets.append(("High Sensitivity", params2))
            
            # Parameter set 3: Low threshold
            params3 = cv2.aruco.DetectorParameters_create()
            params3.adaptiveThreshWinSizeMin = 3
            params3.adaptiveThreshWinSizeMax = 23
            params3.adaptiveThreshWinSizeStep = 10
            params3.adaptiveThreshConstant = 5  # Lower threshold
            params3.minMarkerPerimeterRate = 0.01  # Allow smaller markers
            params3.maxMarkerPerimeterRate = 8.0   # Allow larger markers
            params3.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
            parameter_sets.append(("Low Threshold", params3))
            
            # Try detection with different preprocessing and parameters
            all_detections = {}
            detection_attempts = []
            
            # Test different combinations
            for preprocess_name, gray_frame in preprocessed.items():
                for param_name, params in parameter_sets:
                    try:
                        corners, ids, rejected = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=params)
                        
                        if ids is not None and len(ids) > 0:
                            attempt_info = {
                                'preprocess': preprocess_name,
                                'params': param_name,
                                'corners': corners,
                                'ids': ids,
                                'count': len(ids),
                                'rejected': len(rejected) if rejected is not None else 0
                            }
                            detection_attempts.append(attempt_info)
                            
                            # Store unique detections
                            for i, id_ in enumerate(ids.flatten()):
                                if id_ not in all_detections:
                                    all_detections[id_] = {
                                        'corners': corners[i],
                                        'method': "{} + {}".format(preprocess_name, param_name)
                                    }
                    except Exception as e:
                        self.log_debug("Detection attempt failed (%s + %s): %s", preprocess_name, param_name, str(e))
                        continue
            
            # Log detection attempts for debugging
            if detection_attempts:
                best_attempt = max(detection_attempts, key=lambda x: x['count'])
                self.log_info("Best detection: %d markers with %s + %s", 
                             best_attempt['count'], best_attempt['preprocess'], best_attempt['params'])
                
                # Show all unique IDs found
                unique_ids = list(all_detections.keys())
                if unique_ids:
                    self.log_info("Detected ArUco IDs: %s", sorted(unique_ids))
            else:
                self.log_warn("No ArUco markers detected with any method")
                
                # Store debug data even if no detections for troubleshooting
                self.last_preprocessed = preprocessed
                self.last_detections = {}
                return 0
            
            # Process all unique detections
            total_processed = 0
            for id_, detection_data in all_detections.items():
                corners_data = detection_data['corners']
                centro = tuple(np.mean(corners_data.reshape(4, 2), axis=0).astype(int))
                
                # Zone ArUcos (9-16 correspond to zones 1-8)
                if 9 <= id_ <= 16:
                    zona = id_ - 8  # ID 9→Zone 1, ID 10→Zone 2, ..., ID 16→Zone 8
                    aruco_info = {
                        'aruco_id': id_,
                        'centro': centro,
                        'corners': corners_data.reshape(4, 2).tolist(),
                        'detection_method': detection_data['method']
                    }
                    
                    self.info_zonas[zona]['arucos'].append(aruco_info)
                    self.info_zonas[zona]['centros'].append(centro)
                    total_processed += 1
                    self.log_debug("Zone ArUco %d detected in zone %d at %s", id_, zona, centro)
                
                # Reference ArUcos for homography (20-23)
                elif 20 <= id_ <= 23:
                    self.arucos_medio[id_].append({
                        'corners': corners_data,
                        'center': centro,
                        'detection_method': detection_data['method']
                    })
                    total_processed += 1
                    self.log_debug("Reference ArUco %d detected at %s", id_, centro)
                
                # Robot ArUco
                elif id_ == self.robot_id:
                    self.log_info("Robot ArUco %d detected at position: %s (method: %s)", 
                                 id_, centro, detection_data['method'])
                    
                    # Store robot position globally (not tied to a specific zone)
                    aruco_info = {
                        'aruco_id': id_,
                        'centro': centro,
                        'corners': corners_data.reshape(4, 2).tolist(),
                        'detection_method': detection_data['method']
                    }
                    
                    # Add to zone 1 for compatibility (this should be improved)
                    self.info_zonas[1]['arucos'].append(aruco_info)
                    self.info_zonas[1]['centros'].append(centro)
                    total_processed += 1
                
                # Can ArUcos (IDs 1-8 for individual cans)
                elif 1 <= id_ <= 8:
                    # These are individual can markers - assign to appropriate zones
                    # For now, assign to zone based on position or predefined mapping
                    zona = ((id_ - 1) % 6) + 1  # Distribute cans across zones 1-6
                    aruco_info = {
                        'aruco_id': id_,
                        'centro': centro,
                        'corners': corners_data.reshape(4, 2).tolist(),
                        'detection_method': detection_data['method'],
                        'type': 'can'
                    }
                    
                    self.info_zonas[zona]['arucos'].append(aruco_info)
                    self.info_zonas[zona]['centros'].append(centro)
                    total_processed += 1
                    self.log_debug("Can ArUco %d detected in zone %d at %s", id_, zona, centro)
                
                else:
                    self.log_debug("Unknown ArUco ID %d detected at %s", id_, centro)
            
            self.log_info("Processed %d ArUco markers total", total_processed)
            
            # Store debug data
            self.last_preprocessed = preprocessed
            self.last_detections = all_detections
            
            return total_processed
            
        except Exception as e:
            self.log_error("Error in detect_arucos: %s", str(e))
            return 0

    def calculate_homography(self, frame):
        """
        Calculates homography matrix using reference ArUcos
        """
        try:
            # Check if we have all 4 reference ArUcos
            available_arucos = sum(1 for aruco_id in range(20, 24) 
                                 if self.arucos_medio[aruco_id])
            
            if available_arucos < 4:
                # Use log_info with throttling simulation
                if not hasattr(self, '_last_homography_warning') or time.time() - self._last_homography_warning > 2:
                    self.log_info("Need 4 reference ArUcos (20-23). Only have %d", available_arucos)
                    self._last_homography_warning = time.time()
                return None
                
            # Extract centers of the 4 reference ArUcos
            pts = np.zeros((4, 2), dtype=np.float32)
            for i, aruco_id in enumerate(range(20, 24)):
                if not self.arucos_medio[aruco_id]:
                    return None
                    
                aruco_data = self.arucos_medio[aruco_id][-1]
                center = aruco_data['center']
                pts[i] = [center[0], center[1]]
                
                # Draw reference point for debugging
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, "Ref {}".format(aruco_id), 
                           (center[0], center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Calculate homography
            H, status = cv2.findHomography(pts, self.world_pts, cv2.RANSAC, 5.0)
            
            if H is not None and status.sum() >= 3:
                self.current_homography = H
                return H
            else:
                self.log_warn("Could not calculate good homography")
                return None
                
        except Exception as e:
            self.log_error("Error calculating homography: %s", str(e))
            return None

    def draw_zone_info(self, frame, zona):
        """
        Draws zone information on the frame
        """
        try:
            if zona not in self.info_zonas:
                return frame
                
            zona_info = self.info_zonas[zona]
            
            # Draw ArUco centers
            for centro in zona_info['centros']:
                cv2.circle(frame, centro, 4, (0, 255, 0), -1)
            
            # Draw ArUco labels
            for aruco in zona_info['arucos']:
                center = aruco['centro']
                label = "ID:{} Z:{}".format(aruco['aruco_id'], zona)
                cv2.putText(frame, label, (center[0] + 10, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return frame
            
        except Exception as e:
            self.log_error("Error in draw_zone_info: %s", str(e))
            return frame

    def draw_navigation_vector(self, frame, warped_frame, H):
        """
        Draws navigation vectors on both original and warped frames
        """
        try:
            pos_robot = self.get_robot_position()
            if not pos_robot:
                return
                
            destino_id = self.robot_state['almacen_destino']
            pos_destino = self.get_zone_position(destino_id)
            
            if not pos_destino:
                return
            
            # Draw arrow on original frame
            cv2.arrowedLine(frame, pos_robot, pos_destino, (0, 0, 255), 2)
            
            # Transform positions to warped space and draw arrow
            robot_pts = np.array([[pos_robot]], dtype=np.float32)
            destino_pts = np.array([[pos_destino]], dtype=np.float32)
            
            warped_robot = cv2.perspectiveTransform(robot_pts, H)[0][0]
            warped_destino = cv2.perspectiveTransform(destino_pts, H)[0][0]
            
            warped_robot = (int(warped_robot[0]), int(warped_robot[1]))
            warped_destino = (int(warped_destino[0]), int(warped_destino[1]))
            
            cv2.arrowedLine(warped_frame, warped_robot, warped_destino, (0, 0, 255), 2)
            
        except Exception as e:
            self.log_error("Error drawing navigation vector: %s", str(e))

    def process_robot_message(self, msg, topic_type):
        """
        Processes messages received from robot topics
        """
        try:
            # Parse JSON message
            try:
                data = json.loads(msg.data)
                
                # Map alternative key names
                if 'alm1' in data:
                    data['origen'] = data['alm1']
                if 'alm2' in data:
                    data['destino'] = data['alm2']
                    
            except ValueError:
                # Parse key:value format
                data = {}
                for item in msg.data.split(','):
                    if ':' in item:
                        key, value = item.split(':', 1)
                        data[key.strip()] = value.strip()
            
            self.log_info("Received %s message: %s", topic_type, data)
            
            # Update robot state
            with self.processing_lock:
                self.robot_state['modo'] = topic_type
                
                if topic_type == "mover_lata":
                    if 'lata' in data:
                        self.robot_state['lata_actual'] = int(data['lata'])
                    if 'origen' in data:
                        self.robot_state['almacen_origen'] = int(data['origen'])
                    if 'destino' in data:
                        self.robot_state['almacen_destino'] = int(data['destino'])
                        
                elif topic_type == "mover_a_lata":
                    if 'origen' in data:
                        self.robot_state['almacen_origen'] = int(data['origen'])
                    if 'destino' in data:
                        self.robot_state['almacen_destino'] = int(data['destino'])
                
                self.robot_state['navegando'] = True
            
            # Calculate and execute navigation
            self.execute_navigation()
            
        except Exception as e:
            self.log_error("Error processing robot message %s: %s", topic_type, str(e))

    def execute_navigation(self):
        """
        Calculates and executes navigation commands
        """
        try:
            with self.processing_lock:
                if not self.robot_state['navegando']:
                    return
                    
                origen_id = self.robot_state['almacen_origen']
                destino_id = self.robot_state['almacen_destino']
                
                if origen_id is None or destino_id is None:
                    self.log_warn("Origin or destination not defined")
                    return
                
                # Get robot position
                pos_robot = self.get_robot_position()
                if not pos_robot:
                    self.log_warn("Cannot locate robot")
                    action = "Parar"
                else:
                    # Get zone positions
                    pos_origen = self.get_zone_position(origen_id)
                    pos_destino = self.get_zone_position(destino_id)
                    
                    if not pos_origen or not pos_destino:
                        self.log_warn("Cannot get zone positions")
                        action = "Parar"
                    else:
                        # Calculate navigation direction
                        action = self.calculate_robot_direction(pos_robot, pos_origen, pos_destino)
                
                # Check gripper control based on distance to target
                self.check_gripper_control()
                
                # Update robot state
                self.robot_state['accion'] = action
                
                # Publish action
                message = "accion:{}".format(action)
                if self.robot_state['modo'] == "mover_lata":
                    self.mover_lata_pub.publish(String(data=message))
                elif self.robot_state['modo'] == "mover_a_lata":
                    self.mover_a_lata_pub.publish(String(data=message))
                
                self.log_info("Published action: %s", action)
                
                # Check if navigation is complete
                if action == "Destino":
                    self.robot_state['navegando'] = False
                    # Reset gripper states for next task
                    self.robot_state['approaching_can'] = False
                    self.robot_state['can_picked'] = False
                    
                    completion_msg = "Destino"
                    if self.robot_state['modo'] == "mover_lata":
                        self.mover_lata_pub.publish(String(data=completion_msg))
                    else:
                        self.mover_a_lata_pub.publish(String(data=completion_msg))
                    self.log_info("Navigation completed: %s", self.robot_state['modo'])
                    
        except Exception as e:
            self.log_error("Error in navigation execution: %s", str(e))

    def get_zone_position(self, zona_id):
        """
        Gets the center position of a specific zone
        """
        try:
            if zona_id not in self.info_zonas:
                return None
                
            zona = self.info_zonas[zona_id]
            centros = zona['centros']
            
            if not centros:
                return None
                
            # Calculate average center
            x_avg = sum(float(c[0]) for c in centros) / len(centros)
            y_avg = sum(float(c[1]) for c in centros) / len(centros)
            
            return (int(x_avg), int(y_avg))
            
        except Exception as e:
            self.log_error("Error getting zone %d position: %s", zona_id, str(e))
            return None

    def get_robot_position(self):
        """
        Gets current robot position and orientation based on robot ArUco detection
        """
        try:
            for zona_id, zona_info in self.info_zonas.items():
                for aruco in zona_info['arucos']:
                    if aruco['aruco_id'] == self.robot_id:
                        position = aruco['centro']
                        
                        # Calculate orientation from ArUco corners
                        if 'corners' in aruco:
                            orientation_data = self.calculate_robot_orientation(aruco['corners'])
                            
                            # Update robot state with position and orientation
                            with self.processing_lock:
                                self.robot_state['position'] = position
                                if orientation_data:
                                    self.robot_state['orientation'] = orientation_data['angle']
                                    self.robot_state['orientation_vector'] = orientation_data['vector']
                                    self.robot_state['last_valid_orientation'] = orientation_data['angle']
                        
                        return position
            return None
            
        except Exception as e:
            self.log_error("Error getting robot position: %s", str(e))
            return None

    def calculate_robot_orientation(self, corners):
        """
        Calculates robot orientation from ArUco marker corners
        
        Args:
            corners: List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    ArUco corners are ordered: top-left, top-right, bottom-right, bottom-left
        
        Returns:
            dict: {
                'angle': float,      # Orientation in degrees (0-360)
                'vector': tuple,     # (dx, dy) unit vector of facing direction
                'confidence': float  # Confidence in the measurement (0-1)
            }
        """
        try:
            if not corners or len(corners) != 4:
                return None
                
            # Convert to numpy array for easier manipulation
            corners_array = np.array(corners, dtype=np.float32)
            
            # Calculate the center of the marker
            center = np.mean(corners_array, axis=0)
            
            # Method 1: Use the direction from center to top edge center
            # Top edge is between corners 0 and 1 (top-left to top-right)
            top_left = corners_array[0]
            top_right = corners_array[1]
            top_center = (top_left + top_right) / 2.0
            
            # Vector from marker center to top center (this is the "forward" direction)
            forward_vector = top_center - center
            
            # Calculate angle in degrees (0 = pointing right, 90 = pointing up)
            angle_rad = np.arctan2(forward_vector[1], forward_vector[0])
            angle_deg = np.degrees(angle_rad)
            
            # Normalize to 0-360 degrees
            if angle_deg < 0:
                angle_deg += 360
                
            # Create unit vector
            magnitude = np.linalg.norm(forward_vector)
            if magnitude > 0:
                unit_vector = forward_vector / magnitude
            else:
                unit_vector = np.array([1.0, 0.0])  # Default to pointing right
            
            # Calculate confidence based on marker size and shape regularity
            confidence = self.calculate_orientation_confidence(corners_array)
            
            # Apply smoothing if we have a previous orientation
            if self.robot_state['orientation'] is not None:
                prev_angle = self.robot_state['orientation']
                angle_diff = abs(angle_deg - prev_angle)
                
                # Handle wrap-around (e.g., 359° to 1°)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Only update if change is significant or confidence is high
                if angle_diff < self.min_orientation_change and confidence < 0.8:
                    # Use smoothed value
                    smoothing = self.orientation_smoothing
                    angle_deg = prev_angle * smoothing + angle_deg * (1 - smoothing)
            
            return {
                'angle': float(angle_deg),
                'vector': (float(unit_vector[0]), float(unit_vector[1])),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            self.log_error("Error calculating robot orientation: %s", str(e))
            return None

    def calculate_orientation_confidence(self, corners):
        """
        Calculates confidence in orientation measurement based on marker quality
        
        Args:
            corners: numpy array of 4 corner points
            
        Returns:
            float: Confidence value between 0 and 1
        """
        try:
            # Calculate side lengths
            side_lengths = []
            for i in range(4):
                p1 = corners[i]
                p2 = corners[(i + 1) % 4]
                length = np.linalg.norm(p2 - p1)
                side_lengths.append(length)
            
            # Check if marker is roughly square (good for orientation)
            mean_length = np.mean(side_lengths)
            if mean_length == 0:
                return 0.0
                
            length_variance = np.var(side_lengths) / (mean_length ** 2)
            
            # Calculate area (larger markers are more reliable)
            area = cv2.contourArea(corners)
            
            # Confidence based on:
            # 1. How square the marker is (low variance in side lengths)
            # 2. Size of the marker (larger = more reliable)
            # 3. Minimum area threshold
            
            square_confidence = max(0, 1.0 - length_variance * 10)  # Penalize non-square shapes
            size_confidence = min(1.0, area / 1000.0)  # Normalize area to 0-1
            
            overall_confidence = (square_confidence * 0.7 + size_confidence * 0.3)
            
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception as e:
            self.log_error("Error calculating orientation confidence: %s", str(e))
            return 0.0

    def calculate_distance(self, point1, point2):
        """
        Calculates Euclidean distance between two points
        """
        return np.sqrt(float(point2[0] - point1[0])**2 + float(point2[1] - point1[1])**2)

    def control_gripper(self, action):
        """
        Controls the robot gripper
        
        Args:
            action: "open" or "close"
        """
        try:
            if action not in ["open", "close"]:
                self.log_warn("Invalid gripper action: %s", action)
                return
                
            # Only send command if state is changing
            if self.robot_state['gripper_state'] != action:
                message = "gripper:{}".format(action)
                self.gripper_pub.publish(String(data=message))
                self.robot_state['gripper_state'] = action
                self.log_info("Gripper command sent: %s", action)
                
        except Exception as e:
            self.log_error("Error controlling gripper: %s", str(e))

    def get_target_can_position(self):
        """
        Gets the position of the target can based on current task
        
        Returns:
            tuple: (x, y) position of target can, or None if not found
        """
        try:
            if self.robot_state['modo'] == "mover_lata":
                # Look for the specific can ID in the origin zone
                lata_id = self.robot_state['lata_actual']
                origen_id = self.robot_state['almacen_origen']
                
                if origen_id in self.info_zonas:
                    for aruco in self.info_zonas[origen_id]['arucos']:
                        if aruco['aruco_id'] == lata_id:
                            return aruco['centro']
                            
            elif self.robot_state['modo'] == "mover_a_lata":
                # For "mover_a_lata", navigate to the destination zone center
                # Zones 7-8 are destination storage zones where cans accumulate
                destino_id = self.robot_state['almacen_destino']
                
                if destino_id in self.info_zonas:
                    # Return the zone center position for navigation
                    # This allows the robot to navigate to the zone for dropping cans
                    return self.get_zone_position(destino_id)
                            
            return None
            
        except Exception as e:
            self.log_error("Error getting target can position: %s", str(e))
            return None

    def check_gripper_control(self):
        """
        Checks if gripper should be opened or closed based on distance to target can
        """
        try:
            if not self.robot_state['navegando']:
                return
                
            robot_pos = self.get_robot_position()
            target_can_pos = self.get_target_can_position()
            
            if not robot_pos or not target_can_pos:
                return
                
            distance_to_can = self.calculate_distance(robot_pos, target_can_pos)
            
            # Logic for gripper control based on task mode
            if self.robot_state['modo'] == "mover_lata":
                # Moving a can: open gripper when approaching, close when close enough
                if not self.robot_state['can_picked']:
                    if distance_to_can <= self.gripper_open_threshold and self.robot_state['gripper_state'] == 'closed':
                        self.control_gripper("open")
                        self.robot_state['approaching_can'] = True
                        self.log_info("Approaching can, opening gripper. Distance: %.1f", distance_to_can)
                        
                    elif distance_to_can <= self.gripper_close_threshold and self.robot_state['gripper_state'] == 'open':
                        self.control_gripper("close")
                        self.robot_state['can_picked'] = True
                        self.robot_state['approaching_can'] = False
                        self.log_info("Can reached, closing gripper. Distance: %.1f", distance_to_can)
                        
            elif self.robot_state['modo'] == "mover_a_lata":
                # Moving to a destination zone: prepare gripper for can placement
                # Since zones 7-8 are storage zones, robot should arrive with gripper closed (holding can)
                # and open gripper to drop the can when close to zone center
                if distance_to_can <= self.gripper_close_threshold and self.robot_state['gripper_state'] == 'closed':
                    # Robot has arrived at destination zone with can - open gripper to drop it
                    self.control_gripper("open")
                    self.log_info("Arrived at destination zone, opening gripper to drop can. Distance: %.1f", distance_to_can)
                    
        except Exception as e:
            self.log_error("Error in gripper control check: %s", str(e))

    def calculate_robot_direction(self, pos_actual, pos_origen, pos_destino):
        """
        Calculates the direction the robot should take using ACTUAL robot orientation
        Enhanced with proper orientation-based navigation
        """
        try:
            # Calculate distances
            dist_to_origin = self.calculate_distance(pos_actual, pos_origen)
            dist_to_destination = self.calculate_distance(pos_actual, pos_destino)
            
            # Check if we're close to destination first
            if dist_to_destination < self.destination_threshold:
                # Handle gripper control at destination
                if self.robot_state['modo'] == "mover_lata" and self.robot_state['can_picked']:
                    # Drop the can - open gripper
                    self.control_gripper("open")
                    self.robot_state['can_picked'] = False
                    self.log_info("Destination reached, dropping can")
                return "Destino"
            
            # Get current robot orientation
            current_orientation = self.robot_state.get('orientation')
            orientation_vector = self.robot_state.get('orientation_vector')
            
            if current_orientation is None or orientation_vector is None:
                # Fallback to old method if orientation is not available
                self.log_warn("Robot orientation not available, using fallback navigation")
                return self.calculate_robot_direction_fallback(pos_actual, pos_origen, pos_destino)
            
            # Calculate desired direction vector (from current position to destination)
            desired_vector = (pos_destino[0] - pos_actual[0], pos_destino[1] - pos_actual[1])
            desired_magnitude = np.sqrt(float(desired_vector[0])**2 + float(desired_vector[1])**2)
            
            if desired_magnitude == 0:
                return "Parar"
            
            # Normalize desired direction vector
            desired_unit_vector = (desired_vector[0] / desired_magnitude, desired_vector[1] / desired_magnitude)
            
            # Calculate desired angle
            desired_angle = np.degrees(np.arctan2(desired_vector[1], desired_vector[0]))
            if desired_angle < 0:
                desired_angle += 360
            
            # Calculate angle difference between current orientation and desired direction
            angle_diff = desired_angle - current_orientation
            
            # Normalize angle difference to [-180, 180]
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360
            
            # Log orientation information for debugging
            self.log_debug("Robot orientation: %.1f°, Desired: %.1f°, Diff: %.1f°", 
                          current_orientation, desired_angle, angle_diff)
            
            # Decision logic based on angle difference
            angle_tolerance = 15.0  # Degrees of tolerance for "straight" movement
            large_angle_threshold = 120.0  # Threshold for backward movement
            
            if abs(angle_diff) <= angle_tolerance:
                # Robot is facing roughly the right direction - go straight
                return "Recto"
                
            elif abs(angle_diff) >= large_angle_threshold:
                # Large angle difference - it might be easier to go backward
                # Check if going backward would be more efficient
                backward_angle_diff = angle_diff + 180 if angle_diff < 0 else angle_diff - 180
                
                if abs(backward_angle_diff) < abs(angle_diff):
                    self.log_info("Large angle difference (%.1f°), moving backward", angle_diff)
                    return "Atras"
                else:
                    # Still need to turn, decide direction
                    return "GirarD" if angle_diff > 0 else "GirarI"
                    
            elif angle_diff > angle_tolerance:
                # Need to turn left (counter-clockwise)
                return "GirarI"
                
            elif angle_diff < -angle_tolerance:
                # Need to turn right (clockwise)
                return "GirarD"
            
            else:
                # Should not reach here, but safety fallback
                return "Parar"
                
        except Exception as e:
            self.log_error("Error calculating robot direction: %s", str(e))
            return "Parar"

    def calculate_robot_direction_fallback(self, pos_actual, pos_origen, pos_destino):
        """
        Fallback navigation method when orientation is not available
        Uses the old position-based approach
        """
        try:
            self.log_warn("Using fallback navigation - orientation not available")
            
            # Calculate distances
            dist_to_origin = self.calculate_distance(pos_actual, pos_origen)
            dist_to_destination = self.calculate_distance(pos_actual, pos_destino)
            
            # Calculate direction vectors
            vector_to_destination = (pos_destino[0] - pos_actual[0], pos_destino[1] - pos_actual[1])
            vector_from_origin = (pos_actual[0] - pos_origen[0], pos_actual[1] - pos_origen[1])
            
            # Calculate magnitude of vector to destination
            mag_to_dest = np.sqrt(float(vector_to_destination[0])**2 + float(vector_to_destination[1])**2)
            
            # Avoid division by zero
            if mag_to_dest == 0:
                return "Parar"
            
            # Special case: if robot is very close to origin and needs to move away first
            origin_proximity_threshold = 40
            if dist_to_origin < origin_proximity_threshold:
                # Calculate if we need to move backward from origin
                dot_origin_dest = (vector_from_origin[0] * vector_to_destination[0] + 
                                  vector_from_origin[1] * vector_to_destination[1])
                
                # If destination is roughly in the opposite direction from where we came from origin
                if dot_origin_dest < 0:
                    self.log_info("Moving backward from origin (fallback)")
                    return "Atras"
            
            # Normal navigation logic
            # Calculate current heading (assume robot is facing away from origin)
            mag_from_origin = np.sqrt(float(vector_from_origin[0])**2 + float(vector_from_origin[1])**2)
            
            if mag_from_origin > 0:
                # Calculate angle between current heading and desired direction
                dot_product = (vector_from_origin[0] * vector_to_destination[0] + 
                              vector_from_origin[1] * vector_to_destination[1])
                cos_angle = float(dot_product) / (mag_from_origin * mag_to_dest)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to avoid numerical errors
                
                angle_rad = np.arccos(cos_angle)
                angle_degrees = np.degrees(angle_rad)
                
                # Calculate cross product to determine turn direction
                cross_product = (vector_from_origin[0] * vector_to_destination[1] - 
                               vector_from_origin[1] * vector_to_destination[0])
                
                # Decision logic with improved thresholds
                if angle_degrees < 15:  # Small angle threshold for going straight
                    return "Recto"
                elif angle_degrees > 150:  # Large angle - need to go backward
                    return "Atras"
                elif cross_product > 0:
                    return "GirarD"  # Turn right
                else:
                    return "GirarI"  # Turn left
            else:
                # Robot is at origin, decide initial direction
                if abs(vector_to_destination[0]) > abs(vector_to_destination[1]):
                    return "GirarD" if vector_to_destination[0] > 0 else "GirarI"
                else:
                    return "Recto" if vector_to_destination[1] > 0 else "Atras"
                
        except Exception as e:
            self.log_error("Error in fallback direction calculation: %s", str(e))
            return "Parar"

    def save_calibration(self):
        """
        Saves calibration parameters to file
        """
        try:
            np.savez('camera_calibration.npz', 
                    camera_matrix=self.camera_matrix, 
                    dist_coeffs=self.dist_coeffs,
                    frame_size=np.array(self.frame_size))
            
            self.log_info("Calibration parameters saved to 'camera_calibration.npz'")
            return True
            
        except Exception as e:
            self.log_error("Error saving calibration: %s", str(e))
            return False

    def load_calibration(self):
        """
        Loads calibration parameters from file
        """
        try:
            if os.path.exists('camera_calibration.npz'):
                data = np.load('camera_calibration.npz')
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                saved_size = data['frame_size']
                
                if tuple(saved_size) == self.frame_size:
                    self.calibration_done = True
                    self.log_info("Calibration parameters loaded")
                    return True
                else:
                    self.log_warn("Frame size mismatch in calibration")
            return False
            
        except Exception as e:
            self.log_error("Error loading calibration: %s", str(e))
            return False

    def process_frame(self, frame):
        """
        Main image processing function
        """
        try:
            # Preprocess frame
            preprocessed = self.preprocess_frame(frame)
            if preprocessed is None:
                return
            
            # Detect ArUcos
            num_detected = self.detect_arucos(frame)
            
            # Calculate homography and create bird's eye view
            H = self.calculate_homography(frame)
            warped_frame = None
            
            if H is not None:
                warped_frame = cv2.warpPerspective(frame, H, (1000, 1000))
                
                # Draw zone information on warped frame
                for zona in range(1, 9):
                    warped_frame = self.draw_zone_info(warped_frame, zona)
                
                # Draw navigation vector if navigating
                if self.robot_state['navegando']:
                    self.draw_navigation_vector(frame, warped_frame, H)
                
                cv2.imshow("Vista Cenital", warped_frame)
            
            # Draw zone information on original frame
            frame_viz = frame.copy()
            for zona in range(1, 9):
                frame_viz = self.draw_zone_info(frame_viz, zona)
            
            # Add status information
            status_text = "Mode: {} | Action: {} | Detected: {}".format(
                self.robot_state.get('modo', 'None'),
                self.robot_state.get('accion', 'None'),
                num_detected
            )
            cv2.putText(frame_viz, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add gripper status
            gripper_text = "Gripper: {} | Can: {}".format(
                self.robot_state.get('gripper_state', 'unknown'),
                'Picked' if self.robot_state.get('can_picked', False) else 'Free'
            )
            cv2.putText(frame_viz, gripper_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add orientation information
            orientation = self.robot_state.get('orientation')
            if orientation is not None:
                orientation_text = "Orientation: {:.1f}°".format(orientation)
                cv2.putText(frame_viz, orientation_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw orientation arrow on robot if position is known
                robot_pos = self.robot_state.get('position')
                orientation_vector = self.robot_state.get('orientation_vector')
                if robot_pos and orientation_vector:
                    # Calculate end point for orientation arrow
                    arrow_length = 50
                    end_x = int(robot_pos[0] + orientation_vector[0] * arrow_length)
                    end_y = int(robot_pos[1] + orientation_vector[1] * arrow_length)
                    
                    # Draw orientation arrow
                    cv2.arrowedLine(frame_viz, robot_pos, (end_x, end_y), (0, 255, 0), 3)
                    cv2.putText(frame_viz, "Robot Heading", 
                               (robot_pos[0] + 10, robot_pos[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                orientation_text = "Orientation: Unknown"
                cv2.putText(frame_viz, orientation_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add camera source info
            camera_text = "Camera: {}".format(self.camera_source)
            cv2.putText(frame_viz, camera_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("ArUco Detection", frame_viz)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False  # Signal to quit
            elif key == ord('c'):
                # Capture calibration frame
                self.log_info("Calibration frame captured")
            elif key == ord('r'):
                # Reset robot state
                self.robot_state['navegando'] = False
                self.robot_state['accion'] = "Parar"
                self.log_info("Robot state reset")
            elif key == ord('d'):
                # Toggle debug mode
                self.debug_mode = not self.debug_mode
                self.log_info("Debug mode: %s", "ON" if self.debug_mode else "OFF")
                if not self.debug_mode:
                    cv2.destroyWindow("ArUco Detection Debug")
            
            # Show debug windows if enabled
            if self.debug_mode and self.last_preprocessed and self.last_detections:
                self.show_debug_windows(frame, self.last_preprocessed, self.last_detections)
            
            return True
            
        except Exception as e:
            self.log_error("Error in process_frame: %s", str(e))
            return True

    def show_debug_windows(self, frame, preprocessed, all_detections):
        """
        Shows debug windows with different preprocessing methods and detection results
        """
        try:
            if not preprocessed:
                return
            
            # Create a combined view of all preprocessing methods
            methods = ['original', 'enhanced', 'blurred', 'bilateral', 'morphological']
            debug_frames = []
            
            for method in methods:
                if method in preprocessed:
                    debug_frame = preprocessed[method].copy()
                    
                    # Draw detected ArUcos on this frame
                    for id_, detection_data in all_detections.items():
                        if detection_data['method'].startswith(method):
                            corners = detection_data['corners'].reshape(4, 2).astype(int)
                            cv2.polylines(debug_frame, [corners], True, (255, 255, 255), 2)
                            center = tuple(np.mean(corners, axis=0).astype(int))
                            cv2.putText(debug_frame, str(id_), center, 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add method label
                    cv2.putText(debug_frame, method.upper(), (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Resize for display
                    debug_frame = cv2.resize(debug_frame, (320, 240))
                    debug_frames.append(debug_frame)
            
            # Combine frames into a grid
            if len(debug_frames) >= 4:
                top_row = np.hstack([debug_frames[0], debug_frames[1]])
                bottom_row = np.hstack([debug_frames[2], debug_frames[3]])
                if len(debug_frames) == 5:
                    # Add the 5th frame below
                    combined = np.vstack([top_row, bottom_row])
                    fifth_frame = cv2.resize(debug_frames[4], (640, 240))
                    combined = np.vstack([combined, fifth_frame])
                else:
                    combined = np.vstack([top_row, bottom_row])
                
                cv2.imshow("ArUco Detection Debug", combined)
            
        except Exception as e:
            self.log_error("Error in debug visualization: %s", str(e))

    def log_info(self, message, *args):
        """
        Unified logging function that works with or without ROS
        """
        if self.use_ros and ROS_AVAILABLE:
            rospy.loginfo(message, *args)
        else:
            if args:
                print(message % args)
            else:
                print(message)

    def log_warn(self, message, *args):
        """
        Unified warning logging function
        """
        if self.use_ros and ROS_AVAILABLE:
            rospy.logwarn(message, *args)
        else:
            if args:
                print("WARNING: " + message % args)
            else:
                print("WARNING: " + message)

    def log_error(self, message, *args):
        """
        Unified error logging function
        """
        if self.use_ros and ROS_AVAILABLE:
            rospy.logerr(message, *args)
        else:
            if args:
                print("ERROR: " + message % args)
            else:
                print("ERROR: " + message)

    def log_debug(self, message, *args):
        """
        Unified debug logging function
        """
        if self.use_ros and ROS_AVAILABLE:
            rospy.logdebug(message, *args)
        else:
            if args:
                print("DEBUG: " + message % args)
            else:
                print("DEBUG: " + message)

    def run(self):
        """
        Main execution loop
        """
        self.log_info("Starting robot navigation system...")
        self.log_info("Camera source: %s", self.camera_source)
        self.log_info("Press 'q' to quit, 'c' to capture calibration frame, 'r' to reset robot state")
        
        try:
            if self.camera_source == "ros" and self.use_ros:
                # Use ROS spin for ROS camera
                rospy.spin()
            else:
                # Use manual frame capture loop for other camera sources
                while True:
                    frame = self.get_frame()
                    if frame is not None:
                        if not self.process_frame(frame):
                            break  # Quit signal received
                    else:
                        self.log_info("No frame received, retrying...")
                        time.sleep(0.1)
                        
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.03)  # ~30 FPS
                    
        except KeyboardInterrupt:
            self.log_info("Shutting down robot navigation system")
        except Exception as e:
            self.log_error("Error in main loop: %s", str(e))
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Clean up resources
        """
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.log_info("Cleanup completed")
        except Exception as e:
            self.log_error("Error during cleanup: %s", str(e))

    def generate_aruco_markers(self, output_dir="aruco_markers"):
        """
        Generates ArUco markers for testing
        """
        try:
            import os
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            
            # Generate markers for different purposes
            marker_sets = {
                'reference': [20, 21, 22, 23],  # Reference markers for homography
                'zones': list(range(9, 17)),     # Zone markers (9-16 for zones 1-8)
                'cans': list(range(1, 9)),       # Can markers (1-8)
                'robot': [self.robot_id]         # Robot marker
            }
            
            marker_size = 200  # pixels
            
            for set_name, marker_ids in marker_sets.items():
                set_dir = os.path.join(output_dir, set_name)
                if not os.path.exists(set_dir):
                    os.makedirs(set_dir)
                
                for marker_id in marker_ids:
                    # Generate marker
                    marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)
                    
                    # Add border and label
                    bordered_img = np.ones((marker_size + 60, marker_size + 60), dtype=np.uint8) * 255
                    bordered_img[30:30+marker_size, 30:30+marker_size] = marker_img
                    
                    # Add ID label
                    cv2.putText(bordered_img, "ID: {}".format(marker_id), 
                               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Add purpose label
                    purpose = ""
                    if set_name == 'reference':
                        purpose = "Reference"
                    elif set_name == 'zones':
                        zone_num = marker_id - 8
                        purpose = "Zone {}".format(zone_num)
                    elif set_name == 'cans':
                        purpose = "Can {}".format(marker_id)
                    elif set_name == 'robot':
                        purpose = "Robot"
                    
                    cv2.putText(bordered_img, purpose, 
                               (10, marker_size + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Save marker
                    filename = "aruco_{:02d}_{}.png".format(marker_id, purpose.lower().replace(" ", "_"))
                    filepath = os.path.join(set_dir, filename)
                    cv2.imwrite(filepath, bordered_img)
                    
                    self.log_info("Generated marker: %s", filepath)
            
            self.log_info("ArUco markers generated in directory: %s", output_dir)
            return True
            
        except Exception as e:
            self.log_error("Error generating ArUco markers: %s", str(e))
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot Navigation System with Mobile Camera Support')
    parser.add_argument('--camera', choices=['mobile', 'usb', 'file', 'ros'], default='mobile',
                       help='Camera source type (default: mobile)')
    parser.add_argument('--ip', type=str, 
                       help='Mobile camera IP address (e.g., 192.168.1.100:8080)')
    parser.add_argument('--ros', action='store_true',
                       help='Enable ROS functionality')
    parser.add_argument('--robot-id', type=int, default=0,
                       help='Robot ArUco marker ID (default: 0)')
    parser.add_argument('--generate-markers', action='store_true',
                       help='Generate ArUco markers for testing and exit')
    
    args = parser.parse_args()
    
    # Generate markers if requested
    if args.generate_markers:
        print("🎯 Generating ArUco markers...")
        navigator = Navegacion_Robot(camera_source='usb', use_ros=False)
        if args.robot_id != 0:
            navigator.robot_id = args.robot_id
        success = navigator.generate_aruco_markers()
        if success:
            print("✅ ArUco markers generated successfully!")
            print("📁 Check the 'aruco_markers' directory")
        else:
            print("❌ Failed to generate markers")
        exit(0)
    
    # Print usage instructions
    print("=" * 60)
    print("🤖 ROBOT NAVIGATION SYSTEM WITH MOBILE CAMERA")
    print("=" * 60)
    
    if args.camera == 'mobile':
        if not args.ip:
            print("📱 MOBILE CAMERA SETUP INSTRUCTIONS:")
            print("1. Install 'IP Webcam' app on your Android phone")
            print("   or 'EpocCam' for iPhone")
            print("2. Connect your phone and computer to the same WiFi")
            print("3. Start the camera app and note the IP address")
            print("4. Run this script with: --ip YOUR_PHONE_IP:8080")
            print("\nExample:")
            print("  python visión_cenitalV1_movil.py --camera mobile --ip 192.168.1.100:8080")
            print("\nAlternatively, enter IP now:")
            try:
                # Python 2/3 compatibility for input
                mobile_ip = raw_input("Enter mobile IP (e.g., 192.168.1.100:8080): ").strip()
            except NameError:
                # Python 3
                mobile_ip = input("Enter mobile IP (e.g., 192.168.1.100:8080): ").strip()
            
            if not mobile_ip:
                print("❌ No IP provided. Exiting.")
                exit(1)
            args.ip = mobile_ip
        
        print("📱 Using mobile camera at: {}".format(args.ip))
        
    elif args.camera == 'usb':
        print("🔌 Using USB camera")
        
    elif args.camera == 'file':
        print("📁 Using video file")
        
    elif args.camera == 'ros':
        print("🤖 Using ROS camera topic")
        if not ROS_AVAILABLE:
            print("❌ ROS not available!")
            exit(1)
    
    print("\n🎯 CONTROLS:")
    print("  'q' - Quit")
    print("  'c' - Capture calibration frame")
    print("  'r' - Reset robot state")
    print("  'd' - Toggle debug mode (shows detection methods)")
    print("=" * 60)
    
    try:
        # Create navigator instance
        navigator = Navegacion_Robot(
            camera_source=args.camera,
            mobile_ip=args.ip,
            use_ros=args.ros
        )
        
        # Set robot ID if specified
        if args.robot_id != 0:
            navigator.robot_id = args.robot_id
            print("🤖 Robot ArUco ID set to: {}".format(args.robot_id))
        
        # Check if camera initialization was successful
        if not navigator.camera_active:
            print("❌ Failed to initialize camera. Please check your setup.")
            exit(1)
        
        # Run the navigation system
        navigator.run()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print("❌ Error: {}".format(str(e)))
        if ROS_AVAILABLE:
            import traceback
            traceback.print_exc() 