import cv2

# Ruta de la imagen
image_path = 'pruebas/a_12.jpeg'

# Tamaño del patrón de esquinas internas (7x7 para un tablero de 8x8 cuadrados)
pattern_size = (7, 7)

# Leer imagen
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en: {image_path}")

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Buscar esquinas
ret, corners = cv2.findChessboardCorners(gray, pattern_size)

# Dibujar esquinas si se detectan
if ret:
    print("✔ Tablero detectado correctamente.")
    cv2.drawChessboardCorners(image, pattern_size, corners, ret)
    output_path = 'ajedrez_corners.png'
    cv2.imwrite(output_path, image)
    print(f"✔ Imagen guardada como: {output_path}")
else:
    print("❌ No se detectó el tablero. Verifica iluminación, ángulo y patrón (7x7 esquinas internas).")
