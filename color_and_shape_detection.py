import cv2
import numpy as np

def get_color_name(rgb):
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128)
    }
    min_distance = float("inf")
    color_name = "unknown"
    for name, value in colors.items():
        distance = np.sqrt(np.sum((np.array(value) - rgb) ** 2))
        if distance < min_distance:
            min_distance = distance
            color_name = name
    return color_name

def detect_color(image, bbox):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    avg_color_per_row = np.average(roi, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    color_name = get_color_name(avg_color)
    return color_name

def detect_shape(image, bbox):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape = "unidentified"
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        elif len(approx) > 4:
            # Verificare adițională pentru a diferenția între cerc și alte forme
            area = cv2.contourArea(contour)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            if abs(area - circle_area) / circle_area < 0.2:  
                shape = "circle"
            else:
                shape = "oval"  

    return shape
