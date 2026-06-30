from collections import deque
import cv2
import imutils
import numpy as np
import argparse
import time

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Rangos HSV de colores
color_ranges = {
    "blue":  {"lower": (90, 120, 0),  "upper": (150, 255, 255), "draw_color": (255, 0, 0)},
    "green": {"lower": (40, 70, 70),  "upper": (80, 255, 255),  "draw_color": (0, 255, 0)},
    "red1":  {"lower": (0, 120, 70),  "upper": (10, 255, 255),  "draw_color": (0, 0, 255)},
    "red2":  {"lower": (170, 120, 70),"upper": (180, 255, 255), "draw_color": (0, 0, 255)}
}

pts = {color: deque(maxlen=args["buffer"]) for color in color_ranges}

# Clasificadores Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Fuente de vídeo
vs = cv2.VideoCapture(0) if not args.get("video", False) else cv2.VideoCapture(args["video"])
time.sleep(2.0)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de colores
    for color_name, color_info in color_ranges.items():
        mask = cv2.inRange(hsv, color_info["lower"], color_info["upper"])

        # Combinar rojo1 y rojo2
        if color_name == "red1":
            red1_mask = mask
            continue
        elif color_name == "red2":
            mask = cv2.bitwise_or(red1_mask, mask)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 10:
                    cv2.circle(frame, (int(x), int(y)), int(radius), color_info["draw_color"], 2)
                    cv2.circle(frame, center, 5, color_info["draw_color"], -1)
                    pts[color_name].appendleft(center)

    # Detección de cara, ojos y boca
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Buscar ojos solo en la mitad superior del rostro
        upper_face_gray = roi_gray[0:int(h/2), :]
        upper_face_color = roi_color[0:int(h/2), :]
        eyes = eye_cascade.detectMultiScale(upper_face_gray, scaleFactor=1.1, minNeighbors=15)

        for (ex, ey, ew, eh) in eyes:
            # Filtra detecciones demasiado bajas
            if ey > int(h * 0.3):  # 30% de la altura de la cara
                continue
            cv2.rectangle(upper_face_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

        # Boca
        mouth_region_gray = roi_gray[int(h/2):, :]
        mouth_region_color = roi_color[int(h/2):, :]
        mouths = mouth_cascade.detectMultiScale(mouth_region_gray, 1.7, 20)
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(mouth_region_color, (mx, my), (mx+mw, my+mh), (255, 0, 255), 2)
            break  # Solo una boca

    cv2.imshow("Detección completa", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()