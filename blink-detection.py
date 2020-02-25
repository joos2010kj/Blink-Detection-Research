import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, face)

        left_eye_coordinates_x = []
        left_eye_coordinates_y = []
        right_eye_coordinates_x = []
        right_eye_coordinates_y = []

        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        #     if n >= 36 and n < 42:
        #         left_eye_coordinates_x.append(x)
        #         left_eye_coordinates_y.append(y)
        #     else:
        #         right_eye_coordinates_x.append(x)
        #         right_eye_coordinates_y.append(y)

        # cv2.rectangle(frame, (min(left_eye_coordinates_x), min(left_eye_coordinates_y)), (max(left_eye_coordinates_x), max(left_eye_coordinates_y)), (0, 255, 0), 3)
        # cv2.rectangle(frame, (min(right_eye_coordinates_x), min(right_eye_coordinates_y)), (max(right_eye_coordinates_x), max(right_eye_coordinates_y)), (0, 255, 0), 3)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break
