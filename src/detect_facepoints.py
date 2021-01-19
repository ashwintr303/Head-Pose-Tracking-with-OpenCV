import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import argparse

from detect_faces import detect_faces

### get coordinates for 68 face points
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

### run face landmark detection algorithm
def detect_facepoints(image):
    try:
        face_rect, min_dist_box = detect_faces(image)
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        for (i, rect) in enumerate(face_rect):
            shape = predictor(image, rect)
            shape = shape_to_np(shape)
            if rect == min_dist_box:
                points_to_return = shape
            ### draw face landmarks
            for (x, y) in shape:
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        plt.figure()
        plt.imshow(image)
        plt.savefig('facepoints_output.png')
        return points_to_return
    except:
        return


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    im = cv2.imread(args.file, cv2.IMREAD_COLOR)
    # im = cv2.imread('test_img2.jpg', cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    points = detect_facepoints(im)

