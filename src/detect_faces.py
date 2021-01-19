import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
import math
import argparse

### convert rectangle from dlib to bounding box coordinates for openCV
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

### detect faces and face points
def detect_faces(image):
    ### use dlib face detector to get faces
    detector = dlib.get_frontal_face_detector()
    face_rect = detector(image, 1)

    ### find center of the image
    # print(image.shape)
    height, width = image.shape[0], image.shape[1]
    img_center = (width/2, height/2)

    ### find face closest to center 
    min_dist_to_center = float('inf')
    if len(face_rect) > 0:
        for rect in face_rect:
            dist = math.sqrt((rect.center().x - img_center[0])**2 + (rect.center().y - img_center[1])**2)
            if dist < min_dist_to_center:
                min_dist_to_center = dist
                min_dist_box = rect

        ### draw bounding boxes and face points
        for (i, rect) in enumerate(face_rect):
            (x, y, w, h) = rect_to_bb(rect)
            if rect == min_dist_box:
                ### closest bounding box in green
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                ### other bounding boxes in red
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        ### plot the output
        plt.figure()
        plt.imshow(image)
        plt.savefig('face_detection_output.png')

        return face_rect, min_dist_box
    else:
        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    im = cv2.imread(args.file, cv2.IMREAD_COLOR)
    # im = cv2.imread('test_img2.jpg', cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    boxes, min_dist_box = detect_faces(im)

