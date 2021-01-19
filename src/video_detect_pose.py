from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
import math

### convert rectangle from dlib to bounding box coordinates for openCV
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

### get coordinates for 68 face points
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

### initialize dlib's face detector 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

### initialize the video stream and sleep for a bit, allowing the
### camera sensor to warm up
vs = VideoStream(src=0, framerate=30).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

frame_width = 1024
frame_height = 576
fps = 30

### 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip 34
                            (0.0, -330.0, -65.0),        # Chin 9
                            (-225.0, 170.0, -135.0),     # Left eye left corner 37
                            (225.0, 170.0, -135.0),      # Right eye right corne 46
                            (-150.0, -150.0, -125.0),    # Left Mouth corner 49
                            (150.0, -150.0, -125.0)      # Right mouth corner 55

                        ])

while True:
    time.sleep(0.1)
    ### grab the frame
    frame = vs.read()
    try:
        frame = imutils.resize(frame, width=1024, height=576)
        size = frame.shape
        frame_center = (size[1]/2, size[0]/2/2)
        min_dist_to_center = float('inf')

        ### detect faces 
        rects = detector(frame, 0)

        ### find face closest to center
        if len(rects) > 0:
            for rect in rects:
                dist = math.sqrt((rect.center().x - frame_center[0])**2 + (rect.center().y - frame_center[1])**2)
                if dist < min_dist_to_center:
                    min_dist_to_center = dist
                    min_dist_box = rect

        for rect in rects:
            (x, y, w, h) = rect_to_bb(rect)
            if rect == min_dist_box:
                ### closest bounding box in green
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                ### other bounding boxes in red
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            shape = predictor(frame, rect)
            shape = shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            if rect == min_dist_box:
                    points = shape

            ### model points from detector
            image_points = np.array([
                                        points[30],     # Nose tip
                                        points[8],      # Chin
                                        points[36],     # Left eye left corner
                                        points[45],     # Right eye right corne
                                        points[48],     # Left Mouth corner
                                        points[54]      # Right mouth corner
                                    ], dtype="double")

            ### camera internals
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array([[focal_length,0,center[0]],[0, focal_length, center[1]],[0,0,1]], dtype="double")

            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)#flags=cv2.CV_ITERATIVE)
            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            ### draw line from nose tip
            cv2.line(frame, p1, p2, (255,0,0), 2)
        
        ### show the frame
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

        ### Press 'q' on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
    except:
        continue

cv2.destroyAllWindows()
vs.stop()