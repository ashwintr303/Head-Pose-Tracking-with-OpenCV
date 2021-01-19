import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

from detect_facepoints import detect_facepoints

def detect_pose(img):
    points = detect_facepoints(img)
    ### 3D model points
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

    ### model points from detector
    image_points = np.array([
                                    points[30],     # Nose tip
                                    points[8],      # Chin
                                    points[36],     # Left eye left corner
                                    points[45],     # Right eye right corne
                                    points[48],     # Left Mouth corner
                                    points[54]      # Right mouth corner
                                ], dtype="double")

    size = img.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                        )
    dist_coeffs = np.zeros((4,1)) 
    ### solve iteratively 
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, 
                                                        camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    ### draw line from nose tip
    cv2.line(img, p1, p2, (0, 255, 255), 2)    
    
    plt.figure()
    plt.imshow(im)
    plt.savefig('pose_vector_output.png')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    im = cv2.imread(args.file, cv2.IMREAD_COLOR)
    # im = cv2.imread('test_img2.jpg', cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    detect_pose(im)


    

