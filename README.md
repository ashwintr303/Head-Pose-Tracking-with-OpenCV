# Head-Pose-Tracking-with-OpenCV
Detect faces, facial landmarks and track head pose from webcam video.

## File Description
1. detect_faces.py : Detect faces (single/multiple) in an image and draw faces using the dlib[1] library.
2. detect_facepoints.py : Detect 68 key facial landmarks in the detected faces.
3. detect_pose.py: Identify the head pose either using the landmarks and mark the direction in which the person is looking at with a vector.
4. video_detect_pose.py: Integrate the above three problems to detect head pose on a video streaming from webcam.

## Dependencies
1. Python   3.7  
2. openCV   3.4 
3. dlib     19.21.0
4. imutils  0.5.3 


To install the complete list of dependencies, run:  
```
pip install -r requirements.txt
```

## Running the files:

The individual files to detect faces, facepoints or pose requires the image path as an argument.
Example usage:  
```
python detect_faces.py PATH_TO_IMAGE
```

The file to run head pose detection on webcam video can be run as any other python file, and does not take any arguments.
Example usage:  
```
python video_detect_pose.py 
```

### Note:
1. Press 'q' to quit the webcam video.
2. The pretrained model used for face landmark detection can be downloaded from [here](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat).
3. In case of multiple faces, head pose is detected only for the primary face (face closest to the center of the screen).

### Example Outputs
| ![single.png](https://github.com/ashwintr303/Head-Pose-Tracking-with-OpenCV/blob/main/images/pose_vector_output1.png)|
|:--:| 
| *Head pose detection with a single face* |
| ![multiple.png](https://github.com/ashwintr303/Head-Pose-Tracking-with-OpenCV/blob/main/images/pose_vector_output2.png)|
| *Head pose detection with multiple faces* |

#### References:
1. [http://dlib.net](http://dlib.net)
2. [https://livecodestream.dev/post/detecting-face-features-with-python/](https://livecodestream.dev/post/detecting-face-features-with-python/)
3. [https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)
