# Background Subtraction based Drone detections.

## Steps to use.

1. main.cpp - takes video input and using background subtraction gets candidate drone region. Background Subtraction Library can be found here - 
https://github.com/andrewssobral/bgslibrary  . Follow the instructions to install it.
2. preprocess_train.cpp - to prepare data to train cascade classifier. OpenCV Cascade Training followed using
```
opencv_traincascade
```
Followed instruction from https://docs.opencv.org/2.4.13.2/doc/user_guide/ug_traincascade.html
3.run_classifier.cpp - detect the drones on frames after background subtraction. Cascade Classifier trained by preprocess_train.cpp.
