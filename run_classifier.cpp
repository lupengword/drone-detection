//
//  run_classifier.cpp
//  SemProj
//
//  Created by Vidit Singh on 10/05/16.
//  Copyright © 2016 Vidit Singh. All rights reserved.
//

#include "run_classifier.hpp"
#include <iostream>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
/*
 * Detect drones using trained cascade classifier. Video input is background subtracted.
 */

void run_classifier(string model, string video){
    
    
    cv::VideoCapture cap(video);
    if(!cap.isOpened()) {
        
        std::cout << "Error reading video file" << std::endl;
        return ;
    }
    
    
    // classifier object
    cv::CascadeClassifier drone_cascade;
    // load the cascade
    if (!drone_cascade.load(model)) {
        std::cout << "Error : Unable to load cascade detector" << std::endl;
        return ;
    }

    
    int counter = 0;
    
    cv::Mat image;
    // Read until end of file.
    for(; ;)
    {

        cap >> image;
        if(! image.data ) {
            printf("Video is over\n");
            return ;
        }
        
        if(image.rows > 400)
            cv::resize(image, image, cv::Size(), 0.8, 0.8);
        
        cv::Mat image_gray;
        cv::cvtColor(image, image_gray, CV_BGR2GRAY);
        
        std::vector<cv::Rect> detections;
        std::vector<int> reject_levels;
        std::vector<double> level_weights;

        // detect the drones
        drone_cascade.detectMultiScale(image_gray, detections, 3.1f, 20, 0, cv::Size(), cv::Size());

        
        // place the detection on the frames
        for (int n = 0; n < detections.size(); n++) {

            cv::rectangle(image, detections[n], cv::Scalar(255,255,0), 1.5);
            cv::putText(image, std::string(), //std::to_string(reject_levels[n]),
                        cv::Point(detections[n].x, detections[n].y), 1, 1, cv::Scalar(0,0,255));
        }
        
        cv::imshow("detections", image);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q')
        {
            break;
        }
        counter++;
    }
    
    
}

// Testing the trained classifier
int main(){

    string modelFile = "/Users/vidit/Desktop/Environment Modelling/detection/outdoor/classifier/cascade.xml";
    string testFile  = "/Users/vidit/Desktop/Environment Modelling/videos/experiment_53.avi";
    run_classifier(modelFile, testFile);
    
}