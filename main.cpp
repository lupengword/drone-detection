//
//  main.cpp
//  SemProj
//
//  Created by Vidit Singh on 04/03/16.
//  Copyright © 2016 Vidit Singh. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/videoio/videoio.hpp>

//Different Background Subtractions
#include <package_bgs/MixtureOfGaussianV2BGS.h>
#include <package_bgs/AdaptiveBackgroundLearning.h>
#include <package_bgs/AdaptiveSelectiveBackgroundLearning.h>
//#include <package_bgs/db/IndependentMultimodalBGS.h>
//#include "package_bgs/lb/LBMixtureOfGaussians.h"
#include "package_bgs/lb/LBAdaptiveSOM.h"
//#include "package_bgs/dp/DPWrenGABGS.h"
//#include "package_bgs/dp/DPEigenbackgroundBGS.h"
//#include "package_bgs/tb/FuzzyChoquetIntegral.h"
//#include "package_bgs/tb/T2FGMM_UM.h"

#include <iostream>
#include <time.h>
#include <algorithm>

#include "preprocess_train.hpp"
#include "run_classifier.hpp"

using namespace cv;
using namespace std;

/*
 * RESIZEF - resize factor for the image indoor scene - 0.3 , outdoor scene - 0.8
 * BORDER  - remove the border pixels
 * AFFINE  - 0 for perpective , 1 for affine
 * BGSL    - 1 if BGSLibrary is to be used
 * BHIS    - No. of frames after which feature points will be recalculated
 * PROFILE - 1 for display of timings
 */



#define RESIZEF			 0.8
#define BORDER			 50
#define AFFINE 			 0
#define BGSL 			 0
#define BGHIS            30
#define PROFILE          1


/*
 * Class for the feature point tracking. The feature points are obtained with goodFeaturesToTrack() function and tracked
 * using calcOpticalFlowPyrLK(). ,
 */

class Tracker {
    Mat             prevGray;
    bool            freshMat;
    vector<Mat>      matVec;
    
public:
    vector<Point2f> trackedFeatures;
    bool            freshStart;
    Mat_<float>     rigidTransform;
    static int      counter ;
    
    Tracker():freshStart(true) {
        freshMat = true;
        rigidTransform = Mat::eye(3,3,CV_32FC1);
    }
    
    void processImage(Mat& img) {
        
        Mat gray,H= Mat::eye(3,3,CV_32FC1); cvtColor(img,gray,CV_BGR2GRAY);
        vector<Point2f> corners;
        
        /*  Re-calculate feature points when below they are below 100. Also
         * they are updated when BGHIS number of frames have been seen.
         */
        
        if(trackedFeatures.size() < 100 || counter > BGHIS) {
            if (trackedFeatures.size() < 100 ) {
                rigidTransform = Mat::eye(3,3,CV_32FC1);
                matVec.clear();
                freshMat = false;
            }
            trackedFeatures.clear();
            goodFeaturesToTrack(gray,corners,300,0.01,10); // Max feature points 300
            cout << "found " << corners.size() << " features\n";
            for (int i = 0; i < corners.size(); ++i) {
                trackedFeatures.push_back(corners[i]);
            }
        }
        
        if (trackedFeatures.size() == 0) {
            cout<< "No features found" << endl;
            cout << "cataclysmic error \n";
            rigidTransform = Mat::eye(3,3,CV_32FC1);
            trackedFeatures.clear();
            matVec.clear();
            freshMat = false;
            prevGray.release();
            freshStart = true;
            return;
        }
        
        // Optical Flow Calculation
        if(!prevGray.empty()) {
            
            vector<uchar> status; vector<float> errors;
            calcOpticalFlowPyrLK(prevGray,gray,trackedFeatures,corners,status,errors,Size(10,10));
            if(countNonZero(status) < status.size() * 0.8) {
                cout << "cataclysmic error \n";
                rigidTransform = Mat::eye(3,3,CV_32FC1);
                trackedFeatures.clear();
                matVec.clear();
                freshMat = false;
                prevGray.release();
                freshStart = true;
                return;
            } else
                freshStart = false;
            
            
            if (counter > BGHIS) {
                if (freshMat)
                    rigidTransform = Mat::eye(3,3,CV_32FC1);
                freshMat = true;
                counter = 0;
            }
            
            
            
#if AFFINE
            Mat_<float> newRigidTransform = estimateRigidTransform(trackedFeatures,corners,false);
            Mat_<float> nrt33 = Mat_<float>::eye(3,3);
            newRigidTransform.copyTo(nrt33.rowRange(0,2));
            rigidTransform *= nrt33;
#else
            
            H = findHomography(trackedFeatures,corners,CV_RANSAC);
            H.convertTo(H, CV_32FC1);
            rigidTransform *= H;
            
#endif
            trackedFeatures.clear();
            /*
             * Remove the feature points which cannot be tracked
             */
            for (int i = 0; i < status.size(); ++i) {
                if(status[i]) {
                    trackedFeatures.push_back(corners[i]);
                }
            }
            
        }
        
        gray.copyTo(prevGray);
        counter++;
    }
};


int Tracker::counter = 0;



void saveImage2(Mat img0, Mat img1,String folder, String method0,String method1){
    
    static int numFrame = 0;
    numFrame++;
    
    stringstream ststream;
    ststream << "/Users/vidit/Desktop/Environment Modelling/Results/"<<folder<<"/"<< method0 << numFrame << ".jpg";
    string filename = ststream.str();
    imwrite(filename,img0);
    
    if(!img1.empty()){
        ststream.str("");
        ststream << "/Users/vidit/Desktop/Environment Modelling/Results/"<<folder<<"/"<< method1 << numFrame << ".jpg";
        filename = ststream.str();
        
        imwrite(filename,img1);
    }
    
}


/*
 * Main Function. Track frames and compensate for motion. Use background substraction to detect moving drones.
 * At the end, morphological filling operation.
 */


int main() {
    
    
#if PROFILE
    clock_t start;
    float ttime    = 0.0;
    float duration = 0.0;
#endif


    VideoCapture vc;
    
    vc.open("/Users/vidit/Desktop/Environment Modelling/videos/experiment_53.avi");
    Mat frame,orig,orig_warped,orig_fp,tmp;
    
    Mat fgMaskMOG2,orgFgMaskMOG2;
    
    String method = "opencvMOG2";
    
    Ptr<BackgroundSubtractor> pMOG2    = createBackgroundSubtractorMOG2(BGHIS);
    Ptr<BackgroundSubtractor> orgMOG2  = createBackgroundSubtractorMOG2(BGHIS);
    
    
    
    
    //   IBGS *bgs = new MixtureOfGaussianV2BGS;
    //   IBGS *bgs = new AdaptiveSelectiveBackgroundLearning;
    IBGS *bgs = new LBAdaptiveSOM;
    //   IBGS *bgs = new DPWrenGABGS;
    //   IBGS *bgs = new DPEigenbackgroundBGS;
    //   IBGS *bgs = new FuzzyChoquetIntegral;
    //   IBGS *bgs = new T2FGMM_UM;
    
    cout << "Frame Rate:" << vc.get(CV_CAP_PROP_FPS);
    
    
    Tracker tracker;
    int count = 0;
    while(vc.isOpened()) {
        
        for( ;count < 10;count++)
            vc >> frame;
        
        
        vc >> frame;
        if(frame.empty()) break;
        frame.copyTo(orig);
        
        
        
    #if PROFILE
        start = clock();
    #endif
        resize(orig,orig,Size(),RESIZEF,RESIZEF,INTER_AREA); // resizing to avoid points on the net
        orig  = orig(Rect(BORDER,0,orig.cols-2*BORDER,orig.rows));    // removing the boundary of the image frame
    #if PROFILE
        duration  = (clock()-start) / (double)CLOCKS_PER_SEC * 1000;
        cout << "PreProcessing step:" << duration << endl;
        ttime += duration;
    #endif
        
        
/*
 * Beginning of Motion Compensation Step
 */
        
    #if PROFILE
        start = clock();
    #endif
        
        tracker.processImage(orig);
    #if PROFILE
        duration  = (clock()-start) / (double)CLOCKS_PER_SEC * 1000;
        cout << "Tracking step:" << duration << endl;
        ttime += duration;
    #endif
        
        
#if AFFINE
        Mat invTrans = tracker.rigidTransform.inv(DECOMP_SVD);
        warpAffine(orig,orig_warped,invTrans.rowRange(0,2),Size());
#else
        Mat invTrans = tracker.rigidTransform.inv(DECOMP_SVD);
        warpPerspective(orig,orig_warped,invTrans,Size());
        
        
        // Uncomment to display feature points
        
        //        for (int i = 0; i < tracker.trackedFeatures.size(); ++i) {
        //            circle(orig,tracker.trackedFeatures[i],3,Scalar(0,0,255),CV_FILLED);
        //        }
        
    #if PROFILE
        start = clock();
    #endif
        
        warpPerspective(orig,orig_fp,invTrans,Size());
        
    #if PROFILE
        duration  = (clock()-start) / (double)CLOCKS_PER_SEC * 1000;
        cout << "Warp step:" << duration << endl;
        ttime += duration;
    #endif
        
        
#endif
        
        
/*
 * End of Motion Compensation Step
 */
   
        
        
#if BGSL
        Mat img_mask ;
        Mat img_model ;
        
    #if PROFILE
        start = clock();
    #endif
        bgs->process(orig_warped, img_mask, img_model);
        
        
    #if PROFILE
        duration  = (clock()-start) / (double)CLOCKS_PER_SEC * 1000;
        cout << "Background Substraction step:" << duration << endl;
        ttime += duration;
        cout << "Total Time:" << ttime << endl;
        ttime = 0.0;
    #endif

        saveImage2(img_mask,Mat(),"BGSCompare/outdoor","LBAdaptiveSOM","");
        
#else
        
        
        if(Tracker::counter == BGHIS){
            pMOG2.release();
            pMOG2    = createBackgroundSubtractorMOG2(BGHIS);
            
            fgMaskMOG2.release();
        }
        
        

    #if PROFILE
        start = clock();
    #endif
        
        pMOG2->apply(orig_warped, fgMaskMOG2);
        
        
    #if PROFILE
        duration  = (clock()-start) / (double)CLOCKS_PER_SEC * 1000;
        cout << "Background Substraction step:" << duration << endl;
        ttime += duration;
    #endif
        
        orgMOG2->apply(orig, orgFgMaskMOG2);
        
        
        
    #if PROFILE
        start = clock();
    #endif
      
/*
 * Set condition to 1 in order to try morphological operation.
 */
#if 0
        Mat elem33 = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
        Mat elem55 = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
        morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_ERODE, elem33);
        morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_DILATE, elem55);
        threshold(fgMaskMOG2, fgMaskMOG2, 50, 1, THRESH_BINARY);
#else
        medianBlur(fgMaskMOG2, fgMaskMOG2, 3);
        threshold(fgMaskMOG2, fgMaskMOG2, 50, 1, THRESH_BINARY);
#endif
        /*
         * Following steps to prepare the frames for the classifier.
         */
        vector<Mat> channels;
        channels.push_back(fgMaskMOG2);
        channels.push_back(fgMaskMOG2);
        channels.push_back(fgMaskMOG2);
        Mat fgMaskMOG23D;
        merge(channels, fgMaskMOG23D);
        
        orig_warped = orig_warped.mul(fgMaskMOG23D);
        
    #if PROFILE
        duration  = (clock()-start) / (double)CLOCKS_PER_SEC * 1000;
        cout << "Post Proceesing step:" << duration << endl;
        ttime += duration;
        cout << "Total Time:" << ttime << endl;
        ttime = 0.0;
    #endif
        
        /* Shows the foreground and warped frame */
        imshow("fg_warped",fgMaskMOG2); // remove thresholding on foreground for correct display
        imshow("orig_warped",orig_warped);
        
        // saveImage2(orig_warped, Mat(), "TPFP/exp8/dilate", "opencvMOG2", "orig");
        
        
#endif
        count++;
        int c = waitKey(1);
        if(c == ' ') {
            if(waitKey()==27) break;
        } else
            if(c == 27) break;
    }
    vc.release();
    delete bgs;
    
}
