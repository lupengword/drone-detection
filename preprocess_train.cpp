//
//  preprocess_train.cpp
//  SemProj
//
//  Created by Vidit Singh on 09/05/16.
//  Copyright © 2016 Vidit Singh. All rights reserved.
//

#include "preprocess_train.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>




#include <fstream>
#include <vector>

using namespace std;
typedef vector<pair<pair<int, int>,pair<int, int>> > boundingBoxVec;

/*  Prepare data for cascade classifier training
 *  Directory structure required
 *  .
 *      /pos_img/img*.jpg
 *      /neg_img/img*.jpg
 */

/*
 *  Output in same directory
 *  .
 *      info.dat
 *      bg.txt
 */

void prepareImageSamples(string filename,string video){
    vector<int> negSample;
    vector<pair<int,boundingBoxVec>>posSample;
    ifstream annotationFile(filename);
    string line;
    string delimiter1 = ": ";
    string delimiter2 = " ";
    string delimiter3 = "(",delimiter4 = ")";
    string delimiter5 = ",";
    
    string parentDir   = "/Users/vidit/Desktop/Environment Modelling/detection/outdoor/";
    string extn        = video;
    extn = extn.erase(0,extn.find("_"));
    extn = extn.substr(0,extn.find("."));
    string posDir      = "pos_img" + extn + "/";
    string negDir      = "neg_img" + extn + "/";
    string posImageLoc = parentDir + posDir;
    string negImageLoc = parentDir + negDir;;
    string infoLoc     = parentDir + "info" + extn + ".dat";
    string negInfoLoc  = parentDir + "bg"   + extn + ".txt";
    
    while (getline(annotationFile, line)) {
        
        line.erase(0,line.find(delimiter1)+delimiter1.length());
        string frameNumber = line.substr(0,line.find(delimiter2));
        line.erase(0,line.find(delimiter2)+delimiter2.length());
        size_t pos = line.find(delimiter2);
        line.erase(0,pos+delimiter2.length());
        if(( pos = line.find(delimiter2)) == string::npos)
            negSample.push_back(stoi(frameNumber));
        else{
            boundingBoxVec bBoxVec;
            while ((pos = line.find(delimiter3)) != string::npos) {
                line.erase(0,pos+delimiter3.length());
                int rectX1 = stoi(line.substr(0,line.find(delimiter5)));
                line.erase(0,line.find(delimiter5)+delimiter5.length());
                
                int rectY1 = stoi(line.substr(0,line.find(delimiter5)));
                line.erase(0,line.find(delimiter5)+delimiter5.length());
                
                int rectX2 = stoi(line.substr(0,line.find(delimiter5)));
                line.erase(0,line.find(delimiter5)+delimiter5.length());
                
                int rectY2 = stoi(line.substr(0,line.find(delimiter4)));
                line.erase(0,line.find(delimiter4)+delimiter4.length());
                
                
                bBoxVec.push_back(make_pair(make_pair(rectX1,rectY1),make_pair(abs(rectX1-rectX2),abs(rectY1-rectY2))));
                
                
            }
            posSample.push_back(make_pair(stoi(frameNumber),bBoxVec));
        }
        
        
    }
    
    cv::VideoCapture vc;
    vc.open(video);
    
    cv::Mat frame;
    int numFrame = 1;
    
    ofstream info,negInfo;
    info.open(infoLoc);
    negInfo.open(negInfoLoc);
   
    while(vc.isOpened() && (negSample.size() != 0 || posSample.size() != 0)) {
        
        vc >> frame;
        stringstream ststream;
        if (numFrame == negSample[0]){
            ststream << negImageLoc << "img_" << numFrame << ".jpg";
            cv::imwrite(ststream.str(),frame);
            negSample.erase(negSample.begin());
            negInfo << negDir << "img_" << numFrame << ".jpg" <<"\n";

        }
        else{
            ststream << posImageLoc << "img_" << numFrame << ".jpg";
            cv::imwrite(ststream.str(),frame);
            boundingBoxVec bBoxVec;
            bBoxVec = posSample[0].second;
            stringstream boundStream;
            for (int i = 0 ; i<bBoxVec.size(); i++) {
                boundStream << " " << bBoxVec[i].first.second << " " << bBoxVec[i].first.first << " " << bBoxVec[i].second.first << " " << bBoxVec[i].second.second;
            }
            
            info << posDir << "img_" << numFrame << ".jpg" << " " << bBoxVec.size()  << boundStream.str() << "\n";
            posSample.erase(posSample.begin());
        }
        numFrame ++;
      
    }
    
    info.close();
    negInfo.close();

    
    
}

// Creating the dat file for training cascade classifier
int main(){
    string annotationFile = "/Users/vidit/Desktop/Environment Modelling/envmod/detection/video_annotations/Video_53.txt";
    string videoFile = "/Users/vidit/Desktop/Environment Modelling/videos/experiment_53.avi";
    prepareImageSamples(annotationFile,videoFile);

}
