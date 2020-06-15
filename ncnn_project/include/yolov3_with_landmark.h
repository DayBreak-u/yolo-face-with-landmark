#ifndef __OCR_H__
#define __OCR_H__



#include <vector>
#include "net.h"
#include <algorithm>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"    
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <string>
#include <iostream>


using namespace std;


struct Point{
    float _x;
    float _y;
};
struct Bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point point[5];
};

class YOLOFACE
{
    public:
        YOLOFACE();
        void detect(cv::Mat im_bgr,int long_size,   std::vector<Bbox> &prebox);
  


    private:

        ncnn::Net  yolofacenet;
        ncnn::Mat  img;
        int num_thread = 1;
        bool letter_box = true;
        float score_threh = 0.15;
        const float mean_vals[3] = { 0,0,0 };\
        const float norm_vals[3] = { 1.0 / 255.0, 1.0 / 255.0, 1.0 /255.0};
        std::vector<int> minsize0 = {12  , 20 , 32  };
        std::vector<int> minsize1 = {48  , 72 , 128 };
        std::vector<int> minsize2 = {196 , 320, 480 };


};




#endif