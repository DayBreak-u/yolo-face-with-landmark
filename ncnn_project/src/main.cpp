#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "yolov3_with_landmark.h"

using namespace std;

int main(int argc, char** argv)
{
     if (argc != 3) {
        fprintf(stderr, "Usage: %s [imagepath max_side] \n", argv[0]);
        return -1;
    }

    const char *imgPath = argv[1];
    const int max_side = atoi(argv[2]);


    YOLOFACE *detector = new YOLOFACE();

    for	(int i = 0; i < 1; i++){


        cv::Mat img = cv::imread(imgPath);

        if (img.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imgPath);
            return -1;
        }

 
        std::vector<Bbox> boxes;
        detector->detect(img, max_side, boxes);
    
//         printf("%d \n", boxes.size());
//         draw image
         for (int j = 0; j < boxes.size(); ++j) {
             cv::Rect rect(boxes[j].x1, boxes[j].y1, boxes[j].x2 - boxes[j].x1, boxes[j].y2 - boxes[j].y1);
             cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
             char test[80];
             sprintf(test, "%.2f", boxes[j].s * 100 );

             cv::putText(img, test, cv::Size((boxes[j].x1), boxes[j].y1), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
             cv::circle(img, cv::Point(boxes[j].point[0]._x , boxes[j].point[0]._y ), 1, cv::Scalar(0, 0, 225), 4);
             cv::circle(img, cv::Point(boxes[j].point[1]._x , boxes[j].point[1]._y ), 1, cv::Scalar(0, 255, 225), 4);
             cv::circle(img, cv::Point(boxes[j].point[2]._x , boxes[j].point[2]._y ), 1, cv::Scalar(255, 0, 225), 4);
             cv::circle(img, cv::Point(boxes[j].point[3]._x , boxes[j].point[3]._y ), 1, cv::Scalar(0, 255, 0), 4);
             cv::circle(img, cv::Point(boxes[j].point[4]._x , boxes[j].point[4]._y ), 1, cv::Scalar(255, 0, 0), 4);
         }
         cv::imwrite("test.png", img);
    }
    delete detector;

    return 0;
}