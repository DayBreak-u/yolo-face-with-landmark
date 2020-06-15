#include "yolov3_with_landmark.h"



void nms(std::vector<Bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}



YOLOFACE::YOLOFACE()
{
  
    yolofacenet.load_param("../models/mbv3_small_1.param");
    yolofacenet.load_model("../models/mbv3_small_1.bin");

}



cv::Mat resize_img(cv::Mat src,const int long_size)
{
    int w = src.cols;
    int h = src.rows;
    // std::cout<<"原图尺寸 (" << w << ", "<<h<<")"<<std::endl;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)long_size / w;
        w = long_size;
        h = h * scale;
    }
    else
    {
        scale = (float)long_size / h;
        h = long_size;
        w = w * scale;
    }
    if (h % 32 != 0)
    {
        h = (h / 32 + 1) * 32;
    }
    if (w % 32 != 0)
    {
        w = (w / 32 + 1) * 32;
    }
    // std::cout<<"缩放尺寸 (" << w << ", "<<h<<")"<<std::endl;
    cv::Mat result;

    cv::resize(src, result, cv::Size(w, h));
    return result;
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}


void yolodecode(ncnn::Mat pre,std::vector<int> anchor,std::vector<Bbox> & prebox,float confidence_threshold,
        int net_w, int net_h ,int ori_w , int ori_h)
{

    
    int w = pre.w;
    int h = pre.h;

    for (int c = 0 ; c < anchor.size() ; c++){
         float bias = float(anchor[c]);
         float* ptr_x  = pre.channel(c * 16);
         float* ptr_y  = pre.channel(c * 16 + 1 );
         float* ptr_w  = pre.channel(c * 16 + 2 );
         float* ptr_h  = pre.channel(c * 16 + 3 );
         float* ptr_lx1  = pre.channel(c * 16 + 4 );
         float* ptr_ly1  = pre.channel(c * 16 + 5 );
         float* ptr_lx2  = pre.channel(c * 16 + 6 );
         float* ptr_ly2  = pre.channel(c * 16 + 7 );
         float* ptr_lx3  = pre.channel(c * 16 + 8 );
         float* ptr_ly3  = pre.channel(c * 16 + 9 );
         float* ptr_lx4  = pre.channel(c * 16 + 10 );
         float* ptr_ly4  = pre.channel(c * 16 + 11 );
         float* ptr_lx5  = pre.channel(c * 16 + 12 );
         float* ptr_ly5  = pre.channel(c * 16 + 13 );
         float* ptr_box_score  = pre.channel(c * 16 + 14 );
//         float* ptr_class_score  = pre.channel(c * 16 + 15 );
         float stridew = net_w / w;
         float strideh = net_h / h;

         for (int i = 0 ; i < h ; i++){
             for (int j = 0 ; j < w ; j++){
                 // box score
//                float box_score = sigmoid(ptr_box_score[0]);
//                float class_score = sigmoid(ptr_class_score[0]);
//                float confidence = box_score * class_score;
                float confidence = sigmoid(ptr_box_score[0]);

                if (confidence >= confidence_threshold)
                    {
                                            // region box
                        float bbox_cx = (j + sigmoid(ptr_x[0])) / w;
                        float bbox_cy = (i + sigmoid(ptr_y[0])) / h;
                        float bbox_w = static_cast<float>(exp(ptr_w[0]) * bias / net_w);
                        float bbox_h = static_cast<float>(exp(ptr_h[0]) * bias / net_h);

                        float bbox_xmin = bbox_cx - bbox_w * 0.5f;
                        float bbox_ymin = bbox_cy - bbox_h * 0.5f;
                        float bbox_xmax = bbox_cx + bbox_w * 0.5f;
                        float bbox_ymax = bbox_cy + bbox_h * 0.5f;


                        float lx1 = (j + ptr_lx1[0] * bias / stridew ) / w;
                        float ly1 = (i + ptr_ly1[0] * bias / strideh ) / h;
                        float lx2 = (j + ptr_lx2[0] * bias / stridew ) / w;
                        float ly2 = (i + ptr_ly2[0] * bias / strideh ) / h;
                        float lx3 = (j + ptr_lx3[0] * bias / stridew ) / w;
                        float ly3 = (i + ptr_ly3[0] * bias / strideh ) / h;
                        float lx4 = (j + ptr_lx4[0] * bias / stridew ) / w;
                        float ly4 = (i + ptr_ly4[0] * bias / strideh ) / h;
                        float lx5 = (j + ptr_lx5[0] * bias / stridew ) / w;
                        float ly5 = (i + ptr_ly5[0] * bias / strideh ) / h;


                        Bbox temp_box ;
                        temp_box.s = confidence ;
                        temp_box.x1 = bbox_xmin * ori_w ;
                        temp_box.y1 = bbox_ymin * ori_h ;
                        temp_box.x2 = bbox_xmax * ori_w ;
                        temp_box.y2 = bbox_ymax * ori_h ;
                        temp_box.point[0]._x = lx1 * ori_w ;
                        temp_box.point[0]._y = ly1 * ori_h ;
                        temp_box.point[1]._x = lx2 * ori_w ;
                        temp_box.point[1]._y = ly2 * ori_h ;
                        temp_box.point[2]._x = lx3 * ori_w ;
                        temp_box.point[2]._y = ly3 * ori_h ;
                        temp_box.point[3]._x = lx4 * ori_w ;
                        temp_box.point[3]._y = ly4 * ori_h ;
                        temp_box.point[4]._x = lx5 * ori_w ;
                        temp_box.point[4]._y = ly5 * ori_h ;

                        prebox.push_back(temp_box);
            
                    }

                    ptr_x++;
                    ptr_y++;
                    ptr_w++;
                    ptr_h++;
                    ptr_lx1++;
                    ptr_ly1++;
                    ptr_lx2++;
                    ptr_ly2++;
                    ptr_lx3++;
                    ptr_ly3++;
                    ptr_lx4++;
                    ptr_ly4++;
                    ptr_lx5++;
                    ptr_ly5++;

//                    ptr_class_score++;
                    ptr_box_score++;

             }
         }


    }
}



void    YOLOFACE::detect(cv::Mat im_bgr,int long_size,   std::vector<Bbox> &prebox)
{

        // 图像缩放

    int ori_w = im_bgr.cols;
    int ori_h = im_bgr.rows;
    cv::Mat im ;
    if (letter_box)  im = resize_img(im_bgr, long_size);
    else{
         if (long_size % 32 != 0)
        {
            long_size = (long_size / 32 + 1) * 32;
        }
        cv::resize(im_bgr, im, cv::Size(long_size, long_size));
    }




//    float h_scale = im_bgr.rows * 1.0 / im.rows;
//    float w_scale = im_bgr.cols * 1.0 / im.cols;

    ncnn::Mat in = ncnn::Mat::from_pixels(im.data, ncnn::Mat::PIXEL_BGR2RGB, im.cols, im.rows);
    in.substract_mean_normalize(mean_vals,norm_vals);

    // std::cout << "输入尺寸 (" << in.w << ", " << in.h << ")" << std::endl;

    ncnn::Extractor ex = yolofacenet.create_extractor();
    ex.set_num_threads(num_thread);
    ex.input("input0", in);
    ncnn::Mat pred1,pred2,pred3;
    double time1 = static_cast<double>( cv::getTickCount());

    ex.extract("s32", pred3); //
    ex.extract("s16", pred2); //
    ex.extract("s8", pred1); //
    


    yolodecode(pred1,minsize0,prebox,score_threh,in.w , in.h, ori_w, ori_h);
    yolodecode(pred2,minsize1,prebox,score_threh,in.w , in.h, ori_w, ori_h);
    yolodecode(pred3,minsize2,prebox,score_threh,in.w , in.h, ori_w, ori_h);


    nms(prebox,0.35);
////
//    for (int i = 0 ; i < prebox.size() ; i ++)
//    {
//           Bbox temp_box = prebox[i];
//         printf("\n x1 = %f y1 = %f x2 = %f y2 = %f \n",temp_box.x1,temp_box.y1,temp_box.x2,temp_box.y2);
//    }



    std::cout << "前向时间:" << (static_cast<double>( cv::getTickCount()) - time1) / cv::getTickFrequency() << "s" << std::endl;
    // std::cout << "网络输出尺寸 (" << preds.w << ", " << preds.h << ", " << preds.c << ")" << std::endl;


}


