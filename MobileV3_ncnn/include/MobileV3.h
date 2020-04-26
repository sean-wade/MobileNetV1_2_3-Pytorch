#ifndef __MOBILEV3_H__
#define __MOBILEV3_H__


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

class MobileV3
{
    public:
        MobileV3();
        void detect(cv::Mat im_bgr);


    private:
        ncnn::Net MobileNet;
        ncnn::Mat img;
        int num_thread = 4;
        int MobileV3_w = 224;
        int MobileV3_h = 224;

        const float mean_vals[3] = { 127.f, 127.f, 127.f };
        const float norm_vals[3] = { 1.0 / 128, 1.0 / 128, 1.0 / 128 };

};




#endif
