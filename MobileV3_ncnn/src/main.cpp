#include "MobileV3.h"

int main(int argc, char **argv) 
{
    if (argc != 2) 
    {
        fprintf(stderr, "Usage: %s [./Detector imagepath \n", argv[0]);
        return -1;
    }



    const char *imagepath = argv[1];
    cv::Mat im_bgr = cv::imread(imagepath, 1);
    if (im_bgr.empty()) 
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    
    MobileV3 *engine = new MobileV3();


    double time1 = static_cast<double>(cv::getTickCount());

    for(int i=0; i<100; i++)
    {
        engine->detect(im_bgr);
    }

    std::cout << "MobileNet 100 times 时间:" << (static_cast<double>( cv::getTickCount()) - time1) / cv::getTickFrequency() << "s" << std::endl;

    delete engine;
    return 0;
}
