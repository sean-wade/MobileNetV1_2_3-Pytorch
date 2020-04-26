#include "MobileV3.h"


MobileV3::MobileV3()
{
    MobileNet.load_param("/home/robot/code/MobileV3_ncnn/models/MobineV3_Large.param");
    MobileNet.load_model("/home/robot/code/MobileV3_ncnn/models/MobileV3_Large.bin");
}


void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}


void MobileV3::detect(cv::Mat im)
{
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(im.data, ncnn::Mat::PIXEL_BGR2RGB, im.cols, im.rows, 224, 224);
    in.substract_mean_normalize(mean_vals, norm_vals);

    std::cout << "输入尺寸 (" << in.w << ", " << in.h << ")" << std::endl;

    ncnn::Extractor ex = MobileNet.create_extractor();
    ex.set_num_threads(num_thread);
    ex.input("input", in);
    ncnn::Mat preds;
    ex.extract("output", preds);
    std::cout << "网络输出尺寸 (" << preds.w << ", " << preds.h << ", " << preds.c << ")" << std::endl;
    pretty_print(preds);
}

 
//torch:   [[-4.5641,    1.2246,    0.7689,   -4.7141,   -9.3101,   -2.8519,   12.0943     ]]
//onnx:    [[-4.5987167  1.2692472  0.7589344 -4.751481  -9.331574  -2.81181   12.08503    ]]
//ncnn:    [[-4.598713   1.269243   0.758936  -4.751481  -9.331571  -2.811809  12.085027   ]]




