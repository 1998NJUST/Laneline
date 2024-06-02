#include<iostream>
#include "depth_chart.h"
using namespace std;
using namespace cv;

int main()
{   
    std::string path ="/home/llx/gao_slam/lane_line/ipm_park";
    image_data d1;
    d1.Select_Type(path);
    d1.load_img(path);   
    return 0;

}