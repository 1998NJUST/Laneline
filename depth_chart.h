//@author:lenglinxiao
//@data:2024/3/20
//@email:lenglinxiao@foxmail.com
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <functional>
#include <algorithm> 
#include <regex>
#include <numeric>
#include "parking_judgment.h"
using namespace std;
using namespace cv;
using namespace Eigen;


class image_data
{
    public:
    //定义鸟瞰图内参
    float center_x ;
    float center_y ;
    float ipm_meters_per_pixel;
    float center_offset;
    float ipm_parsing_image_width;
    float ipm_parsing_image_height;
    int frame_id;
    Parking paking_type_1;
    Parking paking_type_2;
    Parking paking_type_3;
    Parking paking_final;
    struct LaneLinePoints
    {
      LaneLinePoints(cv::Point2f p1,cv::Point2f p2)
      {
        line_start_point_vcs=p1;
        line_end_point_vcs=p2;
      }
      cv::Point2f line_start_point_vcs;
      cv::Point2f line_end_point_vcs;  
    };
    //类初始化
    image_data()
        : center_x(224.0f), center_y(256.5f), ipm_meters_per_pixel(0.04f),
          center_offset(1.4f), ipm_parsing_image_width(448.0f), ipm_parsing_image_height(448.0f),frame_id(0) {}
    
    std::vector<LaneLinePoints>find_point(const cv::Mat&mat1);
    void load_img(const string&dir);
    double calculatedistance(const Point&point1,const Point&point2);
    float lineDistance(const cv::Vec4i& line1, const cv::Vec4i& line2);
    std::vector<int> DBSCAN(const std::vector<cv::Vec4i>& lines, float eps, int minPts); 
    void IPMtoVCS(const cv::Point2f&pt_img,cv::Point2f*pt_vcs);
    void VCStoIPM(const cv::Point2f& pt_vcs, cv::Point2f* pt_img);
    void findClosestLines(std::vector<cv::Vec4f>& fitted_lines);
    double calculateLineAngles(const cv::Point2f& point1,const cv::Point2f& point2);
    double calculateMaxXDistance(const cv::Vec4i& line1, const cv::Vec4i& line2);
    float calculate_average_center(const std::vector<cv::Vec4i>& lines);   
    std::vector<int> mergeClusters2(const std::vector<Vec4i>& lines, std::vector<int>& labels, int maxClusters, float maxGapX);
    std::map<int, std::vector<cv::Vec4i>> getClosestClustersToCenter(const std::map<int, 
    std::vector<cv::Vec4i>>& clusters, int imageCenterX);
    std::map<int, std::vector<cv::Vec4i>> filterOutliersInClusters(const std::map<int, std::vector<cv::Vec4i>>& clusters,
                                                                          float maxYDifference, float minLineLength, float minTotalLength);
    void Select_Type(const string&dir);
    
   
    
 };


