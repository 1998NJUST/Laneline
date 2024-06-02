//@author:lenglinxiao
//@data:2024/5/7
//@email:lenglinxiao@foxmail.com
#include "parking_judgment.h"
parking_judgment::parking_judgment(/*arg*/)
{
   preprocessmat=cv::Mat::zeros(448,448,CV_8UC1);
}

void parking_judgment::preprocess(const cv::Mat& mat1)
{
    if(mat1.empty())
    {
        std::cerr<<"Error: image could not be loaded!"<<std::endl;
        return;
    }

    // Check if the semantic elements exist in the image
    bool has_lane_line = false;
    bool has_parking_line = false;

    for(size_t i = 0; i < mat1.rows; i++)
    {
        for(size_t j = 0; j < mat1.cols; j++)
        {
            if(mat1.at<uchar>(i,j) == SpLabels20::LANE_LINE)
            {
                has_lane_line = true;
            }
            else if(mat1.at<uchar>(i,j) == SpLabels20::PARKING_LINE)
            {
                has_parking_line = true;
            }
        }
    }

    // If neither lane line nor parking line exist, handle the situation
    if(!has_lane_line && !has_parking_line)
    {
        std::cerr << "Error: image does not contain lane line or parking line!" << std::endl;
        return;
    }

    // Process the image
    for(size_t i = 0; i < mat1.rows; i++)
    {
        for(size_t j = 0; j < mat1.cols; j++)
        {
            if(mat1.at<uchar>(i,j) == SpLabels20::LANE_LINE)
            {
                preprocessmat.at<uchar>(i,j) = 125;
            }
            else if(mat1.at<uchar>(i,j) == SpLabels20::PARKING_LINE)
            {
                preprocessmat.at<uchar>(i,j) = 255;
            }
            else
            {
                preprocessmat.at<uchar>(i,j) = 0;
            }
        }
    }
}

std::vector<finalLine> parking_judgment::extractLines(const cv::Mat& image, int thresholdValue)
{  
    cv::Mat binaryImage;
    double rho = 1;
    double theta = CV_PI / 180;
    int id=0;
    cv::inRange(image,thresholdValue,thresholdValue,binaryImage);
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(binaryImage,lines,rho,theta,30,10);
    std::vector<finalLine> finallines;
    for(const auto&line:lines)
    {   
        finalLine thisline;
        cv::Point2f p1(line[0],line[1]);
        cv::Point2f p2(line[2],line[3]);
        thisline.start=p1;
        thisline.end=p2;
        thisline.length=thisline.computeLineLength();
        thisline.id=id;
        finallines.push_back(thisline);
        id++;
    }
    return finallines;

}

bool compareline(const finalLine&l1,const finalLine&l2)
{
     return l1.length<l2.length;
}




void parking_judgment::Determinespacetype(const cv::Mat& preprocessmat,Parking&parking_type)
{
    bool has_lane_line = false;
    bool has_parking_line = false;
    for(size_t i = 0; i < preprocessmat.rows; i++)
    {
        for(size_t j = 0; j < preprocessmat.cols; j++)
        {
            if(preprocessmat.at<uchar>(i,j) == 125)
            {
                has_lane_line = true;
            }
            else if(preprocessmat.at<uchar>(i,j) == 255)
            {
                has_parking_line = true;
            }
        }
    }
    if(!has_lane_line && !has_parking_line)
    {
        parking_type.parking_type=Parking_type::Level_space;
        return;
        
    }
    vector<finalLine> lines255=extractLines(preprocessmat,255);
    vector<finalLine> lines125=extractLines(preprocessmat,125);
    auto line1=std::max_element(lines125.begin(),lines125.end(),compareline);
    auto line2=std::max_element(lines255.begin(),lines255.end(),compareline);
    double angle;
    finalLine line3=*line1;
    finalLine line4=*line2;
    angle=calculateAngleBetweenLines(line3,line4);
    if(angle>80&&angle<100)
    {
      parking_type.parking_type=Parking_type::Vertical_space;
    }else if(angle<10||angle>170)
    {
      parking_type.parking_type=Parking_type::Level_space;
    }else if(angle>30&&angle<60)
    {
      parking_type.parking_type=Parking_type::Diagonal_spaces;
    }

}

double parking_judgment::calculateAngleBetweenLines(const finalLine& line1, const finalLine& line2) {
    // Calculate vectors
    cv::Point2f vector1(line1.end.x - line1.start.x, line1.end.y - line1.start.y);
    cv::Point2f vector2(line2.end.x - line2.start.x, line2.end.y - line2.start.y);

    // Calculate magnitudes
    double magnitude1 = cv::norm(vector1);
    double magnitude2 = cv::norm(vector2);

    // Check if magnitudes are not zero
    if (magnitude1 < std::numeric_limits<double>::epsilon() || magnitude2 < std::numeric_limits<double>::epsilon()) {
        std::cerr << "Error: Magnitude of one of the vectors is close to zero!" << std::endl;
        return 0; // You can choose to return a default value or handle this case differently
    }

    // Calculate dot product
    double dotProduct = vector1.x * vector2.x + vector1.y * vector2.y;

    // Calculate cross product
    double crossProduct = vector1.x * vector2.y - vector1.y * vector2.x;

    // Calculate angle in radians
    double angle = atan2(crossProduct, dotProduct);

    // Convert angle to degrees
    angle *= 180.0 / CV_PI;

    return angle;
}




