//@author:lenglinxiao
//@data:2024/5/7
//@email:lenglinxiao@foxmail.com
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc/types_c.h>
#include<unistd.h>
#include<cmath>

using namespace std;
using namespace cv;

enum Parking_type
{
  Level_space=0,//水平停车位
  Vertical_space=1,//垂直停车位
  Diagonal_spaces=2,//斜列停车位
};

typedef struct {
  Parking_type parking_type;
  cv::Point2i pos;
}Parking;

enum SpLabels20 {
  ROAD = 0,
  LANE_LINE = 1,//道路线
  PARKING_LINE = 2,//停车线
  PARKING_SLOT = 3,
  ARROW = 4,
  GUIDE_LINE = 5,
  CROSS_WALK_LINE = 6,
  NO_PARKING_SIGN_LINE = 7,
  STOP_LINE = 8,
  SPEED_BUMP = 9,
  OTHER = 10,
  PARKING_LOCK_OPEN = 11,
  PARKING_LOCK_CLOSE = 12,
  TRAFFIC_CONE = 13,
  PARKING_ROD = 14,
  CURB = 15,
  CEMENT_COLUMN = 16,
  IMMOVABLE_OBSTACLE = 17,
  MOVABLE_OBSTACLE = 18,
  BACKGROUND = 19,
  SIDE_WALK = 20,
  PAINTED_WALL_ROOT = 21,
  GENERALIZED_CURB = 22
};

typedef struct finalLine {
  unsigned int id{0};
  cv::Point2i start;
  cv::Point2i end;
  int length{0};
  double computeLineLength() {
    length =
        abs(this->start.x - this->end.x) + abs(this->start.y - this->end.y);
        return length;
  }
} finalLine;


class parking_judgment
{
   public:
   parking_judgment(/*arg*/);
   void preprocess(const cv::Mat&mat1);//预处理函数
   void Determinespacetype(const cv::Mat& preprocessmat,Parking&parking_type);//根据语义图判断停车位类型
   std::vector<finalLine> extractLines(const cv::Mat& image, int thresholdValue);
   double calculateAngleBetweenLines(const finalLine&line1,const finalLine&line2);
   cv::Mat preprocessmat;
   private:
   
   

};