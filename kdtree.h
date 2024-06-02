//@author:lenglinxiao
//@data:2024/4/26
//@lenglinxiao@foxmail.com
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
using namespace std;
using namespace cv;
struct LineSegment
 {
    cv::Point2f start;
    cv::Point2f end;
    cv::Point2f getCenter() const;
    LineSegment(cv::Vec4i lines)
    {
        start=cv::Point2f(lines[0],lines[1]);
        end=cv::Point2f(lines[2],lines[3]);
    };
    float distanceTo(const LineSegment& other) const;
    bool operator ==(const LineSegment& other)const
    {
        return (start.x == other.start.x) && (start.y == other.start.y) &&
               (end.x == other.end.x) && (end.y == other.end.y);
    }
 };

 struct KDNode
 {
    LineSegment line;
    KDNode* left;
    KDNode* right;
    KDNode(const LineSegment& line) : line(line), left(nullptr), right(nullptr) {}

 };
 
 class KDtree
 {
    public:
    KDtree():root(nullptr) 
    {
        
    }
    void insert(const LineSegment& line);
    std::vector<LineSegment> rangeSearch(const LineSegment& queryLine, float range);
    private:
    KDNode* root;
    KDNode* insertRecursive(KDNode* node, const LineSegment& line, int depth);
    void rangeSearchRecursive(KDNode* node, const LineSegment& queryLine, float range, int depth,
    std::vector<LineSegment>& results); 

 };