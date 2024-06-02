//@author:lenglinxiao
//@data:2024/4/26
//@lenglinxiao@foxmail.com
#include "kdtree.h"

cv::Point2f LineSegment::getCenter() const {
        return cv::Point2f((start.x + end.x) / 2.0, (start.y + end.y) / 2.0);
    }

float LineSegment::distanceTo(const LineSegment& other) const {
    float distanceStartStart = std::sqrt(std::pow(start.x - other.start.x, 2) + std::pow(start.y - other.start.y, 2));
    float distanceStartEnd = std::sqrt(std::pow(start.x - other.end.x, 2) + std::pow(start.y - other.end.y, 2));
    float distanceEndStart = std::sqrt(std::pow(end.x - other.start.x, 2) + std::pow(end.y - other.start.y, 2));
    float distanceEndEnd = std::sqrt(std::pow(end.x - other.end.x, 2) + std::pow(end.y - other.end.y, 2));
    float minDistanceEndpoints = std::min({distanceStartStart, distanceStartEnd, distanceEndStart, distanceEndEnd});
    float distanceCenter = std::sqrt(std::pow(getCenter().x - other.getCenter().x, 2) + std::pow(getCenter().y - other.getCenter().y, 2));
    return std::min(minDistanceEndpoints, distanceCenter);
    }

void KDtree::insert(const LineSegment& line) {
        root = insertRecursive(root, line, 0);
    }

KDNode* KDtree::insertRecursive(KDNode* node, const LineSegment& line, int depth)
{
    if (node == nullptr) 
    {
        return new KDNode(line);
    }
    int dim = depth % 2;
    float lineCoord = (dim == 0) ? line.start.x : line.start.y;
    float nodeCoord = (dim == 0) ? node->line.start.x : node->line.start.y;
    if (lineCoord < nodeCoord)
    {
        node->left = insertRecursive(node->left, line, depth + 1);
    } else{
        node->right = insertRecursive(node->right, line, depth + 1);
    }
    return node;
}

std::vector<LineSegment> KDtree::rangeSearch(const LineSegment& queryLine, float range)
{
    std::vector<LineSegment> results;
    rangeSearchRecursive(root, queryLine, range,0,results);
    return results;
}


void KDtree::rangeSearchRecursive(KDNode* node, const LineSegment& queryLine, float range,int depth, 
std::vector<LineSegment>& results)
{
  if (node == nullptr) 
  {
    return;
  }
  if (node->line.distanceTo(queryLine) < range) 
  {
    results.push_back(node->line);
  }
  int dim = depth % 2;
  float diff = (dim == 0 ? queryLine.getCenter().x - node->line.getCenter().x : queryLine.getCenter().y - node->line.getCenter().y);
  if (diff < 0) 
  {
    rangeSearchRecursive(node->left, queryLine, range, depth + 1, results);
    if (std::abs(diff) < range) 
    {
        rangeSearchRecursive(node->right, queryLine, range, depth + 1, results);
    }
    } else 
    {
        rangeSearchRecursive(node->right, queryLine, range, depth + 1, results);
    if (std::abs(diff) < range)
    {
        rangeSearchRecursive(node->left, queryLine, range, depth + 1, results);
    }
    }

} 