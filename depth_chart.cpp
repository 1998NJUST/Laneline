//@author:lenglinxiao
//@data:2024/3/20
//@email:lenglinxiao@foxmail.com
#include "depth_chart.h"

std::ostream &operator<<(std::ostream &os, const cv::Point &point)
{
    os << "(" << point.x << ", " << point.y << ")";
    return os;
}

CvPoint operator-(CvPoint p1, CvPoint p2)
{
    CvPoint p3;
    p3.x = p1.x - p2.x;
    p3.y = p1.y - p2.y;
    return p3;
}

bool operator==(Parking p1,Parking p2)
{
    return p1.parking_type==p2.parking_type;
}

float image_data::lineDistance(const cv::Vec4i &line1, const cv::Vec4i &line2)
{
    cv::Point pt1(line1[0], line1[1]), pt2(line1[2], line1[3]),
        pt3(line2[0], line2[1]), pt4(line2[2], line2[3]),
        pt5((pt1 + pt2) / 2), pt6((pt3 + pt4) / 2);

    // 计算所有可能的点对组合中的最小距离
    float minDist = std::min({calculatedistance(pt1, pt3), calculatedistance(pt1, pt4),
                              calculatedistance(pt2, pt3), calculatedistance(pt2, pt4), calculatedistance(pt5, pt6)});
    return minDist;
}

std::vector<int> image_data::DBSCAN(const std::vector<cv::Vec4i> &lines, float eps, int minPts)
{
    std::vector<int> labels(lines.size(), -1); // 初始化所有线段的标签为 -1（未分类）
    int clusterId = 0;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        if (labels[i] != -1)
            continue; // 如果线段已经被分类，则跳过

        // 寻找线段的邻域
        std::vector<int> neighbors;
        for (size_t j = 0; j < lines.size(); ++j)
        {
            if (lineDistance(lines[i], lines[j]) < eps)
            {
                neighbors.push_back(j);
            }
        }

        if (neighbors.size() < minPts)
        {
            labels[i] = 0; // 标记为噪声
            continue;
        }

        // 创建新的聚类
        clusterId++;
        labels[i] = clusterId;

        // 为当前线段的所有邻域线段分配聚类标签
        for (size_t k = 0; k < neighbors.size(); ++k)
        {
            int neighborIdx = neighbors[k];
            if (labels[neighborIdx] == -1)
            {
                labels[neighborIdx] = clusterId;
                // 将邻域线段的邻域也加入到当前聚类中   DBSCAN的第二个思想:领域集合思想
                for (size_t l = 0; l < lines.size(); ++l)
                {
                    if (lineDistance(lines[neighborIdx], lines[l]) < eps && std::find(neighbors.begin(), neighbors.end(), l) == neighbors.end())
                    {
                        neighbors.push_back(l);
                    }
                }
            }
        }
    }
    return labels;
}

// 计算点距离函数
double image_data::calculatedistance(const Point &point1, const Point &point2)
{
    double dist;
    dist = sqrt(pow((point1.x - point2.x), 2) + pow((point1.y - point2.y), 2));
    return dist;
}

// IPM图坐标系转到VCS坐标系
void image_data::IPMtoVCS(const cv::Point2f &pt_img, cv::Point2f *pt_vcs)
{
    (*pt_vcs).x = (ipm_parsing_image_height / 2 - pt_img.y) * ipm_meters_per_pixel + center_offset;
    (*pt_vcs).y = (ipm_parsing_image_width / 2 - pt_img.x) * ipm_meters_per_pixel;
}
// VCS坐标系转回到IPM坐标系
void image_data::VCStoIPM(const cv::Point2f &pt_vcs, cv::Point2f *pt_img)
{
    (*pt_img).y = ipm_parsing_image_height / 2 - (pt_vcs.x - center_offset) / ipm_meters_per_pixel;
    (*pt_img).x = ipm_parsing_image_width / 2 - pt_vcs.y / ipm_meters_per_pixel;
}

// 角度计算函数
double image_data::calculateLineAngles(const cv::Point2f &point1, const cv::Point2f &point2)
{
    double angle = std::atan2(point2.y - point1.y, point2.x - point1.x) * 180.0 / CV_PI;
    if (angle < 0)
    {
        angle += 180;
    }
    return angle;
}

// 计算X坐标最大值
double image_data::calculateMaxXDistance(const cv::Vec4i &line1, const cv::Vec4i &line2)
{
    int minX1 = std::min(line1[1], line1[3]);
    int maxX1 = std::max(line1[1], line1[3]);
    int minX2 = std::min(line2[1], line2[3]);
    int maxX2 = std::max(line2[1], line2[3]);
    int maxDistance = std::max(std::abs(minX1 - minX2), std::abs(maxX1 - maxX2));
    return maxDistance;
}

// 根据X重新聚类函数
std::vector<int> image_data::mergeClusters2(const std::vector<Vec4i> &lines, std::vector<int> &labels, int maxClusters, float maxGapX)
{
    std::map<int, std::vector<cv::Vec4i>> clusters;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        if (labels[i] > 0)
        {
            clusters[labels[i]].push_back(lines[i]);
        }
    }

    // 强制合并X坐标差值小于maxGapX的聚类
    bool mergeOccurred;
    do
    {
        mergeOccurred = false;
        for (auto jt1 = clusters.begin(); jt1 != clusters.end() && !mergeOccurred; ++jt1)
        {
            for (auto jt2 = std::next(jt1); jt2 != clusters.end() && !mergeOccurred; ++jt2)
            {
                float average_center1 = calculate_average_center(jt1->second);
                float average_center2 = calculate_average_center(jt2->second);
                if (std::abs(average_center1 - average_center2) < maxGapX)
                {
                    // 更新标签
                    int mergeId1 = jt1->first;
                    int mergeId2 = jt2->first;
                    for (auto &label : labels)
                    {
                        if (label == mergeId2)
                            label = mergeId1;
                    }
                    // 合并聚类
                    clusters[mergeId1].insert(clusters[mergeId1].end(), clusters[mergeId2].begin(), clusters[mergeId2].end());
                    clusters.erase(mergeId2);
                    mergeOccurred = true;
                }
            }
        }
    } while (mergeOccurred);

    // 检查并合并聚类直到达到最大聚类数
    while (clusters.size() > maxClusters)
    {
        auto it1 = clusters.begin();
        auto it2 = std::next(it1);
        float minDistance = std::numeric_limits<float>::max();
        int mergeId1 = it1->first, mergeId2 = it2->first;
        for (auto jt1 = clusters.begin(); jt1 != clusters.end(); ++jt1)
        {
            for (auto jt2 = std::next(jt1); jt2 != clusters.end(); ++jt2)
            {
                float average_center1 = calculate_average_center(jt1->second);
                float average_center2 = calculate_average_center(jt2->second);
                float dist = std::abs(average_center1 - average_center2);
                if (dist < minDistance)
                {
                    minDistance = dist;
                    mergeId1 = jt1->first;
                    mergeId2 = jt2->first;
                }
            }
        }
        if (mergeId1 != -1 && mergeId2 != -1)
        {
            for (auto &label : labels)
            {
                if (label == mergeId2)
                    label = mergeId1;
            }
            clusters[mergeId1].insert(clusters[mergeId1].end(), clusters[mergeId2].begin(), clusters[mergeId2].end());
            clusters.erase(mergeId2);
        }
        else
        {
            break;
        }
    }
    return labels;
}



// 返回一个包含两个最近聚类的map
std::map<int, std::vector<cv::Vec4i>> image_data::getClosestClustersToCenter(const std::map<int,std::vector<cv::Vec4i>> &clusters,
                                                                              int imageCenter)
{
    std::pair<int, int> closestClusterIds(-1, -1);
    std::pair<float, float> closestDistances(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    for (const auto &cluster : clusters)
    {
        float clusterCenterX = calculate_average_center(cluster.second);
        float distanceToCenter = std::abs(clusterCenterX - imageCenter);
        if (distanceToCenter < closestDistances.first)
        {
            closestDistances.second = closestDistances.first;
            closestClusterIds.second = closestClusterIds.first;
            closestDistances.first = distanceToCenter;
            closestClusterIds.first = cluster.first;
        }
        else if (distanceToCenter < closestDistances.second)
        {
            closestDistances.second = distanceToCenter;
            closestClusterIds.second = cluster.first;
        }
    }
    std::map<int, std::vector<cv::Vec4i>> closestClusters;
    if (closestClusterIds.first != -1)
    {
        closestClusters[closestClusterIds.first] = clusters.at(closestClusterIds.first);
    }
    if (closestClusterIds.second != -1 && closestClusterIds.first != closestClusterIds.second)
    {
        closestClusters[closestClusterIds.second] = clusters.at(closestClusterIds.second);
    }
    return closestClusters;
}

//过滤掉结果中的孤立短线段
std::map<int, std::vector<cv::Vec4i>> image_data::filterOutliersInClusters(const std::map<int, std::vector<cv::Vec4i>>& clusters,
                                                                          float maxYDifference, float minLineLength, float minTotalLength)
{
    std::map<int, std::vector<cv::Vec4i>> filtered_clusters;
    for (const auto& cluster_pair : clusters)
    {
        int cluster_id = cluster_pair.first;
        const auto& lines = cluster_pair.second;

        //计算总共的聚类长度
        float total_cluster_length = 0.0;
        for (const auto& line : lines) {
            float line_length = cv::norm(cv::Point(line[0], line[1]) - cv::Point(line[2], line[3]));
            total_cluster_length += line_length;
        }

        // 如果这个聚类总长度本来就很短，直接不要了
        if (total_cluster_length < minTotalLength) {
            continue;
        }

        // 如果聚类只有一条线，而且很短，也不要了
        if (lines.size() == 1) {
            float line_length = cv::norm(cv::Point(lines[0][0], lines[0][1]) - cv::Point(lines[0][2], lines[0][3]));
            if (line_length >= minLineLength) {
                filtered_clusters[cluster_id].push_back(lines[0]);
            }
            continue;
        }

        
        for (const auto& line : lines) {
            float line_length = cv::norm(cv::Point(line[0], line[1]) - cv::Point(line[2], line[3]));
            float min_diff = std::numeric_limits<float>::max();

            // 寻找每个类中y差值的最小值
            for (const auto& other_line : lines) {
                if (line == other_line) continue; // 跳过相同的线
                float y_diff = std::abs((line[1] + line[3]) / 2.0 - (other_line[1] + other_line[3]) / 2.0);
                float x_diff = std::abs((line[0]+line[2]))  /2.0 -  (other_line[0] + other_line[2]) / 2.0 ;
               if(paking_final.parking_type==Parking_type::Level_space)
               {
                if (y_diff < min_diff) {
                    min_diff = y_diff;
                }
               }else if(paking_final.parking_type==Parking_type::Vertical_space)
               {
                 if(x_diff <min_diff)
                 {
                    min_diff =x_diff;
                 }
               }
            }

            //如果符合条件，则放到容器里去
            if (min_diff <= maxYDifference || line_length >= minLineLength) {
                filtered_clusters[cluster_id].push_back(line);
            }
        }
    }
    return filtered_clusters;
}

// 计算中心点之和函数
float image_data::calculate_average_center(const std::vector<cv::Vec4i> &lines)
{
    float sum = 0.0f;
    for (const auto &line : lines)
    {   
        if(paking_final.parking_type==Parking_type::Level_space)
        {
           sum += (line[0] + line[2]) / 2.0f; // 计算中点x坐标的平均值
        }else if(paking_final.parking_type==Parking_type::Vertical_space)
        {
           sum += (line[1]+line[3]  )/ 2.0f ;
        }
    }
    if (!lines.empty())
    {
        return sum / lines.size(); // 返回平均值
    }
    return 0.0f; // 防止除以零
}

// 由于glob函数排的顺序达不到预期，利用正则表达式改变匹配规则
bool compareNumeric(const String &a, const String &b)
{
    regex re("(\\d+)");
    smatch match_a, match_b;

    string name_a = a.substr(a.find_last_of("/\\") + 1);
    string name_b = b.substr(b.find_last_of("/\\") + 1);

    if (regex_search(name_a, match_a, re) && regex_search(name_b, match_b, re))
    {
        int num_a = stoi(match_a.str(1));
        int num_b = stoi(match_b.str(1));
        return num_a < num_b;
    }
    return a < b;
}

//选择停车位模式函数，选前三帧作为判断依据，如果都是垂直车位/水平车位/斜列车位 选择进入哪个模式
void image_data::Select_Type(const string&dir)
{
   vector<cv::String> png_files;
   glob(dir+"/*.png",png_files,false);
   sort(png_files.begin(),png_files.end(),compareNumeric);
   Parking p1;
   for(const auto &file_path : png_files)
   {
      Mat mat1 = imread(file_path, 0);
      Parking p1;
      parking_judgment p2;
      p2.preprocess(mat1);
      p2.Determinespacetype(p2.preprocessmat,p1);
      frame_id++;
      if(frame_id==1)
      {
        paking_type_1=p1;
      }else if(frame_id==2)
      {
        paking_type_2=p1;
      }else if(frame_id==3)
      {
        paking_type_3=p1;
      }
      if(frame_id==3)
      {
        break;
      }
   }
   if(paking_type_1 ==paking_type_2&& paking_type_1==paking_type_3)
   {
      paking_final=paking_type_1;
   }
}



// 读取文件夹函数
void image_data::load_img(const string &dir)
{
    vector<cv::String> png_files;
    glob(dir + "/*.png", png_files, false);
    sort(png_files.begin(), png_files.end(), compareNumeric);
    for (const auto &file_path : png_files)
    {
        Mat mat1 = imread(file_path, 0);
        find_point(mat1); 
        cout<<"当前文件夹位置是"<<file_path<<endl;
    }
}

// 点的寻找和显示
std::vector<image_data::LaneLinePoints> image_data::find_point(const cv::Mat&mat1)
{

    
    if (mat1.empty())
    {
        std::cerr << "Error: Image could not be loaded." << std::endl;
        return std::vector<image_data::LaneLinePoints>();
    }

    std::vector<cv::Point2f> vcs_points;
    // 将mat化为二值化图像，只显示label为1的区域，初始化全为0，把label为1的点设为255
    Mat Points_only = Mat::zeros(mat1.size(), CV_8UC1);
    for (int i = 0; i < mat1.rows; i++)
    {
        for (int j = 0; j < mat1.cols; j++)
        {
            if (mat1.at<uchar>(i, j) == 1)
            {
                cv::Point2f pt_img(j, i);
                cv::Point2f pt_vcs;
                IPMtoVCS(pt_img, &pt_vcs);
                vcs_points.push_back(pt_vcs);
                // i是行对应y的值，j是列对应x的值
                Points_only.at<uchar>(i, j) = 255;
            }
        }
    }
    // 显示所有label为1的点
    imshow("points", Points_only);

    // 霍夫变换提取
    std::vector<cv::Vec4i> lines;
    double rho = 1;
    double theta = CV_PI / 180;
    double minLineLength = 30;
    double maxLineGap = 10;

    cv::HoughLinesP(Points_only, lines, rho, theta, minLineLength, maxLineGap);
    cv::Mat lineImage = cv::Mat::zeros(mat1.size(), CV_8UC3);
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4i l = lines[i];
        cv::line(lineImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3, 1);
    }
     cv::imshow("hough_Lines", lineImage);
    // cout<<"size是"<<lines.size()<<endl;

    // 霍夫变换得到的线中把水平以及与主方向偏离较大的线拿掉
    std::vector<cv::Vec4i> major_axis_lines;
    for (const auto &l : lines)
    {
        double line_angle = calculateLineAngles(cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]));
        if(paking_final.parking_type=Parking_type::Level_space)
        {
            if (line_angle >= 80 && line_angle <= 100)
            {
               major_axis_lines.push_back(l);
            }
        }else if(paking_final.parking_type=Parking_type::Vertical_space)
        {
            if(line_angle>=170&&line_angle<=180||line_angle>=0&&line_angle<=10)
            {
                major_axis_lines.push_back(l);
            }
        }else if(paking_final.parking_type==Parking_type::Diagonal_spaces)
        {
            if(line_angle>=30&&line_angle<=60||line_angle>=100&&line_angle<=130)
            {
                major_axis_lines.push_back(l);
            }
        }
    }
    cv::Mat lineImage2 = cv::Mat::zeros(mat1.size(), CV_8UC3);
    for (const auto &l : major_axis_lines)
    {
        cv::line(lineImage2, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3, 1);
    }
    cv::imshow("Filtered Hough Lines", lineImage2);

    // DBS聚类部分
    std::vector<int> Hough_lables = DBSCAN(major_axis_lines, 15, 1);
    std::map<int, cv::Scalar> clusterColors;
    cv::Mat cluster_image2 = cv::Mat::zeros(lineImage.size(), CV_8UC3);

    for (size_t i = 0; i < major_axis_lines.size(); i++)
    {
        int clusterId = Hough_lables[i];
        if (clusterColors.find(clusterId) == clusterColors.end())
        {
            // 为新聚类生成随机颜色
            clusterColors[clusterId] = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        }
        cv::Vec4i l = major_axis_lines[i];
        // if(calculatedistance(cv::Point(l[0], l[1]),cv::Point(l[2], l[3]))>20)
        //{
        cv::line(cluster_image2, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), clusterColors[clusterId], 3, cv::LINE_AA);
        //}
    }
    // cv::namedWindow("Clustered Lines1", WINDOW_FREERATIO);
    cv::imshow("Clustered Lines1", cluster_image2);
    // cv::waitKey(0);

    // 二次聚类函数
    std::vector<int> new_hough_lables = mergeClusters2(major_axis_lines, Hough_lables, 3, 15);
    std::map<int, cv::Scalar> clusterColors2;
    std::map<int, cv::Scalar> clusterColors3;

    clusterColors2[0] = cv::Scalar(0, 0, 255);   // 红色
    clusterColors2[1] = cv::Scalar(0, 255, 255); // 黄色
    clusterColors2[2] = cv::Scalar(255, 0, 0);   // 蓝色
    clusterColors2[3] = cv::Scalar(0, 255, 0);   // 绿色
    cv::Mat cluster_image3 = cv::Mat::zeros(lineImage.size(), CV_8UC3);
    for (size_t i = 0; i < major_axis_lines.size(); i++)
    {
        int clusterId = new_hough_lables[i];
        if (clusterColors3.find(clusterId) == clusterColors3.end())
        {
            clusterColors3[clusterId] = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        }
        cv::Vec4i l = major_axis_lines[i];

        // if(calculatedistance(cv::Point(l[0], l[1]),cv::Point(l[2], l[3]))>10)
        //{
            cv::line(cluster_image3, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), clusterColors3[clusterId], 3, cv::LINE_AA);
        //}
    }
    cv::imshow("Clustered Lines2", cluster_image3);

    // 把得到的聚类结果中的线段连接起来，先把聚类的线放在一起，一个labels对应一系列线
    std::map<int, std::vector<cv::Vec4i>> clustermaps;
    cv::Mat cluster_image5 = cv::Mat::zeros(lineImage.size(), CV_8UC3);
    for (size_t i = 0; i < major_axis_lines.size(); i++)
    {
        cv::Point2f p1(major_axis_lines[i][0], major_axis_lines[i][1]);
        cv::Point2f p2(major_axis_lines[i][1], major_axis_lines[i][2]);
        int this_cluster_id = new_hough_lables[i];
        clustermaps[this_cluster_id].push_back(major_axis_lines[i]);
    }
    // 聚类再处理 只要离车体最近的两个类
    std::map<int, std::vector<cv::Vec4i>> closestClusters = getClosestClustersToCenter(clustermaps, 224);
    std::map<int, std::vector<cv::Vec4i>> filiterClusters = filterOutliersInClusters(closestClusters,148,40,40);
    std::map<int, cv::Scalar> clusterColors4;
    clusterColors4[1] = cv::Scalar(0, 0, 255); // 红色
    clusterColors4[2] = cv::Scalar(0, 255, 255);

    int colorIndex = 1;
    for (const auto &cluster : filiterClusters)
    {
        for (const auto &line : cluster.second)
        {
            cv::Point pt1(line[0], line[1]);
            cv::Point pt2(line[2], line[3]);
            cv::line(cluster_image5, pt1, pt2, clusterColors4[colorIndex], 3, cv::LINE_AA); // Draw line with anti-aliasing
        }
        ++colorIndex;
    }
    cv::imshow("final lines", cluster_image5);

    std::vector<std::vector<cv::Vec4i>> current_clusters;
    for (auto &pair : filiterClusters)
    {
        current_clusters.push_back(pair.second);
    }
    // 得到每个聚类的各条线端点
    std::vector<cv::Vec4f> fitted_lines;
    for (const auto &cluster : current_clusters)
    {
        std::vector<cv::Point2f> this_points;
        for (const auto &line : cluster)
        {
            this_points.push_back(cv::Point2f(line[0], line[1]));
            this_points.push_back(cv::Point2f(line[2], line[3]));
        }

        // 通过PCA主层次分析法取得到各个聚类的主方向  以及最远的两个端点
        cv::Mat m1;
        cv::Mat data_pts = cv::Mat(this_points.size(), 2, CV_32F, &this_points[0]);
        cv::PCA pca_analysis(data_pts, m1, cv::PCA::DATA_AS_ROW);
        // PCA找到中心点
        cv::Point cntr = cv::Point(pca_analysis.mean.at<float>(0, 0), pca_analysis.mean.at<float>(0, 1));
        // PCA找到主方向
        cv::Vec2f eigen_vec = cv::Vec2f(pca_analysis.eigenvectors.at<float>(0, 0), pca_analysis.eigenvectors.at<float>(0, 1));
        eigen_vec = eigen_vec / cv::norm(eigen_vec);

        // PCA得到方向上的最小值和最大值
        float min_proj = std::numeric_limits<float>::max(), max_proj = -std::numeric_limits<float>::max();
        cv::Point2f pt_min, pt_max;
        for (const auto &pt : this_points)
        {
            // 投影到主成分上
            float proj = eigen_vec[0] * (pt.x - cntr.x) + eigen_vec[1] * (pt.y - cntr.y);
            if (proj < min_proj)
            {
                min_proj = proj;
                pt_min = pt;
            }
            if (proj > max_proj)
            {
                max_proj = proj;
                pt_max = pt;
            }
        } // if(calculatedistance(pt_min,pt_max)>30)
        //{
        fitted_lines.push_back(cv::Vec4f(pt_min.x, pt_min.y, pt_max.x, pt_max.y));
        //}
    }

    cv::Mat final_image = cv::Mat::zeros(mat1.size(), CV_8UC3);
    std::vector<image_data::LaneLinePoints> final_lines;
    for (size_t i = 0; i < fitted_lines.size(); i++)
    {
        int clusterId3 = new_hough_lables[i];
        Vec4f line = fitted_lines[i];
        cv::Point start(line[1], line[0]);
        cv::Point end(line[3], line[2]);
        cv::Point2f startPointVCS, endPointVCS;
        IPMtoVCS(start, &startPointVCS);
        IPMtoVCS(end, &endPointVCS);
        //cout << "线" << i << "的起始点:" << startPointVCS << endl;
        //cout << "线" << i << "的终点:" << endPointVCS << endl;
        LaneLinePoints p1(startPointVCS, endPointVCS);
        final_lines.push_back(p1);
        cv::line(final_image, start, end, Scalar(rand() % 255, rand() % 255, rand() % 255), 3, cv::LINE_AA);
    }
    // cv::namedWindow("Fitted Lines",0);
    //cv::imshow("Fitted Lines", final_image);

    // 定义VCS下坐标原点
    cv::Point2f referencePoint(ipm_parsing_image_width / 2, ipm_parsing_image_height / 2);
    cv::Point2f referencePointVCS;
    IPMtoVCS(referencePoint, &referencePointVCS);
    // 定义一个存放到坐标原点距离和自身起始点和终止点的vector
    std::vector<std::pair<float, cv::Vec4f>> distance_lines;
    for (const auto &line : fitted_lines)
    {
        cv::Point2f startPointImg(line[0], line[1]), endPointImg(line[2], line[3]); // 得到VCS转换后的起始点和终止点
        cv::Point2f startPointVCS, endPointVCS;
        IPMtoVCS(startPointImg, &startPointVCS);
        IPMtoVCS(endPointImg, &endPointVCS);
        float dist2_start_point = calculatedistance(startPointVCS, referencePointVCS);
        float dist2_end_point = calculatedistance(endPointVCS, referencePoint);
        float minDist = std::min(dist2_start_point, dist2_end_point);
        distance_lines.emplace_back(minDist, cv::Vec4f(startPointVCS.x, startPointVCS.y, endPointVCS.x, endPointVCS.y));
    }
    std::sort(distance_lines.begin(), distance_lines.end(), [](const std::pair<float, cv::Vec4f> &a, const std::pair<float, cv::Vec4f> &b)
              { return a.first < b.first; });

    // 找到距离最近的两条线
    std::vector<cv::Vec4f> closestLines;
    for (int i = 0; i < std::min(2, static_cast<int>(distance_lines.size())); ++i)
    {
        closestLines.push_back(distance_lines[i].second);
    }
    cv::Mat final_image2 = cv::Mat::zeros(mat1.size(), CV_8UC3);
    for (size_t i = 0; i < closestLines.size(); i++)
    {
        int clusterId4 = Hough_lables[i];
        Vec4f current_line = closestLines[i];
        cv::Point2f vcs_start(current_line[0], current_line[1]);
        cv::Point2f vcs_end(current_line[2], current_line[3]);
        cv::Point2f ipm_start;
        cv::Point2f ipm_end;
        VCStoIPM(vcs_start, &ipm_start);
        VCStoIPM(vcs_end, &ipm_end);
        cv::line(final_image2, ipm_start, ipm_end, Scalar(rand() % 255, rand() % 255, rand() % 255), 3, cv::LINE_AA);
    }
     //cv::imshow("Finnal Lines", final_image2);
    cv::waitKey(0);

    return final_lines;
}
