// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#include "utility.h"

class ImageProjection
{
private:
    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;

    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;

    pcl::PointCloud<PointType>::Ptr fullCloud;     // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat;  // range matrix for range image
    cv::Mat labelMat;  // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    std::vector<std::pair<int8_t, int8_t>> neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

public:
    ImageProjection() : nh("~")
    {

        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info>("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();
    }

    void allocateMemory()
    {

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN * Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN * Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN * Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN * Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN * Horizon_SCAN, 0);

        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1;
        neighbor.second = 0;
        neighborIterator.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = 1;
        neighborIterator.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = -1;
        neighborIterator.push_back(neighbor);
        neighbor.first = 1;
        neighbor.second = 0;
        neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN * Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN * Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN * Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN * Horizon_SCAN];
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection() {}

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {

        cloudHeader = laserCloudMsg->header;
        // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
        // have "ring" channel in the cloud
        if (useCloudRing == true)
        {
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);
            if (laserCloudInRing->is_dense == false)
            {
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                ros::shutdown();
            }
        }
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);
        // 2. Start and end angle of a scan
        findStartEndAngle();
        // 3. Range image projection
        projectPointCloud();
        // 4. Mark ground points
        groundRemoval();
        // 5. Point cloud segmentation
        cloudSegmentation();
        // 6. Publish all clouds
        publishCloud();
        // 7. Reset parameters for next iteration
        resetParameters();
    }

    void findStartEndAngle()
    {
        // start and end orientation of this cloud
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        segMsg.endOrientation = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                       laserCloudIn->points[laserCloudIn->points.size() - 1].x) +
                                2 * M_PI;
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI)
        {
            segMsg.endOrientation -= 2 * M_PI;
        }
        else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
    }

    // 原始点云投影为16x1800的二维深度图
    void projectPointCloud()
    {
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize;
        PointType thisPoint;

        // 点云中点的数量
        cloudSize = laserCloudIn->points.size();

        // 遍历所有点
        for (size_t i = 0; i < cloudSize; ++i)
        {
            // 获取坐标x,y,z
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            // VLP-16激光雷达存在该数据
            if (useCloudRing == true)
            {
                // rowIdn 0-15索引
                rowIdn = laserCloudInRing->points[i].ring;
            }
            else
            {
                // 计算俯仰角
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                // 计算行索引
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            }
            // 判断行索引的合法性
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 水平弧度值 转换至 角度值 -180 ~ 180
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            //
            columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;

            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            // 舍弃深度小于最小阈值的点
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            if (range < sensorMinimumRange)
                continue;

            // 构建深度图
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 点的强度值
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            // 深度图像素坐标对应的索引
            index = columnIdn + rowIdn * Horizon_SCAN;

            // 构建点云
            fullCloud->points[index] = thisPoint;

            fullInfoCloud->points[index] = thisPoint;
            // 修改强度为深度
            fullInfoCloud->points[index].intensity = range;
        }
    }

    // 地面点分割
    void groundRemoval()
    {
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // 遍历下方8x1800的深度图
        for (size_t j = 0; j < Horizon_SCAN; ++j)
        {
            for (size_t i = 0; i < groundScanInd; ++i)
            {
                // 当前像素点的下方像素
                lowerInd = j + (i)*Horizon_SCAN;
                // 当前像素点的上方像素
                upperInd = j + (i + 1) * Horizon_SCAN;
                // 对应像素无深度值
                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1)
                {
                    groundMat.at<int8_t>(i, j) = -1;
                    continue;
                }
                // 构建向量
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;
                // 计算向量与[x_y]平面的夹角
                angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;
                // 激光雷达与地面的夹角，10度范围内均视为地面
                if (abs(angle - sensorMountAngle) <= 10)
                {
                    groundMat.at<int8_t>(i, j) = 1;
                    groundMat.at<int8_t>(i + 1, j) = 1;
                }
            }
        }
        // 标记地面点和无效点
        for (size_t i = 0; i < N_SCAN; ++i)
        {
            for (size_t j = 0; j < Horizon_SCAN; ++j)
            {
                if (groundMat.at<int8_t>(i, j) == 1 || rangeMat.at<float>(i, j) == FLT_MAX)
                {
                    labelMat.at<int>(i, j) = -1;
                }
            }
        }
        // 如果有该话题的订阅者，发布地面点云数据
        if (pubGroundCloud.getNumSubscribers() != 0)
        {
            for (size_t i = 0; i <= groundScanInd; ++i)
            {
                for (size_t j = 0; j < Horizon_SCAN; ++j)
                {
                    if (groundMat.at<int8_t>(i, j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                }
            }
        }
    }

    // 点云分割
    void cloudSegmentation()
    {
        // 分割过程
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i, j) == 0)
                    labelComponents(i, j);

        int sizeOfSegCloud = 0;
        // 提取分割后的点云 里程计
        for (size_t i = 0; i < N_SCAN; ++i)
        {

            segMsg.startRingIndex[i] = sizeOfSegCloud - 1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; ++j)
            {
                if (labelMat.at<int>(i, j) > 0 || groundMat.at<int8_t>(i, j) == 1)
                {
                    // outliers that will not be used for optimization (always continue)
                    if (labelMat.at<int>(i, j) == 999999)
                    {
                        if (i > groundScanInd && j % 5 == 0)
                        {
                            outlierCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                            continue;
                        }
                        else
                        {
                            continue;
                        }
                    }
                    // majority of ground points are skipped
                    if (groundMat.at<int8_t>(i, j) == 1)
                    {
                        if (j % 5 != 0 && j > 5 && j < Horizon_SCAN - 5)
                            continue;
                    }
                    // mark ground points so they will not be considered as edge features later
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i, j) == 1);
                    // mark the points' column index for marking occlusion later
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info
                    segMsg.segmentedCloudRange[sizeOfSegCloud] = rangeMat.at<float>(i, j);
                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                    // size of seg cloud
                    ++sizeOfSegCloud;
                }
            }

            segMsg.endRingIndex[i] = sizeOfSegCloud - 1 - 5;
        }

        // extract segmented cloud for visualization
        if (pubSegmentedCloudPure.getNumSubscribers() != 0)
        {
            for (size_t i = 0; i < N_SCAN; ++i)
            {
                for (size_t j = 0; j < Horizon_SCAN; ++j)
                {
                    if (labelMat.at<int>(i, j) > 0 && labelMat.at<int>(i, j) != 999999)
                    {
                        segmentedCloudPure->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i, j);
                    }
                }
            }
        }
    }

    // 目标聚类分割
    void labelComponents(int row, int col)
    {
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY;
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;

        // 广度优先搜索
        while (queueSize > 0)
        {
            // 提取点
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // 标记该点
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;
            // 循环遍历所有的邻居节点
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter)
            {
                // 计算邻居的索引
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // 判断垂直索引合法性
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // 判断水平索引合法性
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                // 阻止无效循环
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;
                // 深度较大的点
                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                              rangeMat.at<float>(thisIndX, thisIndY));
                // 深度较小的点
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
                              rangeMat.at<float>(thisIndX, thisIndY));
                // 角AOB实际上就是深度图像素点划分时所设定的角度分辨率
                // 若邻域点为同一行的点，则角AOB为水平方向的角度分辨率
                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                // 若邻域点为同一列的点，则角AOB为垂直方向的角度分辨率
                else
                    alpha = segmentAlphaY;
                // 计算角OAB
                angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));
                // 若角OAB大于所设阈值，则认为A和B位于同一目标
                if (angle > segmentTheta)
                {
                    // 重置搜索起点，以当前邻域点为起点继续搜索
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }
        // 检验该分割是否有效
        bool feasibleSegment = false;
        // 若当前目标点个数大于30个，则该目标为有效目标
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        // 若不足30但大于所设最小有效阈值，则判断该目标是否跨越足够多的行。若是，则同样标记为有效目标
        else if (allPushedIndSize >= segmentValidPointNum)
        {
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;
        }
        // 分割有效，标记这些点
        if (feasibleSegment == true)
        {
            ++labelCount;
        }
        // 分割无效，标记这些点
        else
        {
            for (size_t i = 0; i < allPushedIndSize; ++i)
            {
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    void publishCloud()
    {
        // 1. Publish Seg Cloud Info
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);
        // 2. Publish clouds
        sensor_msgs::PointCloud2 laserCloudTemp;

        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);
        // segmented cloud with ground
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);
        // projected full cloud
        if (pubFullCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }
        // original dense ground cloud
        if (pubGroundCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground
        if (pubSegmentedCloudPure.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // projected full cloud info
        if (pubFullInfoCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};

int main(int argc, char **argv)
{

    ros::init(argc, argv, "lego_loam");

    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
