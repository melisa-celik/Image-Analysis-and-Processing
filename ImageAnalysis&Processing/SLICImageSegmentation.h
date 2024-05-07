#pragma once

#ifndef SLIC_IMAGE_SEGMENTATION_H
#define SLIC_IMAGE_SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <vector>

#define SQR(x) ((x) * (x))

class SLICImageSegmentation {
public:
    SLICImageSegmentation(const cv::Mat& img, int k, double balance);
    ~SLICImageSegmentation();

    void initializeClusterCenters();
    void assignPixelsToClusters();
    void updateClusterCenters();
    void segmentImage();
    cv::Mat getSegmentedImage();

private:
    cv::Mat image;
    int K;          // Number of desired clusters
    int S;          // Grid step
    double m;       // Balancing factor

    std::vector<std::vector<double>> clusterCenters;
    std::vector<int> clusterAssignments;
    std::vector<std::vector<double>> clusterCentersPrev; // Previous cluster centers
};

#endif // SLIC_IMAGE_SEGMENTATION_H