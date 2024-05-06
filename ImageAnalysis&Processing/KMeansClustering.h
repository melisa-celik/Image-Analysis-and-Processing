#pragma once

#ifndef KMEANSCLUSTERING_H
#define KMEANSCLUSTERING_H

#include <opencv2/opencv.hpp>
#include <vector>

class KMeansClustering
{
public:
    KMeansClustering(int k);
    void train(const std::vector<cv::Vec2d>& features, int maxIterations = 100);
    std::vector<int> predict(const std::vector<cv::Vec2d>& features);

private:
    int k;
    std::vector<cv::Vec2d> centroids;

    double calculateDistance(const cv::Vec2d& v1, const cv::Vec2d& v2);
};

#endif // KMEANSCLUSTERING_H
