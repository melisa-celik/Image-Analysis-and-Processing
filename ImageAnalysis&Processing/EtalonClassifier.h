#pragma once

#ifndef ETALONCLASSIFIER_H
#define ETALONCLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include "Functions.h"

class EtalonClassifier
{
public:
    EtalonClassifier();
    ~EtalonClassifier();

    void computeEthalons(const std::vector<cv::Mat>& trainingImages, const std::vector<std::string>& labels);
    std::string classifyObject(const cv::Mat& testImage);

private:
    std::map<std::string, cv::Vec2d> ethalons;
    cv::Vec2d computeFeatures(const cv::Mat& binaryImage);
    
    double calculateDistance(const cv::Vec2d& v1, const cv::Vec2d& v2);
};

#endif // ETALONCLASSIFIER_H
