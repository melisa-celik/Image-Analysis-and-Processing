#pragma once

#pragma once

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "EtalonClassifier.h"
#include <opencv2/opencv.hpp>
#include <unordered_map>

double computeArea(const cv::Mat& binaryImage);

cv::Point2d computeCenterOfMass(const cv::Moments& moments);

int computeCircumference(const cv::Mat& binaryImage);

void computeMinMaxMoments(const cv::Moments& moments, double& minMoment, double& maxMoment);

void computeFeatures(const cv::Mat& binaryImage, int imgIndex);

FeatureVector computeFeatures(const cv::Mat& binaryImage);

std::vector<cv::Mat> extractObjects(const cv::Mat& image);

//void drawFigure1(const std::vector<FeatureVector>& featureVectors, const std::vector<std::string>& labels, const std::vector<Etalon>& etalons);
//
//void drawFigure2(const std::vector<FeatureVector>& featureVectors, const std::vector<std::string>& labels, const std::vector<Etalon>& etalons);

#endif // FUNCTIONS_H
