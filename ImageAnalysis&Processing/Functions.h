#pragma once

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "EtalonClassifier.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>

double computeArea(const cv::Mat& binaryImage);
cv::Point2d computeCenterOfMass(const cv::Moments& moments);
int computeCircumference(const cv::Mat& binaryImage);
void computeMinMaxMoments(const cv::Moments& moments, double& minMoment, double& maxMoment);
double computeF1(const cv::Mat& binaryImage);
double computeF2(const cv::Mat& binaryImage);
void computeFeatures(const cv::Mat& binaryImage, int imgIndex, cv::Vec2d& features);
int computeNumberOfCorners(const cv::Mat& binaryImage);

#endif // FUNCTIONS_H
