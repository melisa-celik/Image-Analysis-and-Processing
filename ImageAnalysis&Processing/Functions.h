#pragma once

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>

double computeArea(const cv::Mat& binaryImage);

cv::Point2d computeCenterOfMass(const cv::Moments& moments);

int computeCircumference(const cv::Mat& binaryImage);

void computeMinMaxMoments(const cv::Moments& moments, double& minMoment, double& maxMoment);

void computeFeatures(const cv::Mat& binaryImage, int imgIndex);

void computeEthalons(const std::vector<std::vector<cv::Point>>& contours, const std::vector<int>& labels, std::vector<cv::Point2d>& sqaureEthalon,
	std::vector<cv::Point2d>& rectangleEthalon, std::vector<cv::Point2d>& triangleEthalon);

#endif // FUNCTIONS_H
