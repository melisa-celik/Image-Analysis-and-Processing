#pragma once

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <unordered_map>

double computeArea(const cv::Mat& binaryImage);

cv::Point2d computeCenterOfMass(const cv::Moments& moments);

int computeCircumference(const cv::Mat& binaryImage);

void computeMinMaxMoments(const cv::Moments& moments, double& minMoment, double& maxMoment);

void computeFeatures(const cv::Mat& binaryImage, int imgIndex);

std::unordered_map<std::string, cv::Point2d> computeEtalons(const std::unordered_map<std::string, std::vector<cv::Point2d>>& trainingData);

std::string classifyObjects(const std::unordered_map<std::string, cv::Point2d>& etalons, const cv::Point2d& unknownObject);

void drawFigure1(const std::unordered_map<std::string, std::vector<cv::Point2d>>& trainingData, const std::unordered_map<std::string, cv::Point2d>& etalons);

void drawFigure2(const std::unordered_map<std::string, cv::Point2d>& etalons);

#endif // FUNCTIONS_H
