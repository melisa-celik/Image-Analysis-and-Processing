#pragma once

#ifndef EXERCISES_H
#define EXERCISES_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include "Etalon.h"

void Exercise1(const cv::Mat& image);

void Exercise2(const cv::Mat& image);

void Exercise3(const cv::Mat& image);

void detectAndColorizeObjects(cv::Mat& image, EtalonClassifier& classifier);

void processImage(const cv::Mat& image);

#endif 