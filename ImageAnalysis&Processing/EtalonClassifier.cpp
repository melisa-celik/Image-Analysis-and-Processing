#include "EtalonClassifier.h"
#include <fstream>
#include <iostream>

EtalonClassifier::EtalonClassifier() {}

EtalonClassifier::~EtalonClassifier() {}

cv::Vec2d EtalonClassifier::getFeatures(const cv::Mat& binaryImage)
{
    // Compute features using moments of the contour's binary image
	cv::Moments moments = cv::moments(binaryImage);
	double area = computeArea(binaryImage);
	cv::Point2d centerOfMass = computeCenterOfMass(moments);
	int circumference = computeCircumference(binaryImage);
	double minMoment, maxMoment;
	computeMinMaxMoments(moments, minMoment, maxMoment);
	double F1 = (double)(circumference * circumference) / (100 * area);
	double F2 = minMoment / maxMoment;
	return cv::Vec2d(F1, F2);
}

void EtalonClassifier::computeEthalons(const std::vector<cv::Mat>& trainingImages, const std::vector<std::string>& labels) {
    int imageIndex = 0;

    for (const auto& image : trainingImages) {
		cv::Vec2d features = getFeatures(image);
		ethalons[labels[imageIndex]] += features;
		imageIndex++;
	}

    for (auto& ethalon : ethalons) {
        ethalon.second /= static_cast<double>(std::count(labels.begin(), labels.end(), ethalon.first));
    }
}

void EtalonClassifier::saveEthalons(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        for (const auto& ethalon : ethalons) {
            file.write(reinterpret_cast<const char*>(&ethalon.second), sizeof(cv::Vec2d));
            size_t labelSize = ethalon.first.size();
            file.write(reinterpret_cast<const char*>(&labelSize), sizeof(size_t));
            file.write(ethalon.first.c_str(), labelSize);
        }
        file.close();
    }
    else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

void EtalonClassifier::loadEthalons(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        ethalons.clear();
        while (!file.eof()) {
            cv::Vec2d features;
            size_t labelSize;
            file.read(reinterpret_cast<char*>(&features), sizeof(cv::Vec2d));
            file.read(reinterpret_cast<char*>(&labelSize), sizeof(size_t));
            std::string label(labelSize, '\0');
            file.read(&label[0], labelSize);
            ethalons[label] = features;
        }
        file.close();
    }
    else {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
    }
}

std::string EtalonClassifier::classifyObject(const cv::Mat& testImage) {
    int imageIndex = 0;
    cv::Vec2d testFeatures = getFeatures(testImage);
    double minDistance = std::numeric_limits<double>::max();
    std::string closestLabel;

    for (const auto& ethalon : ethalons) {
        double distance = cv::norm(testFeatures - ethalon.second);
        if (distance < minDistance) {
            minDistance = distance;
            closestLabel = ethalon.first;
        }
    }

    return closestLabel;
}

double EtalonClassifier::computeArea(const cv::Mat& binaryImage) {
    return cv::countNonZero(binaryImage);
}

cv::Point2d EtalonClassifier::computeCenterOfMass(const cv::Moments& moments) {
    return cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);
}

int EtalonClassifier::computeCircumference(const cv::Mat& binaryImage) {
    int circumference = 0;
    for (int y = 1; y < binaryImage.rows - 1; ++y) {
        for (int x = 1; x < binaryImage.cols - 1; ++x) {
            if (binaryImage.at<uchar>(y, x) > 0) {
                if (binaryImage.at<uchar>(y - 1, x) == 0 ||
                    binaryImage.at<uchar>(y + 1, x) == 0 ||
                    binaryImage.at<uchar>(y, x - 1) == 0 ||
                    binaryImage.at<uchar>(y, x + 1) == 0) {
                    circumference++;
                }
            }
        }
    }
    return circumference;
}

void EtalonClassifier::computeMinMaxMoments(const cv::Moments& moments, double& minMoment, double& maxMoment) {
    double mu20 = moments.m20 / moments.m00;
    double mu02 = moments.m02 / moments.m00;
    double mu11 = moments.m11 / moments.m00;

    double diff = sqrt(pow(mu20 - mu02, 2) + 4 * pow(mu11, 2));

    maxMoment = 0.5 * (mu20 + mu02 + diff);
    minMoment = 0.5 * (mu20 + mu02 - diff);
}

double EtalonClassifier::computeF1(const cv::Mat& binaryImage) {
    double F1 = (double)(computeCircumference(binaryImage) * computeCircumference(binaryImage)) / (100 * computeArea(binaryImage));
    return F1;
}

double EtalonClassifier::computeF2(const cv::Mat& binaryImage) {
    cv::Moments moments = cv::moments(binaryImage);
    double minMoment, maxMoment;
    computeMinMaxMoments(moments, minMoment, maxMoment);
    double F2 = minMoment / maxMoment;
    return F2;
}

cv::Vec2d EtalonClassifier::computeFeatures(const cv::Mat& binaryImage, int imgIndex)
{
    // Compute features using moments of the contour's binary image
    cv::Moments moments = cv::moments(binaryImage);
    double area = computeArea(binaryImage);
    cv::Point2d centerOfMass = computeCenterOfMass(moments);
    int circumference = computeCircumference(binaryImage);
    double minMoment, maxMoment;
    computeMinMaxMoments(moments, minMoment, maxMoment);
    double F1 = (double)(circumference * circumference) / (100 * area);
    double F2 = minMoment / maxMoment;
    return cv::Vec2d(F1, F2);
}

std::string EtalonClassifier::classifyShape(const cv::Mat& binaryImage)
{
    if (isSquare(binaryImage)) {
        return "square";
    }
    else if (isRectangle(binaryImage)) {
        return "rectangle";
    }
    else if (isStar(binaryImage)) {
        return "star";
    }
    else {
        return "unknown";
    }
}

bool EtalonClassifier::isSquare(const cv::Mat& binaryImage)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.size() != 1) {
        return false; // Not a single contour
    }

    cv::RotatedRect rect = cv::minAreaRect(contours[0]);
    double aspectRatio = std::min(rect.size.width, rect.size.height) / std::max(rect.size.width, rect.size.height);
    return (aspectRatio > 0.9 && aspectRatio < 1.1);
}

bool EtalonClassifier::isRectangle(const cv::Mat& binaryImage)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.size() != 1) {
        return false; // Not a single contour
    }

    cv::RotatedRect rect = cv::minAreaRect(contours[0]);
    double aspectRatio = std::min(rect.size.width, rect.size.height) / std::max(rect.size.width, rect.size.height);
    return (aspectRatio > 0.5 && aspectRatio < 2.0);
}

bool EtalonClassifier::isStar(const cv::Mat& binaryImage)
{
    std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.size() != 1) {
		return false; // Not a single contour
	}

	cv::RotatedRect rect = cv::minAreaRect(contours[0]);
	double aspectRatio = std::min(rect.size.width, rect.size.height) / std::max(rect.size.width, rect.size.height);
	return (aspectRatio > 0.2 && aspectRatio < 0.5);

    //// Implement star detection logic here
    //// This could involve detecting specific patterns in the contour or using more complex shape descriptors
    //// For simplicity, let's assume a star if the contour has 6 vertices
    //std::vector<std::vector<cv::Point>> contours;
    //cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //if (contours.size() != 1) {
    //    return false; // Not a single contour
    //}

    //return (contours[0].size() == 6);
}
