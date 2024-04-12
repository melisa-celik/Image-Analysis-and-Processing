#ifndef ETALONCLASSIFIER_H
#define ETALONCLASSIFIER_H

#include <opencv2/opencv.hpp>

class EtalonClassifier {
public:
    EtalonClassifier();
    ~EtalonClassifier();
    cv::Vec2d getFeatures(const cv::Mat& binaryImage);
    void computeEthalons(const std::vector<cv::Mat>& trainingImages, const std::vector<std::string>& labels);
    void saveEthalons(const std::string& filename);
    void loadEthalons(const std::string& filename);
    std::string classifyObject(const cv::Mat& testImage);

private:
    std::map<std::string, cv::Vec2d> ethalons;
    double computeArea(const cv::Mat& binaryImage);
    cv::Point2d computeCenterOfMass(const cv::Moments& moments);
    int computeCircumference(const cv::Mat& binaryImage);
    void computeMinMaxMoments(const cv::Moments& moments, double& minMoment, double& maxMoment);
    double computeF1(const cv::Mat& binaryImage);
    double computeF2(const cv::Mat& binaryImage);
    cv::Vec2d computeFeatures(const cv::Mat& binaryImage);
};

#endif // ETALONCLASSIFIER_H
