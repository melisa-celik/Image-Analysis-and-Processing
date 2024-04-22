// EtalonClassifier.h
#ifndef ETALONCLASSIFIER_H
#define ETALONCLASSIFIER_H

#include <opencv2/opencv.hpp>

class EtalonClassifier {
public:
    EtalonClassifier();
    ~EtalonClassifier();
    cv::Vec2d getFeatures(const cv::Mat& binaryImage);
    void computeEthalons(const std::vector<cv::Vec2d>& features, const std::vector<std::string>& labels);
    void saveEthalons(const std::string& filename);
    void loadEthalons(const std::string& filename);
    std::string classifyObject(const cv::Mat& testImage);
    std::string classifyShape(const cv::Mat& binaryImage);
    const std::map<std::string, cv::Vec2d>& getEthalons() const; 

private:
    std::map<std::string, cv::Vec2d> ethalons;
    int numFeatures; // Number of features (F1 and F2)
    void computeMinMaxFeatures(const std::vector<cv::Vec2d>& features, cv::Vec2d& minFeatures, cv::Vec2d& maxFeatures);
    double distance(const cv::Vec2d& v1, const cv::Vec2d& v2);
    cv::Vec2d computeFeatures(const cv::Mat& binaryImage);
};

#endif // ETALONCLASSIFIER_H
