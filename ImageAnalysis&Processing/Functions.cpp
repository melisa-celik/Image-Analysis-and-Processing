#include "Functions.h"

double computeArea(const cv::Mat& binaryImage)
{
    return cv::countNonZero(binaryImage);
}

cv::Point2d computeCenterOfMass(const cv::Moments& moments)
{
    return cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);
}

int computeCircumference(const cv::Mat& binaryImage)
{
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

void computeMinMaxMoments(const cv::Moments& moments, double& minMoment, double& maxMoment)
{
    double mu20 = moments.m20 / moments.m00;
    double mu02 = moments.m02 / moments.m00;
    double mu11 = moments.m11 / moments.m00;

    double diff = sqrt(pow(mu20 - mu02, 2) + 4 * pow(mu11, 2));

    maxMoment = 0.5 * (mu20 + mu02 + diff);
    minMoment = 0.5 * (mu20 + mu02 - diff);
}

void computeFeatures(const cv::Mat& binaryImage, int imgIndex)
{
    cv::Moments moments = cv::moments(binaryImage);

    double area = computeArea(binaryImage);
    cv::Point2d centerOfMass = computeCenterOfMass(moments);
    int circumference = computeCircumference(binaryImage);

    double minMoment, maxMoment;
    computeMinMaxMoments(moments, minMoment, maxMoment);

    double F1 = (double)(circumference * circumference) / (100 * area);
    double F2 = minMoment / maxMoment;

    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Img" << imgIndex << std::endl;
    std::cout << "Area: " << area << std::endl;
    std::cout << "Center of Mass (x_t, y_t): [" << centerOfMass.x << ", " << centerOfMass.y << "]" << std::endl;
    std::cout << "Circumference: " << circumference << std::endl;
    std::cout << "F1: " << F1 << std::endl;
    std::cout << "F2: " << F2 << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
}

FeatureVector computeFeatures(const cv::Mat& binaryImage) 
{
    cv::Moments moments = cv::moments(binaryImage);

    double area = computeArea(binaryImage);
    int circumference = computeCircumference(binaryImage);

    double minMoment, maxMoment;
    computeMinMaxMoments(moments, minMoment, maxMoment);

    double F1 = (double)(circumference * circumference) / (100 * area);
    double F2 = minMoment / maxMoment;

    FeatureVector features;
    features.area = area;
    features.circumference = circumference;
    features.F1 = F1;
    features.F2 = F2;

    return features;
}

std::vector<cv::Mat> extractObjects(const cv::Mat& image) {
    std::vector<cv::Mat> binaryImages;

    cv::Mat grayImage;
    //cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat binaryImage;
    cv::threshold(image, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Mat mask = cv::Mat::zeros(binaryImage.size(), CV_8UC1);
        cv::drawContours(mask, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);

        binaryImages.push_back(mask);
    }

    return binaryImages;
}


//std::unordered_map<std::string, cv::Point2d> computeEtalons(const std::unordered_map<std::string, std::vector<cv::Point2d>>& trainingData)
//{
//    std::unordered_map<std::string, cv::Point2d> etalons;
//
//    for (const auto& entry : trainingData) {
//        const std::string& classLabel = entry.first;
//        const std::vector<cv::Point2d>& features = entry.second;
//
//        cv::Point2d etalon(0, 0);
//
//        for (const auto& feature : features) {
//            etalon.x += feature.x;
//            etalon.y += feature.y;
//        }
//        etalon.x /= features.size();
//        etalon.y /= features.size();
//
//        etalons[classLabel] = etalon;
//    }
//
//    return etalons;
//}
//
//std::string classifyObjects(const std::unordered_map<std::string, cv::Point2d>& etalons, const cv::Point2d& unknownObject)
//{
//    std::string closestClass;
//    double minDistance = std::numeric_limits<double>::max();
//
//    for (const auto& entry : etalons) {
//        const cv::Point2d& etalon = entry.second;
//        double distance = cv::norm(etalon - unknownObject);
//
//        if (distance < minDistance) {
//            minDistance = distance;
//            closestClass = entry.first;
//        }
//    }
//
//    return closestClass;
//}
//
//void drawFigure1(const std::unordered_map<std::string, std::vector<cv::Point2d>>& trainingData, const std::unordered_map<std::string, cv::Point2d>& etalons)
//{
//    cv::Mat visualization(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));
//
//    for (const auto& entry : trainingData) {
//        const std::string& classLabel = entry.first;
//        const std::vector<cv::Point2d>& features = entry.second;
//
//        for (const auto& feature : features) {
//            cv::circle(visualization, feature, 3, cv::Scalar(0, 0, 0), cv::FILLED);
//        }
//
//        cv::Point2d etalon = etalons.at(classLabel);
//        cv::circle(visualization, etalon, 5, cv::Scalar(0, 0, 255), cv::FILLED);
//
//        cv::putText(visualization, classLabel, etalon + cv::Point2d(10, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
//    }
//
//    cv::imshow("Features and Etalons", visualization);
//    cv::waitKey(0);
//}
//
//void drawFigure2(const std::unordered_map<std::string, cv::Point2d>& etalons)
//{
//    cv::Mat illustration(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));
//
//    for (const auto& entry : etalons) {
//        const cv::Point2d& etalon = entry.second;
//
//        cv::circle(illustration, etalon, 3, cv::Scalar(0, 0, 0), cv::FILLED);
//    }
//
//    for (const auto& entry : etalons) {
//        const cv::Point2d& etalon = entry.second;
//
//        cv::circle(illustration, etalon, 5, cv::Scalar(0, 0, 255), cv::FILLED);
//    }
//
//    cv::imshow("Illustration", illustration);
//    cv::waitKey(0);
//}

