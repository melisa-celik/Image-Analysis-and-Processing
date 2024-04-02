#pragma once

#ifndef ETALON_CLASSIFIER_H
#define ETALON_CLASSIFIER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>

struct FeatureVector {
    double area;
    double circumference;
    double F1;
    double F2;
};

class EtalonClassifier {
public:
    EtalonClassifier();

    void addTrainingData(const std::vector<FeatureVector>& featureVectors, const std::vector<std::string>& labels);

    std::string classifyObject(const FeatureVector& unknownObject);

private:
    struct Etalon {
        double F1;
        double F2;
        std::string label;
    };

    std::vector<Etalon> etalons;

    void computeEtalons(const std::vector<FeatureVector>& featureVectors, const std::vector<std::string>& labels);

    double distance(const FeatureVector& v1, const Etalon& etalon);
};

#endif
