#include "EtalonClassifier.h"
#include <cmath>
#include <limits>

EtalonClassifier::EtalonClassifier() {}

void EtalonClassifier::addTrainingData(const std::vector<FeatureVector>& featureVectors, const std::vector<std::string>& labels) {
    computeEtalons(featureVectors, labels);
}

std::string EtalonClassifier::classifyObject(const FeatureVector& unknownObject) {
    double minDist = std::numeric_limits<double>::max();
    std::string closestLabel;

    for (const auto& etalon : etalons) {
        double dist = distance(unknownObject, etalon);
        if (dist < minDist) {
            minDist = dist;
            closestLabel = etalon.label;
        }
    }

    return closestLabel;
}

void EtalonClassifier::computeEtalons(const std::vector<FeatureVector>& featureVectors, const std::vector<std::string>& labels) {
    std::unordered_map<std::string, std::vector<FeatureVector>> classFeatureVectors;

    // Group feature vectors by class
    for (size_t i = 0; i < labels.size(); ++i) {
        classFeatureVectors[labels[i]].push_back(featureVectors[i]);
    }

    // Compute etalons for each class
    for (const auto& pair : classFeatureVectors) {
        const std::string& className = pair.first;
        const std::vector<FeatureVector>& classVectors = pair.second;

        double sumF1 = 0.0;
        double sumF2 = 0.0;

        for (const auto& vector : classVectors) {
            sumF1 += vector.F1;
            sumF2 += vector.F2;
        }

        Etalon etalon;
        etalon.F1 = sumF1 / classVectors.size();
        etalon.F2 = sumF2 / classVectors.size();
        etalon.label = className;

        etalons.push_back(etalon);
    }
}

double EtalonClassifier::distance(const FeatureVector& v1, const Etalon& etalon) {
    return std::sqrt(std::pow(v1.F1 - etalon.F1, 2) + std::pow(v1.F2 - etalon.F2, 2));
}