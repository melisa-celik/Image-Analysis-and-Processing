#include "KMeansClustering.h"
#include <random>
#include <limits>

KMeansClustering::KMeansClustering(int k) : k(k) {}

void KMeansClustering::train(const std::vector<cv::Vec2d>& features, int maxIterations)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> randomIndex(0, features.size() - 1);

    centroids.clear();
    // Initialize centroids randomly from the input features
    for (int i = 0; i < k; ++i) {
        centroids.push_back(features[randomIndex(gen)]);
    }

    // Perform k-means clustering
    for (int iter = 0; iter < maxIterations; ++iter) {
        std::vector<std::vector<cv::Vec2d>> clusters(k);
        std::vector<cv::Vec2d> newCentroids(k);

        // Assign each feature to the closest centroid
        for (const auto& feature : features) {
            double minDistance = std::numeric_limits<double>::max();
            int closestCentroid = -1;

            for (int j = 0; j < k; ++j) {
                double dist = calculateDistance(feature, centroids[j]);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestCentroid = j;
                }
            }

            clusters[closestCentroid].push_back(feature);
        }

        // Update centroids' positions
        bool centroidsChanged = false;
        for (int j = 0; j < k; ++j) {
            if (!clusters[j].empty()) {
                cv::Vec2d sum(0.0, 0.0);
                for (const auto& point : clusters[j]) {
                    sum += point;
                }
                newCentroids[j] = sum / static_cast<double>(clusters[j].size());

                if (calculateDistance(newCentroids[j], centroids[j]) > 1e-6) {
                    centroidsChanged = true;
                }
            }
        }

        if (!centroidsChanged) {
            break; // Convergence achieved
        }

        centroids = newCentroids;
    }
}

std::vector<int> KMeansClustering::predict(const std::vector<cv::Vec2d>& features)
{
    std::vector<int> predictions;
    for (const auto& feature : features) {
        double minDistance = std::numeric_limits<double>::max();
        int closestCentroid = -1;

        for (size_t j = 0; j < centroids.size(); ++j) {
            double dist = calculateDistance(feature, centroids[j]);
            if (dist < minDistance) {
                minDistance = dist;
                closestCentroid = j;
            }
        }

        predictions.push_back(closestCentroid);
    }
    return predictions;
}

double KMeansClustering::calculateDistance(const cv::Vec2d& v1, const cv::Vec2d& v2)
{
    return cv::norm(v1 - v2);
}
