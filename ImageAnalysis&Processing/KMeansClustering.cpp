#include "KMeansClustering.h"
#include <random>
#include <limits>

KMeansClustering::KMeansClustering(int k) : k(k) {}

void KMeansClustering::train(const std::vector<cv::Vec2d>& features, int maxIterations)
{
    // Initialize centroids based on first k features
    centroids.clear();
    for (int i = 0; i < k; ++i) {
        centroids.push_back(features[i]);
    }

    // Perform k-means clustering
    for (int iter = 0; iter < maxIterations; ++iter) {
        std::vector<std::vector<cv::Vec2d>> clusters(k);
        std::vector<cv::Vec2d> newCentroids(k, cv::Vec2d(0.0, 0.0));
        std::vector<int> clusterSizes(k, 0);

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
            newCentroids[closestCentroid] += feature;
            clusterSizes[closestCentroid]++;
        }

        // Update centroids' positions
        bool centroidsChanged = false;
        for (int j = 0; j < k; ++j) {
            if (clusterSizes[j] > 0) {
                newCentroids[j] /= static_cast<double>(clusterSizes[j]);
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
