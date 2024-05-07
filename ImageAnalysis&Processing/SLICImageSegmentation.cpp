#include "SLICImageSegmentation.h"
#include <limits>

SLICImageSegmentation::SLICImageSegmentation(const cv::Mat& img, int k, double balance)
    : image(img), K(k), m(balance)
{
    int N = img.rows * img.cols;
    S = static_cast<int>(sqrt(N / K));

    initializeClusterCenters();
}

SLICImageSegmentation::~SLICImageSegmentation()
{
}

void SLICImageSegmentation::initializeClusterCenters()
{
    clusterCenters.clear();

    // Sample cluster centers at regular grid steps S
    for (int y = S / 2; y < image.rows; y += S) {
        for (int x = S / 2; x < image.cols; x += S) {
            // Find lowest gradient position in a 3x3 neighborhood
            int roiWidth = std::min(3, image.cols - x);
            int roiHeight = std::min(3, image.rows - y);
            cv::Rect roi(std::max(0, x - 1), std::max(0, y - 1), roiWidth, roiHeight);
            cv::Mat neighborhood = image(roi);

            if (neighborhood.empty()) {
                std::cerr << "Error: Neighborhood is empty." << std::endl;
                continue; // Skip to next iteration if neighborhood is empty
            }

            cv::Point min_loc;
            cv::Mat grayNeighborhood;
            cv::cvtColor(neighborhood, grayNeighborhood, cv::COLOR_BGR2GRAY);

            cv::minMaxLoc(grayNeighborhood, nullptr, nullptr, &min_loc);

            // Calculate valid position in the original image
            cv::Point original_loc = roi.tl() + min_loc;

            // Check if original_loc is within the bounds of the original image
            if (original_loc.x < 0 || original_loc.y < 0 || original_loc.x >= image.cols || original_loc.y >= image.rows) {
                std::cerr << "Error: Invalid original_loc coordinates." << std::endl;
                std::cerr << "original_loc: " << original_loc << ", image.cols: " << image.cols << ", image.rows: " << image.rows << std::endl;
                continue; // Skip to next iteration if original_loc is invalid
            }

            cv::Vec3b color = image.at<cv::Vec3b>(original_loc);

            // Initialize cluster center [R, G, B, x, y]
            std::vector<double> center = { static_cast<double>(color[2]), static_cast<double>(color[1]),
                                            static_cast<double>(color[0]), static_cast<double>(x), static_cast<double>(y) };
            clusterCenters.push_back(center);
        }
    }

    // Initialize cluster assignments (each pixel initially assigned to cluster 0)
    clusterAssignments.assign(image.rows * image.cols, 0);

    // Initialize previous cluster centers (same as current cluster centers at start)
    clusterCentersPrev = clusterCenters;
}


void SLICImageSegmentation::assignPixelsToClusters()
{
    const int numClusters = static_cast<int>(clusterCenters.size());

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            double minDistance = std::numeric_limits<double>::max();
            int closestCluster = 0;

            for (int k = 0; k < numClusters; ++k) {
                // Get cluster center coordinates
                double Rk = clusterCenters[k][0];
                double Gk = clusterCenters[k][1];
                double Bk = clusterCenters[k][2];
                double xk = clusterCenters[k][3];
                double yk = clusterCenters[k][4];

                // Calculate distance DS (Equation 3)
                double dRGB = sqrt((Rk - image.at<cv::Vec3b>(y, x)[2]) * (Rk - image.at<cv::Vec3b>(y, x)[2]) +
                    (Gk - image.at<cv::Vec3b>(y, x)[1]) * (Gk - image.at<cv::Vec3b>(y, x)[1]) +
                    (Bk - image.at<cv::Vec3b>(y, x)[0]) * (Bk - image.at<cv::Vec3b>(y, x)[0]));
                double dxy = sqrt((xk - x) * (xk - x) + (yk - y) * (yk - y));
                double DS = dRGB + (m / S) * dxy;

                // Assign pixel to the closest cluster
                if (DS < minDistance) {
                    minDistance = DS;
                    closestCluster = k;
                }
            }

            // Update pixel assignment
            clusterAssignments[y * image.cols + x] = closestCluster;
        }
    }
}

void SLICImageSegmentation::updateClusterCenters()
{
    const int numClusters = static_cast<int>(clusterCenters.size());

    // Reset cluster center accumulators
    std::vector<double> sumR(numClusters, 0.0);
    std::vector<double> sumG(numClusters, 0.0);
    std::vector<double> sumB(numClusters, 0.0);
    std::vector<double> sumX(numClusters, 0.0);
    std::vector<double> sumY(numClusters, 0.0);
    std::vector<int> numPixels(numClusters, 0);

    // Accumulate pixel values for each cluster
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int clusterIdx = clusterAssignments[y * image.cols + x];
            sumR[clusterIdx] += image.at<cv::Vec3b>(y, x)[2];
            sumG[clusterIdx] += image.at<cv::Vec3b>(y, x)[1];
            sumB[clusterIdx] += image.at<cv::Vec3b>(y, x)[0];
            sumX[clusterIdx] += x;
            sumY[clusterIdx] += y;
            numPixels[clusterIdx]++;
        }
    }

    // Update cluster centers
    for (int k = 0; k < numClusters; ++k) {
        if (numPixels[k] > 0) {
            clusterCenters[k][0] = sumR[k] / numPixels[k];
            clusterCenters[k][1] = sumG[k] / numPixels[k];
            clusterCenters[k][2] = sumB[k] / numPixels[k];
            clusterCenters[k][3] = sumX[k] / numPixels[k];
            clusterCenters[k][4] = sumY[k] / numPixels[k];
        }
    }
}

void SLICImageSegmentation::segmentImage()
{
    const double convergenceThreshold = 1.0; // Adjust as needed
    bool converged = false;
    int iteration = 0;

    while (!converged && iteration < 100) { // Limit iterations to avoid infinite loop
        assignPixelsToClusters(); // Assign pixels to clusters
        updateClusterCenters();   // Update cluster centers

        // Check for convergence based on the movement of cluster centers
        double maxCenterShift = 0.0;
        for (int k = 0; k < static_cast<int>(clusterCenters.size()); ++k) {
            double dx = clusterCenters[k][3] - clusterCentersPrev[k][3];
            double dy = clusterCenters[k][4] - clusterCentersPrev[k][4];
            double centerShift = sqrt(dx * dx + dy * dy);
            if (centerShift > maxCenterShift) {
                maxCenterShift = centerShift;
            }
        }

        if (maxCenterShift < convergenceThreshold) {
            converged = true;
        }

        // Update previous cluster centers for next iteration
        clusterCentersPrev = clusterCenters;

        iteration++;
    }
}

cv::Mat SLICImageSegmentation::getSegmentedImage()
{
    cv::Mat segmentedImage = image.clone();

    // Apply colors to each segment based on cluster centers
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int clusterIdx = clusterAssignments[y * image.cols + x];
            cv::Vec3b color(static_cast<uchar>(clusterCenters[clusterIdx][2]),
                static_cast<uchar>(clusterCenters[clusterIdx][1]),
                static_cast<uchar>(clusterCenters[clusterIdx][0]));
            segmentedImage.at<cv::Vec3b>(y, x) = color;
        }
    }

    return segmentedImage;
}
