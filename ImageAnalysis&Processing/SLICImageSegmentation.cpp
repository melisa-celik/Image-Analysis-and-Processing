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

double SLICImageSegmentation::computeGradientMagnitude(const cv::Mat& img, int x, int y)
{
    if (x < 1 || y < 1 || x >= img.cols - 1 || y >= img.rows - 1) {
        return 0.0; 
    }

    cv::Mat grayImg;
    cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    // Sobel gradient calculation
    cv::Mat gradX, gradY;
    cv::Sobel(grayImg, gradX, CV_16S, 1, 0);
    cv::Sobel(grayImg, gradY, CV_16S, 0, 1);

    // Calculate gradient magnitude at (x, y)
    double dx = static_cast<double>(gradX.at<short>(y, x));
    double dy = static_cast<double>(gradY.at<short>(y, x));
    return std::sqrt(dx * dx + dy * dy);
}

void SLICImageSegmentation::initializeClusterCenters()
{
    clusterCenters.clear();

    for (int y = S / 2; y < image.rows; y += S) {
        for (int x = S / 2; x < image.cols; x += S) {
            // Find lowest gradient position in a 3x3 neighborhood
            double minGradient = std::numeric_limits<double>::max();
            cv::Point minLoc;

            for (int ny = std::max(0, y - 1); ny <= std::min(y + 1, image.rows - 1); ++ny) {
                for (int nx = std::max(0, x - 1); nx <= std::min(x + 1, image.cols - 1); ++nx) {
                    double gradient = computeGradientMagnitude(image, nx, ny);
                    if (gradient < minGradient) {
                        minGradient = gradient;
                        minLoc = cv::Point(nx, ny);
                    }
                }
            }

            cv::Vec3b color = image.at<cv::Vec3b>(minLoc);

            // Initialize cluster center [R, G, B, x, y]
            std::vector<double> center = { static_cast<double>(color[2]), static_cast<double>(color[1]),
                                            static_cast<double>(color[0]), static_cast<double>(minLoc.x), static_cast<double>(minLoc.y) };
            clusterCenters.push_back(center);
        }
    }

    // Initialize cluster assignments (each pixel initially assigned to cluster 0)
    clusterAssignments.assign(image.rows * image.cols, 0);

    clusterCentersPrev = clusterCenters;
}

void SLICImageSegmentation::assignPixelsToClusters() {
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

                // Calculate distance using Eq. 3 (dRGB and dxy)
                double dRGB = std::sqrt(SQR(Rk - image.at<cv::Vec3b>(y, x)[2]) +
                    SQR(Gk - image.at<cv::Vec3b>(y, x)[1]) +
                    SQR(Bk - image.at<cv::Vec3b>(y, x)[0]));
                double dxy = std::sqrt(SQR(xk - x) + SQR(yk - y));
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

void SLICImageSegmentation::updateClusterCenters() {
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
    const double convergenceThreshold = 2.0; 
    bool converged = false;
    int iteration = 0;

    while (!converged && iteration < 100) { 
        assignPixelsToClusters(); 
        updateClusterCenters();   

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
