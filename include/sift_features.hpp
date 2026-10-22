#pragma once

// H4 - SIFT keypoints and descriptors.
//
// SIFT (Lowe 2004): keypoints are extrema of the Difference-of-Gaussians in a
// scale space; each one gets a dominant orientation and a descriptor of
// 4 x 4 cells x 8 orientation bins = 128 values (histograms of gradients,
// normalised, so they tolerate brightness changes).

#include "coin_patch.hpp"

#include <opencv2/core.hpp>

#include <vector>

struct CoinFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;  // one row of 128 floats per keypoint
};

// Contrast normalisation applied to every patch before SIFT (see params.hpp).
cv::Mat normalizeContrast(const cv::Mat& gray);

// Detects SIFT keypoints inside the patch mask and computes their descriptors.
// A lower contrast threshold keeps weaker (low-contrast) keypoints too.
// rootSift = true converts the descriptors to RootSIFT (see params.hpp).
CoinFeatures extractSift(const CoinPatch& patch, double contrastThreshold = 0.04,
                         bool rootSift = false);

// Draws the keypoints (circle = scale, line = orientation) on the patch.
cv::Mat drawSiftKeypoints(const CoinPatch& patch, const CoinFeatures& features);
