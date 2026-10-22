#pragma once

// H4a - Diameter pre-filter.
//
// SIFT matching is the expensive step. Before running it, the diameter of each
// Hough candidate is compared with the diameter of every reference coin, both
// measured in pixels on the image (no real-world sizes are used). Only the
// reference types with a compatible diameter are matched later; a candidate
// compatible with none is discarded without computing SIFT.

#include "circle_detection.hpp"
#include "regions.hpp"

#include <opencv2/core.hpp>

#include <vector>

struct SizeCheck {
    double diameterPx = 0.0;          // candidate diameter in search pixels
    std::vector<int> compatibleTypes; // indexes into the reference coins
};

SizeCheck checkDiameter(const CoinCandidate& candidate,
                        const std::vector<ReferenceCoin>& references);

// Prints the accepted diameter range of each reference type.
void printReferenceSizes(const std::vector<ReferenceCoin>& references);

// Prints one row per candidate and how many SIFT comparisons were avoided.
void printSizeChecks(const std::vector<CoinCandidate>& candidates,
                     const std::vector<SizeCheck>& checks,
                     const std::vector<ReferenceCoin>& references);

// Green circle = kept (with its compatible types), red = discarded.
cv::Mat drawSizeChecks(const cv::Mat& searchColor, const std::vector<CoinCandidate>& candidates,
                       const std::vector<SizeCheck>& checks,
                       const std::vector<ReferenceCoin>& references);
