#pragma once

// H6 - Counting, final image and evaluation against a manual ground truth.

#include "circle_detection.hpp"
#include "coin_matching.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct GroundTruthCoin {
    std::string type;      // name of a reference coin type
    cv::Point2f position;  // in search-region coordinates
    std::string note;
};

// Reads "type,x,y,note" lines ('#' starts a comment). Returns an empty list
// (and prints a warning) if the file does not exist.
std::vector<GroundTruthCoin> loadGroundTruth(const std::string& path);

// Final table: how many coins of each reference type were recognised. Each candidate
// is one physical coin (duplicates were removed in H3), so it counts once.
void printCounts(const std::vector<Classification>& results,
                 const std::vector<std::string>& names);

// Every candidate with its label and best SIFT match percentage.
cv::Mat drawFinalResult(const cv::Mat& searchColor,
                        const std::vector<CoinCandidate>& candidates,
                        const std::vector<Classification>& results,
                        const std::vector<std::string>& names);

// Precision / recall of the recognised coins, separating detection errors
// (no Hough circle on the coin) from classification errors (wrong label).
void evaluateAgainstGroundTruth(const std::vector<CoinCandidate>& candidates,
                                const std::vector<Classification>& results,
                                const std::vector<GroundTruthCoin>& groundTruth,
                                const std::vector<std::string>& names);
