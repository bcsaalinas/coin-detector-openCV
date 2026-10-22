#pragma once

// H5 - Matching SIFT descriptors and classifying candidates.
//
// For every candidate that passed the diameter pre-filter, and only against
// the compatible coin types:
//   1. nearest-neighbour search of descriptors (Euclidean / L2 distance),
//   2. Lowe's ratio test + one match per reference keypoint,
//   3. RANSAC similarity transform + geometric sanity checks,
//   4. accept the type if enough matches agree (MIN_INLIERS).

#include "params.hpp"
#include "sift_features.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct MatchResult {
    int type = -1;             // index into the reference coins
    int goodMatches = 0;       // matches that passed the ratio test
    int inliers = 0;           // matches consistent with a valid transform
    double matchPercent = 0.0; // 100 * inliers / keypoints of the smaller set
    double scale = 0.0;        // scale of the RANSAC transform (ideal: 1)
    bool geometryOk = false;   // transform is a plausible coin rotation
    std::vector<cv::DMatch> inlierMatches;  // query = candidate, train = reference
};

// What the cheap pre-filters (diameter, then colour) decided for a candidate.
struct Prefilter {
    std::vector<int> types;      // reference types still compatible -> SIFT
    bool colorDecisive = false;  // one type left and no other reference has its colour
    std::string rejectedBy;      // "" if kept, otherwise "size" or "color"
};

struct Classification {
    std::string label;         // a reference name, or unknown / ambiguous /
                               // size-rejected / color-rejected
    int type = -1;             // index into the reference coins, -1 otherwise
    std::string evidence;      // "SIFT" or "size+color" when a type is given
    MatchResult best;          // best SIFT comparison (empty if rejected)
    std::vector<MatchResult> comparisons;  // one per compatible coin type
};

// Matches candidate descriptors against one reference and verifies geometry.
MatchResult matchToReference(const CoinFeatures& candidate, const CoinFeatures& reference,
                             int type, double loweRatio = params::LOWE_RATIO);

// Decides the label of one candidate (see the rules in coin_matching.cpp).
// `names` are the reference coin names, used as labels.
Classification classifyCandidate(const CoinFeatures& candidate,
                                 const std::vector<CoinFeatures>& references,
                                 const Prefilter& prefilter,
                                 const std::vector<std::string>& names,
                                 int minInliers = params::MIN_INLIERS,
                                 double loweRatio = params::LOWE_RATIO);

// Side-by-side view of candidate and reference with the inlier matches.
cv::Mat drawMatchResult(const CoinPatch& candidatePatch, const CoinFeatures& candidate,
                        const CoinPatch& referencePatch, const CoinFeatures& reference,
                        const MatchResult& result);
