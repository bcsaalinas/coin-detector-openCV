#pragma once

// H4b - Normalised coin patches.
//
// SIFT is invariant to rotation and (approximately) to scale, but it is NOT
// invariant to a non-uniform stretch, and the reference coins are stretched
// vertically by about 20 %. Every coin is therefore resampled into a square
// patch where it is a centred circle of PATCH_DIAMETER pixels.

#include "circle_detection.hpp"
#include "regions.hpp"

#include <opencv2/core.hpp>

#include <vector>

struct CoinPatch {
    cv::Mat gray;  // PATCH_DIAMETER x PATCH_DIAMETER, coin centred
    cv::Mat mask;  // 255 where features may be detected (inside the coin)
};

// Resizes the reference coin's outline box with independent x / y factors so
// the (stretched) coin becomes a circle. `stretch` receives height / width.
CoinPatch makeReferencePatch(const ReferenceCoin& reference, double& stretch);

// Cuts the square around a Hough circle and resizes it. Parts of the square
// outside the search region are excluded from the mask (partial coins).
CoinPatch makeCandidatePatch(const cv::Mat& searchGray, const CoinCandidate& candidate);

// Places patches side by side (masked area only) for a visual check.
cv::Mat makePatchStrip(const std::vector<CoinPatch>& patches);
