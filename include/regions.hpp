#pragma once

// H1 - Regions of interest (ROI): the search area, the reference area and the
// reference coins found inside it.

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct NamedRegion {
    std::string name;
    cv::Rect rect;
};

// One coin type, taken from the reference rectangle.
struct ReferenceCoin {
    std::string name;  // label from params::REFERENCE_NAMES (or "type_k")
    cv::Rect outline;  // tight box around the coin, full-image coordinates
    cv::Mat crop;      // colour pixels inside `outline` (independent copy)
    cv::Mat mask;      // 255 = pixel of this coin inside `crop`
};

// True if the rectangle is non-empty and lies completely inside the image.
bool isInsideImage(const cv::Rect& rect, const cv::Size& imageSize);

// Returns an independent copy of `image(rect)`, or an empty cv::Mat (with an
// error message) if the rectangle is not fully inside the image.
cv::Mat extractRegion(const cv::Mat& image, const NamedRegion& region);

// Finds every coin on the white background of the reference region and
// returns them in reading order (top row left -> right, then the next row).
std::vector<ReferenceCoin> findReferenceCoins(const cv::Mat& image,
                                              const cv::Rect& referenceRegion);

// Sorts coins in reading order (top row left -> right, then the next row).
void sortReadingOrder(std::vector<ReferenceCoin>& coins);

// Names the coins with params::REFERENCE_NAMES (in their current order), or
// "type_1", "type_2", ... if the count differs. Returns false in that case.
bool assignReferenceNames(std::vector<ReferenceCoin>& coins);

// Draws every region on a copy of the image so the coordinates can be checked.
cv::Mat drawRegionsOverview(const cv::Mat& image, const std::vector<NamedRegion>& regions);
