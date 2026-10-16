#pragma once

// H1 - Regions of interest (ROI): the search area and the reference coins.

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct NamedRegion {
    std::string name;
    cv::Rect rect;
};

// The four reference coins, in the order penny, nickel, dime, quarter.
std::vector<NamedRegion> referenceRegions();

// True if the rectangle is non-empty and lies completely inside the image.
bool isInsideImage(const cv::Rect& rect, const cv::Size& imageSize);

// Returns an independent copy of `image(rect)`, or an empty cv::Mat (with an
// error message) if the rectangle is not fully inside the image.
cv::Mat extractRegion(const cv::Mat& image, const NamedRegion& region);

// Draws every region on a copy of the image so the coordinates can be checked.
cv::Mat drawRegionsOverview(const cv::Mat& image, const std::vector<NamedRegion>& regions);
