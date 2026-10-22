#pragma once

// H3 - Circle detection with the Hough transform.
//
// A detected circle is only a *geometric candidate*: it says "something round
// of this size is here", not which coin it is. Classification into the
// reference types (or unknown) happens in H4-H5, which consume CoinCandidate.

#include <opencv2/core.hpp>

#include <vector>

struct CoinCandidate {
    int id;              // 1-based number, as drawn on the output image
    cv::Point2f center;  // in search-region coordinates
    float radius;        // in pixels
    bool cutByBorder;    // circle extends past the search region (partial coin)
};

// The six numbers cv::HoughCircles needs (see params.hpp for their meaning).
struct HoughSettings {
    double dp;
    double minDist;
    double param1;
    double param2;
    int minRadius;
    int maxRadius;
};

// Settings for the still image (the H3 values in params.hpp).
HoughSettings imageHoughSettings();

// Runs cv::HoughCircles on the smoothed grayscale image. Each result is
// (x, y, radius). Without settings, the still-image values are used.
std::vector<cv::Vec3f> detectCircles(const cv::Mat& smoothedGray, const HoughSettings& settings);
std::vector<cv::Vec3f> detectCircles(const cv::Mat& smoothedGray);

// Fraction of the circle's rim (sampled every 4 degrees) that lies on an edge
// of `edges` (a Canny map), counting only the part of the rim inside the image.
// A real coin outline is mostly edge; a "phantom" circle voted together from
// unrelated edges is not.
double rimSupport(const cv::Mat& edges, const cv::Vec3f& circle);

// Keeps one circle per coin (see DUPLICATE_CENTER_FRACTION) and numbers them.
// `verbose` prints how many duplicates were removed (off in the live loop).
std::vector<CoinCandidate> selectCandidates(const std::vector<cv::Vec3f>& circles,
                                            const cv::Size& imageSize, bool verbose = true);

// Draws every candidate (circle, center and id) on a copy of the color image.
cv::Mat drawCandidates(const cv::Mat& colorImage,
                       const std::vector<CoinCandidate>& candidates);

// Prints a table of candidates. `offset` converts search-region coordinates
// into full-image coordinates (the top-left corner of the search region).
void printCandidates(const std::vector<CoinCandidate>& candidates, const cv::Point& offset);
