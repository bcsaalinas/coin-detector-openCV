#pragma once

// H3 - Circle detection with the Hough transform.
//
// A detected circle is only a *geometric candidate*: it says "something round
// of this size is here", not which coin it is. Classification (penny, nickel,
// dime, quarter, unknown) belongs to H4-H5 and will consume CoinCandidate.

#include <opencv2/core.hpp>

#include <vector>

struct CoinCandidate {
    int id;              // 1-based number, as drawn on the output image
    cv::Point2f center;  // in search-region coordinates
    float radius;        // in pixels
    bool cutByBorder;    // circle extends past the search region (partial coin)
};

// Runs cv::HoughCircles on the smoothed grayscale image with the parameters in
// params.hpp. Each result is (x, y, radius).
std::vector<cv::Vec3f> detectCircles(const cv::Mat& smoothedGray);

// Keeps one circle per coin (see DUPLICATE_CENTER_FRACTION) and numbers them.
std::vector<CoinCandidate> selectCandidates(const std::vector<cv::Vec3f>& circles,
                                            const cv::Size& imageSize);

// Draws every candidate (circle, center and id) on a copy of the color image.
cv::Mat drawCandidates(const cv::Mat& colorImage,
                       const std::vector<CoinCandidate>& candidates);

// Prints a table of candidates. `offset` converts search-region coordinates
// into full-image coordinates (the top-left corner of the search region).
void printCandidates(const std::vector<CoinCandidate>& candidates, const cv::Point& offset);
