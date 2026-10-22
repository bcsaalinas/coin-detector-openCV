#pragma once

// H7 - Finding the printed paper in a camera frame.
//
// The sheet has two printed rectangles. In every frame we
//   1. find dark four-sided outlines (Canny + contours + polygon fit),
//   2. keep the two biggest outlines that do not overlap (the edge of the
//      sheet itself contains both, so it is never chosen),
//   3. compute ONE homography that turns the sheet into a top-down view, so
//      both rectangles get the same scale and circles stay circles.

#include <opencv2/core.hpp>

#include <vector>

struct PaperView {
    bool found = false;
    std::vector<cv::Point2f> searchCorners;     // frame pixels, tl tr br bl
    std::vector<cv::Point2f> referenceCorners;  // frame pixels, tl tr br bl
    cv::Mat searchImage;     // top-down view of the search rectangle (BGR)
    cv::Mat referenceImage;  // top-down view of the reference rectangle (BGR)
    cv::Mat searchToFrame;   // 3x3: search-image pixel -> frame pixel
    cv::Mat referenceToFrame;
};

// Looks for the two rectangles in a camera frame. `found` is false when they
// are not both visible.
PaperView findPaper(const cv::Mat& frame);

// Maps points with a 3x3 homography (e.g. searchToFrame).
std::vector<cv::Point2f> mapPoints(const cv::Mat& homography,
                                   const std::vector<cv::Point2f>& points);
