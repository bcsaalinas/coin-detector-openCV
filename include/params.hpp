#pragma once

// All tunable values of the pipeline live in this file, so an experiment only
// needs to change one place. Record every change in the project log
// (PLAN_PROYECTO_CONTEO_MONEDAS.md) together with the observed result.

#include <opencv2/core.hpp>

namespace params {

// ---------------------------------------------------------------------------
// H0 - Input image
// ---------------------------------------------------------------------------
// The ROI coordinates below were measured on this exact resolution. If a
// different version of the image is used they must be measured again.
constexpr int EXPECTED_IMAGE_WIDTH = 2477;
constexpr int EXPECTED_IMAGE_HEIGHT = 1929;

// ---------------------------------------------------------------------------
// H1 - Regions of interest: cv::Rect(x, y, width, height) in full-image pixels
// ---------------------------------------------------------------------------
// Search region: the wooden photo on the left, just inside its dark frame
// (frame columns 92-98 / 1169-1175, frame row 126-127, photo ends at row 1722).
// These match the plan's (86, 106, 880, 1317) scaled by 2477 / 2048 = 1.2095.
const cv::Rect SEARCH_REGION(104, 128, 1064, 1595);

// Reference coins on the white panel. Each box encloses one whole coin with a
// margin of about 5 px. The plan's rectangles (scaled by 1.2095) cut off the
// bottom ~20 px of the dime, so all four were re-measured on this image.
// The quarter box starts at x = 2042 (1 px margin) so that the nickel's rim,
// which reaches x = 2041 at the top-left corner, stays out of the crop.
const cv::Rect PENNY_REGION(1588, 263, 219, 264);
const cv::Rect NICKEL_REGION(1829, 253, 245, 297);
const cv::Rect DIME_REGION(1757, 547, 250, 297);
const cv::Rect QUARTER_REGION(2042, 497, 332, 397);

// ---------------------------------------------------------------------------
// H2 - Grayscale, Gaussian smoothing and Canny
// ---------------------------------------------------------------------------
// Gaussian low-pass filter. Larger sigma removes more wood-grain texture but
// also blurs the coin rims. Kernel size must be odd (about 4.5 * sigma here).
// Plan start (5x5, sigma 1.5) kept too much wood grain and coin engraving.
constexpr int GAUSSIAN_KERNEL_SIZE = 9;
constexpr double GAUSSIAN_SIGMA = 2.0;

// Canny hysteresis: gradients above HIGH start an edge, gradients above LOW
// continue it. LOW = HIGH / 2 on purpose: HoughCircles (HOUGH_GRADIENT) runs
// Canny(param1 / 2, param1) internally, so with HOUGH_PARAM1 = CANNY_HIGH the
// saved edge map is exactly the one the Hough transform votes with.
constexpr double CANNY_HIGH_THRESHOLD = 120.0;
constexpr double CANNY_LOW_THRESHOLD = CANNY_HIGH_THRESHOLD / 2.0;

// ---------------------------------------------------------------------------
// H3 - Hough transform for circles (cv::HOUGH_GRADIENT)
// ---------------------------------------------------------------------------
// dp: inverse resolution of the center accumulator (1 = same as the image,
//     1.2 = slightly coarser cells, so votes of a rim gather more easily).
//     dp >= 1.5 produced dozens of false circles on this image.
constexpr double HOUGH_DP = 1.2;

// Minimum distance between two detected centers, in pixels. Two different
// coins cannot have centers closer than roughly the smallest radius.
// 60 px produced duplicate circles on the same coin; 80-150 px gave none.
constexpr double HOUGH_MIN_DIST = 100.0;

// param1: high threshold of the internal Canny (low = param1 / 2).
constexpr double HOUGH_PARAM1 = CANNY_HIGH_THRESHOLD;

// param2: minimum number of accumulator votes for a center. Lower values find
// more (partial) coins but also circles in the wood grain.
// Tested on this image: <= 32 -> false positives, 35 -> 18 circles,
// 40 -> 17 circles, 45-50 -> 15 circles, 55-60 -> 14 circles (no false
// positives from 33 upwards). 40 keeps a safety margin from the noisy range.
constexpr double HOUGH_PARAM2 = 40.0;

// Radius range in pixels. Coins in the search region measure ~115-200 px.
constexpr int HOUGH_MIN_RADIUS = 95;
constexpr int HOUGH_MAX_RADIUS = 220;

// Explicit duplicate check after Hough: a circle whose center lies closer than
// this fraction of the smaller radius to a stronger circle is the same coin.
constexpr double DUPLICATE_CENTER_FRACTION = 0.5;

} // namespace params
