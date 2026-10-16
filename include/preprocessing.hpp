#pragma once

// H2 - Preprocessing: grayscale -> Gaussian smoothing -> Canny edges.

#include <opencv2/core.hpp>

// BGR color image -> single-channel intensity image.
cv::Mat toGrayscale(const cv::Mat& colorImage);

// Gaussian smoothing with GAUSSIAN_KERNEL_SIZE / GAUSSIAN_SIGMA from params.hpp.
cv::Mat smoothGaussian(const cv::Mat& grayImage);

// Canny edge map (255 = edge, 0 = background) with the thresholds in params.hpp.
cv::Mat detectEdgesCanny(const cv::Mat& smoothedImage);

// Side-by-side strip: color | grayscale | smoothed | edges, scaled by `scale`.
cv::Mat makeComparisonStrip(const cv::Mat& color, const cv::Mat& gray,
                            const cv::Mat& smoothed, const cv::Mat& edges,
                            double scale);
