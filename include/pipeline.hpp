#pragma once

// One function per milestone of PLAN_PROYECTO_CONTEO_MONEDAS.md. Each stage
// prints its parameters and results, saves its images in options.outputDir
// and returns false if something failed (the error is already printed).

#include "circle_detection.hpp"
#include "coin_matching.hpp"
#include "regions.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct Options {
    std::string imagePath;
    std::string groundTruthPath;
    std::string outputDir;
    bool showWindows = false;
};

// H0 - load the image and check its size; create the output folder.
bool runH0LoadImage(const Options& options, cv::Mat& original);

// H1 - cut the search region and find the coins of the reference region.
bool runH1Regions(const Options& options, const cv::Mat& original, cv::Mat& searchRegion,
                  std::vector<ReferenceCoin>& references);

// H2 - grayscale -> Gaussian smoothing -> Canny edges (search region only).
bool runH2Preprocessing(const Options& options, const cv::Mat& searchRegion, cv::Mat& smoothed);

// H3 - Hough circles -> geometric coin candidates.
bool runH3Hough(const Options& options, const cv::Mat& searchRegion, const cv::Mat& smoothed,
                std::vector<CoinCandidate>& candidates);

// H4 + H5 - diameter pre-filter, SIFT features, matching and classification.
bool runH4H5Recognition(const Options& options, const cv::Mat& searchRegion,
                        const std::vector<ReferenceCoin>& references,
                        const std::vector<CoinCandidate>& candidates,
                        std::vector<Classification>& results);

// H6 - counts per type, final image and evaluation against the ground truth.
bool runH6Report(const Options& options, const cv::Mat& searchRegion,
                 const std::vector<ReferenceCoin>& references,
                 const std::vector<CoinCandidate>& candidates,
                 const std::vector<Classification>& results);
