// Coin detection and classification.
//
// Pipeline (one function per milestone, see pipeline.cpp):
//   H0  load the input image and check it
//   H1  cut the search region; find the reference coins in the reference region
//   H2  grayscale -> Gaussian smoothing -> Canny edges
//   H3  Hough circle transform -> geometric coin candidates
//   H4  diameter + colour pre-filters, then SIFT keypoints + descriptors
//   H5  descriptor matching (ratio test + RANSAC) -> reference type / unknown
//   H6  count per type, final image, precision/recall against the ground truth

#include "pipeline.hpp"

#include <opencv2/core.hpp>

#include <iostream>
#include <string>
#include <vector>

#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR "."
#endif

static void printUsage(const char* program)
{
    std::cout << "Usage: " << program
              << " [--image <path>] [--truth <csv>] [--output <dir>] [--show]\n"
              << "  --image   input image (default: assets/coins.png)\n"
              << "  --truth   manual ground truth (default: assets/ground_truth.csv)\n"
              << "  --output  folder for generated images (default: output/)\n"
              << "  --show    also display each stage in a window\n";
}

// Returns false if the program should stop (help requested or bad argument).
static bool parseArguments(int argc, char** argv, Options& options)
{
    const std::string root = PROJECT_ROOT_DIR;
    options.imagePath = root + "/assets/coins.png";
    options.groundTruthPath = root + "/assets/ground_truth.csv";
    options.outputDir = root + "/output";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) {
            options.imagePath = argv[++i];
        } else if (arg == "--truth" && i + 1 < argc) {
            options.groundTruthPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            options.outputDir = argv[++i];
        } else if (arg == "--show") {
            options.showWindows = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "[error] Unknown or incomplete argument: " << arg << "\n";
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    Options options;
    if (!parseArguments(argc, argv, options)) {
        return 1;
    }
    std::cout << "OpenCV " << CV_VERSION << "\n"
              << "Input : " << options.imagePath << "\n"
              << "Output: " << options.outputDir << "\n\n";

    cv::Mat original;
    cv::Mat searchRegion;
    std::vector<ReferenceCoin> references;
    cv::Mat smoothed;
    std::vector<CoinCandidate> candidates;
    std::vector<Classification> results;

    const bool ok =
        runH0LoadImage(options, original) &&
        runH1Regions(options, original, searchRegion, references) &&
        runH2Preprocessing(options, searchRegion, smoothed) &&
        runH3Hough(options, searchRegion, smoothed, candidates) &&
        runH4H5Recognition(options, searchRegion, references, candidates, results) &&
        runH6Report(options, searchRegion, references, candidates, results);
    if (!ok) {
        return 1;
    }

    std::cout << "\nDone.\n";
    return 0;
}
