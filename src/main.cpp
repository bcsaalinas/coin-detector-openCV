// Coin detection and classification - milestones H0 to H3.
//
// Pipeline implemented so far:
//   H0  load the input image and check it
//   H1  cut the search region and the four reference coins
//   H2  grayscale -> Gaussian smoothing -> Canny edges
//   H3  Hough circle transform -> geometric coin candidates
//
// SIFT matching and classification (H4-H6) are intentionally NOT implemented.

#include "circle_detection.hpp"
#include "image_io.hpp"
#include "params.hpp"
#include "preprocessing.hpp"
#include "regions.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <vector>

#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR "."
#endif

struct Options {
    std::string imagePath = std::string(PROJECT_ROOT_DIR) + "/assets/coins.png";
    std::string outputDir = std::string(PROJECT_ROOT_DIR) + "/output";
    bool showWindows = false;
};

static void printUsage(const char* program)
{
    std::cout << "Usage: " << program << " [--image <path>] [--output <dir>] [--show]\n"
              << "  --image   input image (default: assets/coins.png)\n"
              << "  --output  folder for generated images (default: output/)\n"
              << "  --show    also display each stage in a window\n";
}

// Returns false if the program should stop (help requested or bad argument).
static bool parseArguments(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) {
            options.imagePath = argv[++i];
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

// The ROI coordinates are only valid for the resolution they were measured on.
static bool hasExpectedSize(const cv::Mat& image)
{
    if (image.cols == params::EXPECTED_IMAGE_WIDTH &&
        image.rows == params::EXPECTED_IMAGE_HEIGHT) {
        return true;
    }
    std::cerr << "[error] Expected a " << params::EXPECTED_IMAGE_WIDTH << " x "
              << params::EXPECTED_IMAGE_HEIGHT << " image but got " << image.cols
              << " x " << image.rows << ".\n"
              << "        The ROI rectangles in include/params.hpp were measured on the\n"
              << "        expected resolution; measure them again for this image.\n";
    return false;
}

// Builds names such as "02_reference_penny.png" so files sort in pipeline order.
static std::string numberedFileName(int number, const std::string& label)
{
    const std::string prefix = number < 10 ? "0" : "";
    return prefix + std::to_string(number) + "_" + label + ".png";
}

static void waitForKey()
{
    std::cout << "  (press any key in an image window to continue)\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
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

    // ---------------------------------------------------------------- H0 --
    std::cout << "[H0] Loading image\n";
    const cv::Mat original = loadColorImage(options.imagePath);
    if (original.empty()) {
        return 1;
    }
    printImageInfo("  original", original);
    if (!hasExpectedSize(original)) {
        return 1;
    }
    if (!ensureDirectory(options.outputDir)) {
        return 1;
    }

    if (options.showWindows) {
        showImage("H0 - original", original);
        waitForKey();
    }

    // ---------------------------------------------------------------- H1 --
    std::cout << "\n[H1] Extracting regions of interest\n";
    const NamedRegion search{"search region", params::SEARCH_REGION};
    const std::vector<NamedRegion> references = referenceRegions();

    const cv::Mat searchRegion = extractRegion(original, search);
    if (searchRegion.empty()) {
        return 1;
    }
    printImageInfo("  search region", searchRegion);

    std::vector<cv::Mat> referenceCoins;
    for (const NamedRegion& reference : references) {
        cv::Mat coin = extractRegion(original, reference);
        if (coin.empty()) {
            return 1;
        }
        printImageInfo("  reference " + reference.name, coin);
        referenceCoins.push_back(coin);
    }

    std::vector<NamedRegion> allRegions = references;
    allRegions.insert(allRegions.begin(), search);
    bool saved = saveImage(options.outputDir, "00_roi_overview.png",
                           drawRegionsOverview(original, allRegions));
    saved = saveImage(options.outputDir, "01_search_region.png", searchRegion) && saved;
    for (size_t i = 0; i < references.size(); ++i) {
        const std::string fileName =
            numberedFileName(2 + static_cast<int>(i), "reference_" + references[i].name);
        saved = saveImage(options.outputDir, fileName, referenceCoins[i]) && saved;
    }
    if (!saved) {
        return 1;
    }

    if (options.showWindows) {
        showImage("H1 - search region", searchRegion);
        for (size_t i = 0; i < references.size(); ++i) {
            showImage("H1 - " + references[i].name, referenceCoins[i]);
        }
        waitForKey();
    }

    // ---------------------------------------------------------------- H2 --
    // Only the search region is preprocessed; the references are left for H4.
    std::cout << "\n[H2] Grayscale -> Gaussian smoothing -> Canny\n"
              << "  Gaussian kernel " << params::GAUSSIAN_KERNEL_SIZE << "x"
              << params::GAUSSIAN_KERNEL_SIZE << ", sigma " << params::GAUSSIAN_SIGMA
              << "; Canny thresholds low " << params::CANNY_LOW_THRESHOLD << " / high "
              << params::CANNY_HIGH_THRESHOLD << "\n";

    const cv::Mat gray = toGrayscale(searchRegion);
    const cv::Mat smoothed = smoothGaussian(gray);
    const cv::Mat edges = detectEdgesCanny(smoothed);

    const double edgePercent = 100.0 * cv::countNonZero(edges) / edges.total();
    std::cout << "  edge pixels: " << cv::countNonZero(edges) << " ("
              << edgePercent << "% of the search region)\n";

    saved = saveImage(options.outputDir, "06_grayscale.png", gray);
    saved = saveImage(options.outputDir, "07_gaussian.png", smoothed) && saved;
    saved = saveImage(options.outputDir, "08_canny_edges.png", edges) && saved;
    saved = saveImage(options.outputDir, "10_preprocessing_comparison.png",
                      makeComparisonStrip(searchRegion, gray, smoothed, edges, 0.5)) &&
            saved;
    if (!saved) {
        return 1;
    }

    if (options.showWindows) {
        showImage("H2 - grayscale", gray);
        showImage("H2 - gaussian", smoothed);
        showImage("H2 - canny", edges);
        waitForKey();
    }

    // ---------------------------------------------------------------- H3 --
    std::cout << "\n[H3] Hough circle transform\n"
              << "  dp " << params::HOUGH_DP << ", minDist " << params::HOUGH_MIN_DIST
              << ", param1 " << params::HOUGH_PARAM1 << ", param2 " << params::HOUGH_PARAM2
              << ", radius [" << params::HOUGH_MIN_RADIUS << ", "
              << params::HOUGH_MAX_RADIUS << "]\n";

    const std::vector<cv::Vec3f> circles = detectCircles(smoothed);
    const std::vector<CoinCandidate> candidates =
        selectCandidates(circles, smoothed.size());
    printCandidates(candidates, params::SEARCH_REGION.tl());

    const cv::Mat detections = drawCandidates(searchRegion, candidates);
    if (!saveImage(options.outputDir, "09_hough_circles.png", detections)) {
        return 1;
    }

    // A circle is a geometric candidate, not a recognised US coin. Counting by
    // denomination (penny / nickel / dime / quarter) starts in H4-H6.
    std::cout << "\n  Geometric coin candidates: " << candidates.size() << "\n";

    if (options.showWindows) {
        showImage("H3 - Hough candidates", detections);
        waitForKey();
    }

    std::cout << "\nDone.\n";
    return 0;
}
