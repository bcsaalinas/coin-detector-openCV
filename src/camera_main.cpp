// Live coin recognition with a camera (milestone H7).
//
// Put a sheet with two printed rectangles in front of the camera: the
// reference coins go in the smaller rectangle, the coins to identify in the
// bigger one. Every frame:
//   findPaper      -> both rectangles, warped to one top-down view
//   recognizeFrame -> Hough in both, diameter filter, SIFT + RANSAC
//   drawLiveOverlay
//
// Keys: q / Esc = quit, r = re-learn the reference coins, s = save a snapshot,
//       v = show / hide the top-down views. Sliders tune Hough while it runs.

#include "image_io.hpp"
#include "live_recognizer.hpp"
#include "paper_view.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR "."
#endif

struct CameraOptions {
    int camera = 0;
    std::string videoPath;   // read frames from a video file instead
    std::string imagePath;   // process a single photo instead
    std::string outputDir = std::string(PROJECT_ROOT_DIR) + "/output";
    bool showWindow = true;
    int maxFrames = 0;       // 0 = until the stream ends or q is pressed
};

static void printUsage(const char* program)
{
    std::cout << "Usage: " << program
              << " [--camera <index> | --video <file> | --image <file>]\n"
              << "       [--output <dir>] [--frames <n>] [--no-gui]\n"
              << "  --camera  camera index (default 0)\n"
              << "  --video   read frames from a video file\n"
              << "  --image   process one photo and save the result\n"
              << "  --frames  stop after n frames\n"
              << "  --no-gui  no window; print results and save the last frame\n";
}

// Sliders (like the course's InteractiveDetector) to tune the live Hough
// detection on your own camera. Values are integers; percentages for supports.
static const char* CONTROLS = "coin_camera controls";

static void createControls(const LiveSettings& settings)
{
    cv::namedWindow(CONTROLS, cv::WINDOW_NORMAL);
    cv::resizeWindow(CONTROLS, 480, 280);
    cv::createTrackbar("Canny high", CONTROLS, nullptr, 200);
    cv::createTrackbar("Hough votes", CONTROLS, nullptr, 100);
    cv::createTrackbar("Ref rim %", CONTROLS, nullptr, 100);
    cv::createTrackbar("Search rim %", CONTROLS, nullptr, 100);
    cv::createTrackbar("Min inliers", CONTROLS, nullptr, 50);
    cv::createTrackbar("Lowe ratio %", CONTROLS, nullptr, 99);
    cv::setTrackbarPos("Canny high", CONTROLS, cvRound(settings.cannyHigh));
    cv::setTrackbarPos("Hough votes", CONTROLS, cvRound(settings.votes));
    cv::setTrackbarPos("Ref rim %", CONTROLS, cvRound(100 * settings.referenceRimSupport));
    cv::setTrackbarPos("Search rim %", CONTROLS, cvRound(100 * settings.searchRimSupport));
    cv::setTrackbarPos("Min inliers", CONTROLS, settings.minInliers);
    cv::setTrackbarPos("Lowe ratio %", CONTROLS, cvRound(100 * settings.loweRatio));
}

static void readControls(LiveSettings& settings)
{
    settings.cannyHigh = std::max(1, cv::getTrackbarPos("Canny high", CONTROLS));
    settings.votes = std::max(1, cv::getTrackbarPos("Hough votes", CONTROLS));
    settings.referenceRimSupport = cv::getTrackbarPos("Ref rim %", CONTROLS) / 100.0;
    settings.searchRimSupport = cv::getTrackbarPos("Search rim %", CONTROLS) / 100.0;
    settings.minInliers = std::max(1, cv::getTrackbarPos("Min inliers", CONTROLS));
    settings.loweRatio = std::max(50, cv::getTrackbarPos("Lowe ratio %", CONTROLS)) / 100.0;
}

static bool parseArguments(int argc, char** argv, CameraOptions& options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--camera" && hasValue) {
            options.camera = std::stoi(argv[++i]);
        } else if (arg == "--video" && hasValue) {
            options.videoPath = argv[++i];
        } else if (arg == "--image" && hasValue) {
            options.imagePath = argv[++i];
        } else if (arg == "--output" && hasValue) {
            options.outputDir = argv[++i];
        } else if (arg == "--frames" && hasValue) {
            options.maxFrames = std::stoi(argv[++i]);
        } else if (arg == "--no-gui") {
            options.showWindow = false;
        } else {
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

// One line per frame when something changed (keeps the console readable).
static std::string summaryLine(const PaperView& view, const LiveMemory& memory,
                               const LiveResult& result)
{
    if (!view.found) {
        return "paper not found";
    }
    std::string line = "references " + std::to_string(memory.references.size()) +
                       " | candidates " + std::to_string(result.candidates.size()) + " |";
    std::vector<int> counts(memory.references.size(), 0);
    int unknown = 0;
    for (size_t i = 0; i < result.results.size(); ++i) {
        const int type = result.stableTypes.empty() ? result.results[i].type
                                                    : result.stableTypes[i];
        if (type >= 0 && type < static_cast<int>(counts.size())) {
            ++counts[type];
        } else {
            ++unknown;
        }
    }
    for (size_t k = 0; k < counts.size(); ++k) {
        line += " " + memory.references[k].name + " " + std::to_string(counts[k]);
    }
    return line + " | unknown " + std::to_string(unknown);
}

// Prints every candidate's decision (used for single images).
static void printDecisions(const LiveMemory& memory, const LiveResult& result)
{
    const std::vector<std::string> names = referenceNames(memory);
    std::printf("   id | radius | best type | ratio | inliers | match  | label\n");
    for (size_t i = 0; i < result.candidates.size(); ++i) {
        const Classification& d = result.results[i];
        const bool compared = !d.comparisons.empty();
        std::printf("  %3d | %6.1f | %-9s | %5d | %7d | %5.1f%% | %s\n",
                    result.candidates[i].id, result.candidates[i].radius,
                    compared ? names[d.best.type].c_str() : "-",
                    compared ? d.best.goodMatches : 0, compared ? d.best.inliers : 0,
                    compared ? d.best.matchPercent : 0.0, d.label.c_str());
    }
}

int main(int argc, char** argv)
{
    CameraOptions options;
    if (!parseArguments(argc, argv, options) || !ensureDirectory(options.outputDir)) {
        return 1;
    }

    const bool singleImage = !options.imagePath.empty();
    cv::VideoCapture capture;
    if (!singleImage) {
        if (options.videoPath.empty()) {
            capture.open(options.camera);
        } else {
            capture.open(options.videoPath);
        }
        if (!capture.isOpened()) {
            std::cerr << "[error] Cannot open "
                      << (options.videoPath.empty() ? "camera " + std::to_string(options.camera)
                                                    : options.videoPath)
                      << " (on macOS, allow camera access for the terminal app)\n";
            return 1;
        }
    }

    LiveMemory memory;
    LiveSettings settings;
    settings.confidenceOverSeenFrames = singleImage;  // one photo = one frame of evidence
    if (options.showWindow && !singleImage) {
        createControls(settings);
    }
    bool showViews = false;
    double fps = 0.0;
    int frameCount = 0;
    int snapshotCount = 0;
    std::string lastSummary;
    cv::Mat lastOverlay;

    const bool liveCamera = !singleImage && options.videoPath.empty();
    int emptyFrames = 0;

    while (true) {
        cv::Mat frame;
        if (singleImage) {
            frame = loadColorImage(options.imagePath);
        } else {
            capture >> frame;
        }
        if (frame.empty()) {
            // A live camera often returns a few empty frames right after it is
            // opened (it is still starting) or if it hiccups. Wait a little
            // and try again; give up after ~3 s without any frame. For a video
            // file or a photo, an empty frame means the end.
            if (liveCamera && ++emptyFrames <= 150) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            if (liveCamera) {
                std::cerr << "[error] The camera gave no image for 3 seconds (is another app "
                             "using it, or is camera access denied?)\n";
            }
            break;
        }
        emptyFrames = 0;

        if (options.showWindow && !singleImage) {
            const LiveSettings before = settings;
            readControls(settings);
            // New Hough settings may find other reference coins: re-learn.
            memory.forceRelearn = memory.forceRelearn ||
                                  before.cannyHigh != settings.cannyHigh ||
                                  before.votes != settings.votes ||
                                  before.referenceRimSupport != settings.referenceRimSupport;
        }

        const auto start = std::chrono::steady_clock::now();
        const PaperView view = findPaper(frame);
        LiveResult result;
        if (view.found) {
            result = recognizeFrame(view, settings, memory);
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        // Smoothed frame rate (exponential moving average of 1 / time).
        fps = fps == 0.0 ? 1.0 / seconds : 0.9 * fps + 0.1 / seconds;

        lastOverlay = drawLiveOverlay(frame, view, memory, result, fps);
        ++frameCount;

        const std::string summary = summaryLine(view, memory, result);
        if (summary != lastSummary || result.referencesRelearned) {
            std::cout << "frame " << frameCount << ": " << summary
                      << (result.referencesRelearned ? "  (references re-learned)" : "")
                      << "  [" << cvRound(1000 * seconds) << " ms]\n";
            lastSummary = summary;
        }

        if (singleImage) {
            if (view.found) {
                printDecisions(memory, result);
                saveImage(options.outputDir, "camera_reference_view.png", view.referenceImage);
                saveImage(options.outputDir, "camera_search_view.png", view.searchImage);
            }
            break;
        }
        if (options.showWindow) {
            cv::imshow("coin_camera", lastOverlay);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                break;
            }
            if (key == 'r') {
                memory.forceRelearn = true;
            }
            if (key == 'v') {
                showViews = !showViews;
                if (!showViews) {
                    cv::destroyWindow("top-down search");
                    cv::destroyWindow("top-down reference");
                }
            }
            if (showViews && view.found) {
                cv::imshow("top-down search", view.searchImage);
                cv::imshow("top-down reference", view.referenceImage);
            }
            if (key == 's') {
                saveImage(options.outputDir,
                          "camera_snapshot_" + std::to_string(++snapshotCount) + ".png",
                          lastOverlay);
            }
        }
        if (options.maxFrames > 0 && frameCount >= options.maxFrames) {
            break;
        }
    }

    if (!lastOverlay.empty() && (singleImage || !options.showWindow)) {
        saveImage(options.outputDir, "camera_result.png", lastOverlay);
    }
    if (singleImage && options.showWindow && !lastOverlay.empty()) {
        showImage("coin_camera", lastOverlay);
        cv::waitKey(0);
    }
    if (!singleImage) {
        printTrackReport(memory);
    }
    std::cout << "Processed " << frameCount << " frame(s).\n";
    return frameCount > 0 ? 0 : 1;
}
