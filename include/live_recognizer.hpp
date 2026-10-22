#pragma once

// H7 - Recognising coins in one camera frame (after findPaper).
//
//   reference rectangle --Hough--> reference coins --SIFT--> cached features
//   search rectangle    --Hough--> candidates --diameter filter--> SIFT match
//
// Only the reference rectangle defines what a coin type is. The decision is
// Hough + SIFT (ratio test + RANSAC); size (and optionally colour) only skips
// comparisons that cannot match.

#include "circle_detection.hpp"
#include "coin_matching.hpp"
#include "coin_patch.hpp"
#include "paper_view.hpp"
#include "params.hpp"
#include "regions.hpp"
#include "sift_features.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

// Values that can be changed while the program runs (trackbars). They start
// with the LIVE_* values of params.hpp.
struct LiveSettings {
    double cannyHigh = params::LIVE_HOUGH_PARAM1;   // Canny high = Hough param1
    double votes = params::LIVE_HOUGH_PARAM2;       // Hough param2
    double referenceRimSupport = params::LIVE_REFERENCE_MIN_RIM_SUPPORT;
    double searchRimSupport = params::LIVE_SEARCH_MIN_RIM_SUPPORT;
    int minInliers = params::LIVE_MIN_INLIERS;      // SIFT acceptance threshold
    double loweRatio = params::LIVE_LOWE_RATIO;     // ratio test
    // Confidence denominator. Live camera: always LIVE_VOTE_FRAMES, so a new
    // coin needs several frames of evidence. Single photo: only the frames
    // actually seen (1), otherwise a photo could never reach the threshold.
    bool confidenceOverSeenFrames = false;
};

// A search coin followed over time (positions in the top-down search view).
struct TrackedCoin {
    cv::Point2f center;
    float radius = 0.0f;
    std::vector<int> recentTypes;  // last decisions, -1 = not recognised
    int framesMissing = 0;
    int framesSeen = 0;             // since the coin was first seen
    std::vector<int> timesRecognised;  // per reference type, since first seen
};

// What is kept between frames so the reference features are not recomputed
// every time (SIFT is the expensive part) and labels do not flicker.
struct LiveMemory {
    std::vector<ReferenceCoin> references;   // reference-image coordinates
    std::vector<CoinPatch> referencePatches;
    std::vector<CoinFeatures> referenceFeatures;
    bool forceRelearn = false;               // set by the 'r' key
    int changedFrames = 0;                   // consecutive frames with other refs
    std::vector<TrackedCoin> tracks;
};

struct LiveResult {
    std::vector<CoinCandidate> candidates;   // search-image coordinates
    std::vector<Classification> results;     // one per candidate (this frame)
    std::vector<int> stableTypes;            // shown type (-1 if confidence too low)
    std::vector<int> bestGuess;              // most recognised type, even if low
    std::vector<double> confidence;          // share of recent frames (0..1)
    bool referencesRelearned = false;
    int siftComparisons = 0;                 // after the diameter filter
};

// Hough settings for the reference rectangle of the top-down view (radius
// range relative to its width).
HoughSettings liveHoughSettings(const LiveSettings& settings);

// Processes one frame whose paper was found.
LiveResult recognizeFrame(const PaperView& view, const LiveSettings& settings,
                          LiveMemory& memory);

// One line per tracked coin: frames seen and how often each type was
// recognised (useful to judge a real camera setup with --no-gui).
void printTrackReport(const LiveMemory& memory);

// Names of the current reference types, in order.
std::vector<std::string> referenceNames(const LiveMemory& memory);

// Draws rectangles, reference coins, candidates with labels, counts and FPS
// on a copy of the camera frame.
cv::Mat drawLiveOverlay(const cv::Mat& frame, const PaperView& view, const LiveMemory& memory,
                        const LiveResult& result, double fps);
