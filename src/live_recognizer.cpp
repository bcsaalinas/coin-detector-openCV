#include "live_recognizer.hpp"

#include "color_filter.hpp"
#include "params.hpp"
#include "preprocessing.hpp"
#include "size_filter.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

HoughSettings liveHoughSettings(const LiveSettings& settings)
{
    const int minRadius =
        cvRound(params::LIVE_MIN_RADIUS_FRACTION * params::LIVE_SEARCH_WIDTH);
    const int maxRadius =
        cvRound(params::LIVE_MAX_RADIUS_FRACTION * params::LIVE_SEARCH_WIDTH);
    // minDist = smallest radius: two different coins cannot be closer.
    return {params::HOUGH_DP, static_cast<double>(minRadius), settings.cannyHigh,
            settings.votes, minRadius, maxRadius};
}

// Same H2 + H3 steps as the still image (gray -> Gaussian -> Hough), then
// two checks that remove "phantom" circles:
//   - rim support: most of the circle must lie on Canny edges,
//   - (references only) circles may not overlap: reference coins lie apart,
//     so a weaker circle overlapping a stronger one is not a coin.
static std::vector<CoinCandidate> findCoins(const cv::Mat& color, const HoughSettings& settings,
                                            double minRimSupport, bool mustNotOverlap)
{
    const cv::Mat smoothed = smoothGaussian(toGrayscale(color));
    // The same edges HoughCircles uses internally: Canny(param1 / 2, param1).
    cv::Mat edges;
    cv::Canny(smoothed, edges, settings.param1 / 2.0, settings.param1);

    std::vector<cv::Vec3f> kept;
    for (const cv::Vec3f& circle : detectCircles(smoothed, settings)) {
        if (rimSupport(edges, circle) < minRimSupport) {
            continue;
        }
        bool overlaps = false;
        for (const cv::Vec3f& other : kept) {
            const double centers = std::hypot(circle[0] - other[0], circle[1] - other[1]);
            overlaps = overlaps || centers < circle[2] + other[2];
        }
        if (!(mustNotOverlap && overlaps)) {
            kept.push_back(circle);  // HoughCircles lists the strongest first
        }
    }
    return selectCandidates(kept, smoothed.size(), false);
}

// Search-rectangle Hough settings taken from the reference coins: only radii
// that some reference could have (+/- SIZE_TOLERANCE) are searched for.
static HoughSettings searchHoughSettings(const std::vector<ReferenceCoin>& references,
                                         const LiveSettings& live)
{
    double smallest = 1e9;
    double largest = 0.0;
    for (const ReferenceCoin& reference : references) {
        smallest = std::min(smallest, reference.outline.width / 2.0);
        largest = std::max(largest, reference.outline.width / 2.0);
    }
    HoughSettings settings = liveHoughSettings(live);
    settings.minRadius = cvFloor(smallest * (1.0 - params::SIZE_TOLERANCE));
    settings.maxRadius = cvCeil(largest * (1.0 + params::SIZE_TOLERANCE));
    settings.minDist = settings.minRadius;  // two coins cannot be closer
    return settings;
}

// A whole circle of the reference image becomes a reference coin: square crop
// around it and a circular mask.
static bool coinFromCircle(const cv::Mat& image, const CoinCandidate& circle,
                           ReferenceCoin& coin)
{
    const int r = cvRound(circle.radius);
    const cv::Rect box(cvRound(circle.center.x) - r, cvRound(circle.center.y) - r, 2 * r,
                       2 * r);
    if (!isInsideImage(box, image.size())) {
        return false;  // cut by the border: not usable as a reference
    }
    coin.outline = box;
    coin.crop = image(box).clone();
    coin.mask = cv::Mat::zeros(box.size(), CV_8U);
    cv::circle(coin.mask, cv::Point(r, r), r, cv::Scalar(255), cv::FILLED);
    return true;
}

static cv::Point2f outlineCenter(const ReferenceCoin& coin)
{
    return cv::Point2f(coin.outline.x + coin.outline.width / 2.0f,
                       coin.outline.y + coin.outline.height / 2.0f);
}

// True if the reference coins differ from the ones whose features are cached.
static bool referencesChanged(const std::vector<ReferenceCoin>& now, const LiveMemory& memory)
{
    if (now.size() != memory.references.size()) {
        return true;
    }
    for (size_t k = 0; k < now.size(); ++k) {
        const cv::Point2f d = outlineCenter(now[k]) - outlineCenter(memory.references[k]);
        if (std::hypot(d.x, d.y) > params::LIVE_REFERENCE_MOVE_PX) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> referenceNames(const LiveMemory& memory)
{
    std::vector<std::string> names;
    for (const ReferenceCoin& reference : memory.references) {
        names.push_back(reference.name);
    }
    return names;
}

// Follows each candidate over time and computes its confidence: the share of
// its last LIVE_VOTE_FRAMES frames in which SIFT recognised it as a type.
// The top-down view is fixed to the paper, so the same coin keeps (almost)
// the same position from frame to frame.
static void voteOverTime(LiveResult& result, const LiveSettings& settings, LiveMemory& memory)
{
    for (TrackedCoin& track : memory.tracks) {
        ++track.framesMissing;
    }
    const size_t numTypes = memory.references.size();
    for (size_t i = 0; i < result.candidates.size(); ++i) {
        const CoinCandidate& candidate = result.candidates[i];
        TrackedCoin* track = nullptr;
        for (TrackedCoin& t : memory.tracks) {
            const cv::Point2f d = t.center - candidate.center;
            if (std::hypot(d.x, d.y) < 0.5f * candidate.radius) {
                track = &t;
                break;
            }
        }
        if (track == nullptr) {
            memory.tracks.push_back(TrackedCoin());
            track = &memory.tracks.back();
            track->timesRecognised.assign(numTypes, 0);
        }
        const int type = result.results[i].type;
        track->center = candidate.center;
        track->radius = candidate.radius;
        track->framesMissing = 0;
        ++track->framesSeen;
        if (type >= 0) {
            ++track->timesRecognised[type];
        }
        track->recentTypes.push_back(type);
        if (static_cast<int>(track->recentTypes.size()) > params::LIVE_VOTE_FRAMES) {
            track->recentTypes.erase(track->recentTypes.begin());
        }

        // Most recognised type in the window ("not recognised" never wins).
        int bestType = -1;
        int bestVotes = 0;
        for (size_t k = 0; k < numTypes; ++k) {
            const int votes = static_cast<int>(std::count(
                track->recentTypes.begin(), track->recentTypes.end(), static_cast<int>(k)));
            if (votes > bestVotes) {
                bestVotes = votes;
                bestType = static_cast<int>(k);
            }
        }
        // Frames not seen yet count as "not recognised" (live camera), unless
        // only the frames actually seen should count (single photo).
        const int frames = settings.confidenceOverSeenFrames
                               ? static_cast<int>(track->recentTypes.size())
                               : params::LIVE_VOTE_FRAMES;
        const double confidence = static_cast<double>(bestVotes) / frames;
        result.bestGuess.push_back(bestType);
        result.confidence.push_back(confidence);
        result.stableTypes.push_back(confidence >= params::LIVE_MIN_CONFIDENCE ? bestType : -1);
    }
    // Forget coins that have not been seen for a while.
    memory.tracks.erase(std::remove_if(memory.tracks.begin(), memory.tracks.end(),
                                       [](const TrackedCoin& t) {
                                           return t.framesMissing > params::LIVE_VOTE_FRAMES;
                                       }),
                        memory.tracks.end());
}

void printTrackReport(const LiveMemory& memory)
{
    std::printf("Coins followed at the end (position in the top-down search view):\n");
    for (const TrackedCoin& track : memory.tracks) {
        std::printf("  (%4.0f, %4.0f) r %5.1f  seen %3d frames  recognised:", track.center.x,
                    track.center.y, track.radius, track.framesSeen);
        bool any = false;
        for (size_t k = 0; k < track.timesRecognised.size(); ++k) {
            if (track.timesRecognised[k] > 0) {
                std::printf(" %s %d (%.0f%%)", memory.references[k].name.c_str(),
                            track.timesRecognised[k],
                            100.0 * track.timesRecognised[k] / track.framesSeen);
                any = true;
            }
        }
        std::printf("%s\n", any ? "" : " never");
    }
}

LiveResult recognizeFrame(const PaperView& view, const LiveSettings& settings,
                          LiveMemory& memory)
{
    LiveResult result;

    // ---- Reference rectangle: Hough finds the coins that define the types.
    std::vector<ReferenceCoin> references;
    for (const CoinCandidate& circle :
         findCoins(view.referenceImage, liveHoughSettings(settings),
                   settings.referenceRimSupport, true)) {
        ReferenceCoin coin;
        if (coinFromCircle(view.referenceImage, circle, coin)) {
            references.push_back(coin);
        }
    }
    sortReadingOrder(references);
    assignReferenceNames(references);

    // SIFT on the references only when they changed (cached otherwise), and
    // only when the change is confirmed over several frames.
    const bool firstTime = memory.references.empty() && !references.empty();
    memory.changedFrames = referencesChanged(references, memory) ? memory.changedFrames + 1 : 0;
    if (firstTime || memory.forceRelearn ||
        memory.changedFrames >= params::LIVE_REFERENCE_CONFIRM_FRAMES) {
        memory.changedFrames = 0;
        memory.references = references;
        memory.referencePatches.clear();
        memory.referenceFeatures.clear();
        for (const ReferenceCoin& reference : references) {
            double stretch = 1.0;
            memory.referencePatches.push_back(makeReferencePatch(reference, stretch));
            memory.referenceFeatures.push_back(
                extractSift(memory.referencePatches.back(),
                            params::LIVE_SIFT_CONTRAST_THRESHOLD, params::LIVE_ROOT_SIFT));
        }
        memory.forceRelearn = false;
        memory.tracks.clear();  // old votes refer to the old type numbers
        result.referencesRelearned = true;
    }
    if (memory.references.empty()) {
        return result;  // nothing to compare with
    }

    // Optional colour signatures (same light for both rectangles).
    std::vector<ColorSignature> referenceColors;
    cv::Mat searchLab;
    if (params::LIVE_USE_COLOR_FILTER) {
        for (const ReferenceCoin& reference : memory.references) {
            referenceColors.push_back(measureReferenceColor(reference));
        }
        cv::cvtColor(view.searchImage, searchLab, cv::COLOR_BGR2Lab);
    }

    // ---- Search rectangle: Hough candidates, diameter filter, SIFT match.
    const std::vector<std::string> names = referenceNames(memory);
    const cv::Mat searchGray = toGrayscale(view.searchImage);
    result.candidates = findCoins(view.searchImage, searchHoughSettings(memory.references, settings),
                                  settings.searchRimSupport, false);
    for (const CoinCandidate& candidate : result.candidates) {
        // Both rectangles come from the same top-down view, so the pixel
        // diameters are directly comparable (REFERENCE_TO_SEARCH_SCALE = 1).
        const SizeCheck size = checkDiameter(candidate, memory.references);
        Prefilter prefilter;
        prefilter.types = size.compatibleTypes;
        if (params::LIVE_USE_COLOR_FILTER && !prefilter.types.empty()) {
            prefilter.types = filterTypesByColor(
                prefilter.types, measureCandidateColor(searchLab, candidate), referenceColors);
        }
        if (size.compatibleTypes.empty()) {
            prefilter.rejectedBy = "size";
        } else if (prefilter.types.empty()) {
            prefilter.rejectedBy = "color";
        }
        // prefilter.colorDecisive stays false: in live mode SIFT decides.

        CoinFeatures features;
        if (!prefilter.types.empty()) {
            features = extractSift(makeCandidatePatch(searchGray, candidate),
                                   params::LIVE_SIFT_CONTRAST_THRESHOLD, params::LIVE_ROOT_SIFT);
            result.siftComparisons += static_cast<int>(prefilter.types.size());
        }
        result.results.push_back(
            classifyCandidate(features, memory.referenceFeatures, prefilter, names,
                              settings.minInliers, settings.loweRatio));
    }
    voteOverTime(result, settings, memory);
    return result;
}

// ------------------------------------------------------------- drawing --

static cv::Scalar typeColor(int type)
{
    const cv::Scalar palette[] = {cv::Scalar(0, 120, 255), cv::Scalar(255, 200, 0),
                                  cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 0),
                                  cv::Scalar(255, 120, 120), cv::Scalar(120, 255, 255)};
    return palette[type % 6];
}

// A circle of the rectified image, drawn in the (possibly tilted) camera frame:
// sample points on the circle and map them with the homography.
static void drawMappedCircle(cv::Mat& canvas, const cv::Mat& toFrame, cv::Point2f center,
                             float radius, const cv::Scalar& color, int thickness)
{
    std::vector<cv::Point2f> ring;
    for (int k = 0; k < 40; ++k) {
        const double angle = 2.0 * CV_PI * k / 40.0;
        ring.emplace_back(center.x + radius * static_cast<float>(std::cos(angle)),
                          center.y + radius * static_cast<float>(std::sin(angle)));
    }
    std::vector<cv::Point> polygon;
    for (const cv::Point2f& p : mapPoints(toFrame, ring)) {
        polygon.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    cv::polylines(canvas, polygon, true, color, thickness, cv::LINE_AA);
}

static void putLabel(cv::Mat& canvas, const std::string& text, cv::Point origin,
                     const cv::Scalar& color, double scale = 0.6)
{
    cv::putText(canvas, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 5,
                cv::LINE_AA);
    cv::putText(canvas, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, 2, cv::LINE_AA);
}

static void drawQuad(cv::Mat& canvas, const std::vector<cv::Point2f>& corners,
                     const cv::Scalar& color)
{
    std::vector<cv::Point> polygon;
    for (const cv::Point2f& p : corners) {
        polygon.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    cv::polylines(canvas, polygon, true, color, 2, cv::LINE_AA);
}

cv::Mat drawLiveOverlay(const cv::Mat& frame, const PaperView& view, const LiveMemory& memory,
                        const LiveResult& result, double fps)
{
    cv::Mat canvas = frame.clone();
    char text[128];

    if (!view.found) {
        std::snprintf(text, sizeof(text), "paper not found  |  %.1f fps", fps);
        cv::rectangle(canvas, cv::Rect(0, 0, canvas.cols, 34), cv::Scalar(30, 30, 30),
                      cv::FILLED);
        putLabel(canvas, text, cv::Point(10, 24), cv::Scalar(0, 0, 255));
        return canvas;
    }

    drawQuad(canvas, view.searchCorners, cv::Scalar(0, 200, 0));
    drawQuad(canvas, view.referenceCorners, cv::Scalar(255, 150, 0));

    // Reference coins (they define the types).
    for (size_t k = 0; k < memory.references.size(); ++k) {
        const ReferenceCoin& reference = memory.references[k];
        const cv::Point2f center = outlineCenter(reference);
        drawMappedCircle(canvas, view.referenceToFrame, center, reference.outline.width / 2.0f,
                         typeColor(static_cast<int>(k)), 3);
        const cv::Point2f at = mapPoints(view.referenceToFrame, {center})[0];
        putLabel(canvas, reference.name, cv::Point(cvRound(at.x) - 30, cvRound(at.y)),
                 typeColor(static_cast<int>(k)));
    }

    // Candidates with their decision.
    std::vector<int> counts(memory.references.size(), 0);
    int unknown = 0;
    for (size_t i = 0; i < result.candidates.size(); ++i) {
        const CoinCandidate& candidate = result.candidates[i];
        const Classification& decision = result.results[i];
        const int type = result.stableTypes.empty() ? decision.type : result.stableTypes[i];
        const bool recognised = type >= 0 && type < static_cast<int>(counts.size());
        const cv::Scalar color = recognised ? typeColor(type) : cv::Scalar(180, 180, 180);
        const int guess = result.bestGuess.empty() ? -1 : result.bestGuess[i];
        const double confidence = result.confidence.empty() ? 0.0 : result.confidence[i];
        drawMappedCircle(canvas, view.searchToFrame, candidate.center, candidate.radius, color,
                         recognised ? 4 : 2);

        const cv::Point2f at = mapPoints(view.searchToFrame, {candidate.center})[0];
        // Label = type + confidence (share of the last frames it was recognised).
        if (recognised) {
            ++counts[type];
            std::snprintf(text, sizeof(text), "%s %.0f%%", memory.references[type].name.c_str(),
                          100.0 * confidence);
        } else if (guess >= 0 && guess < static_cast<int>(counts.size())) {
            ++unknown;
            std::snprintf(text, sizeof(text), "? %s %.0f%%",
                          memory.references[guess].name.c_str(), 100.0 * confidence);
        } else {
            ++unknown;
            std::snprintf(text, sizeof(text), "?");
        }
        putLabel(canvas, text, cv::Point(cvRound(at.x) - 30, cvRound(at.y)), color);
    }

    // Banner: counts per reference type + speed.
    std::string banner;
    for (size_t k = 0; k < counts.size(); ++k) {
        banner += memory.references[k].name + " " + std::to_string(counts[k]) + "  ";
    }
    std::snprintf(text, sizeof(text), "| unknown %d | %.1f fps", unknown, fps);
    banner += text;
    cv::rectangle(canvas, cv::Rect(0, 0, canvas.cols, 34), cv::Scalar(30, 30, 30), cv::FILLED);
    putLabel(canvas, banner, cv::Point(10, 24), cv::Scalar(255, 255, 255));
    return canvas;
}
