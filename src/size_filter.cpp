#include "size_filter.hpp"

#include "params.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <string>

// Accepted candidate diameters (search pixels) for one reference coin: from
// its shorter to its longer axis, widened by SIZE_TOLERANCE on both sides.
static void diameterRange(const ReferenceCoin& reference, double& low, double& high)
{
    const double shortAxis = std::min(reference.outline.width, reference.outline.height);
    const double longAxis = std::max(reference.outline.width, reference.outline.height);
    low = shortAxis * params::REFERENCE_TO_SEARCH_SCALE * (1.0 - params::SIZE_TOLERANCE);
    high = longAxis * params::REFERENCE_TO_SEARCH_SCALE * (1.0 + params::SIZE_TOLERANCE);
}

SizeCheck checkDiameter(const CoinCandidate& candidate,
                        const std::vector<ReferenceCoin>& references)
{
    SizeCheck check;
    check.diameterPx = 2.0 * candidate.radius;
    for (size_t type = 0; type < references.size(); ++type) {
        double low = 0.0;
        double high = 0.0;
        diameterRange(references[type], low, high);
        if (check.diameterPx >= low && check.diameterPx <= high) {
            check.compatibleTypes.push_back(static_cast<int>(type));
        }
    }
    return check;
}

void printReferenceSizes(const std::vector<ReferenceCoin>& references)
{
    for (const ReferenceCoin& reference : references) {
        double low = 0.0;
        double high = 0.0;
        diameterRange(reference, low, high);
        std::printf("  %-8s outline %3d x %3d px -> accepts diameters %.0f-%.0f px\n",
                    reference.name.c_str(), reference.outline.width, reference.outline.height,
                    low, high);
    }
}

static std::string typeList(const std::vector<int>& types,
                            const std::vector<ReferenceCoin>& references, const char* separator)
{
    std::string text;
    for (int type : types) {
        text += (text.empty() ? "" : separator) + references[type].name;
    }
    return text;
}

void printSizeChecks(const std::vector<CoinCandidate>& candidates,
                     const std::vector<SizeCheck>& checks,
                     const std::vector<ReferenceCoin>& references)
{
    std::printf("   id | diameter px | compatible reference types\n");
    int comparisons = 0;
    int rejected = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        std::string types = typeList(checks[i].compatibleTypes, references, ", ");
        if (types.empty()) {
            types = "none -> discarded, no SIFT";
            ++rejected;
        }
        comparisons += static_cast<int>(checks[i].compatibleTypes.size());
        std::printf("  %3d | %11.1f | %s\n", candidates[i].id, checks[i].diameterPx,
                    types.c_str());
    }

    const int allComparisons = static_cast<int>(candidates.size() * references.size());
    std::printf("  discarded candidates: %d of %zu; SIFT comparisons: %d instead of %d "
                "(%.0f%% avoided)\n",
                rejected, candidates.size(), comparisons, allComparisons,
                100.0 * (allComparisons - comparisons) / allComparisons);
    std::fflush(stdout);
}

cv::Mat drawSizeChecks(const cv::Mat& searchColor, const std::vector<CoinCandidate>& candidates,
                       const std::vector<SizeCheck>& checks,
                       const std::vector<ReferenceCoin>& references)
{
    cv::Mat canvas = searchColor.clone();
    for (size_t i = 0; i < candidates.size(); ++i) {
        const bool kept = !checks[i].compatibleTypes.empty();
        const cv::Scalar color = kept ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        const cv::Point center(cvRound(candidates[i].center.x), cvRound(candidates[i].center.y));
        cv::circle(canvas, center, cvRound(candidates[i].radius), color, 4, cv::LINE_AA);

        char line1[64];
        std::snprintf(line1, sizeof(line1), "#%d %.0f px", candidates[i].id,
                      checks[i].diameterPx);
        const std::string line2 =
            kept ? typeList(checks[i].compatibleTypes, references, "/") : "discarded";
        const std::string lines[2] = {line1, line2};
        for (int k = 0; k < 2; ++k) {
            const cv::Point origin = center + cv::Point(-75, -5 + 32 * k);
            cv::putText(canvas, lines[k], origin, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 0), 6, cv::LINE_AA);
            cv::putText(canvas, lines[k], origin, cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        cv::LINE_AA);
        }
    }
    return canvas;
}
