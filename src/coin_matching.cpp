#include "coin_matching.hpp"

#include "params.hpp"

#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#else
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <map>

// Steps 1 and 2: nearest neighbours + Lowe's ratio test. Several candidate
// keypoints may pick the same reference keypoint; only the closest one is
// kept, so each correspondence is one-to-one.
static std::vector<cv::DMatch> ratioTestMatches(const cv::Mat& candidateDescriptors,
                                                const cv::Mat& referenceDescriptors,
                                                double loweRatio)
{
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(candidateDescriptors, referenceDescriptors, knn, 2);

    std::map<int, cv::DMatch> bestPerReference;
    for (const std::vector<cv::DMatch>& pair : knn) {
        if (pair.size() < 2 || pair[0].distance >= loweRatio * pair[1].distance) {
            continue;  // ambiguous: the second neighbour is almost as close
        }
        const cv::DMatch& match = pair[0];
        auto found = bestPerReference.find(match.trainIdx);
        if (found == bestPerReference.end() || match.distance < found->second.distance) {
            bestPerReference[match.trainIdx] = match;
        }
    }

    std::vector<cv::DMatch> good;
    for (const auto& entry : bestPerReference) {
        good.push_back(entry.second);
    }
    return good;
}

MatchResult matchToReference(const CoinFeatures& candidate, const CoinFeatures& reference,
                             int type, double loweRatio)
{
    MatchResult result;
    result.type = type;
    if (candidate.descriptors.rows < 2 || reference.descriptors.rows < 2) {
        return result;
    }

    const std::vector<cv::DMatch> good =
        ratioTestMatches(candidate.descriptors, reference.descriptors, loweRatio);
    result.goodMatches = static_cast<int>(good.size());
    if (good.size() < 3) {
        return result;  // too few points to estimate a transform
    }

    // Step 3: RANSAC. Repeatedly fit a similarity transform to random minimal
    // subsets of matches and keep the one that most matches agree with.
    std::vector<cv::Point2f> referencePoints;
    std::vector<cv::Point2f> candidatePoints;
    for (const cv::DMatch& match : good) {
        referencePoints.push_back(reference.keypoints[match.trainIdx].pt);
        candidatePoints.push_back(candidate.keypoints[match.queryIdx].pt);
    }
    std::vector<uchar> inlierMask;
    const cv::Mat transform =
        cv::estimateAffinePartial2D(referencePoints, candidatePoints, inlierMask, cv::RANSAC,
                                    params::RANSAC_REPROJECTION_PX);
    if (transform.empty()) {
        return result;
    }

    // [a -b tx; b a ty]: scale = sqrt(a^2 + b^2). Both coins are centred and
    // have the same diameter, so a real match keeps the centre in place and
    // has a scale close to 1.
    const double a = transform.at<double>(0, 0);
    const double b = transform.at<double>(1, 0);
    result.scale = std::hypot(a, b);
    const double c = params::PATCH_DIAMETER / 2.0;
    const double mappedX = a * c - b * c + transform.at<double>(0, 2);
    const double mappedY = b * c + a * c + transform.at<double>(1, 2);
    const double centerShift = std::hypot(mappedX - c, mappedY - c);
    result.geometryOk = result.scale >= params::MIN_TRANSFORM_SCALE &&
                        result.scale <= params::MAX_TRANSFORM_SCALE &&
                        centerShift <= params::MAX_CENTER_SHIFT_PX;
    if (!result.geometryOk) {
        return result;  // the agreeing matches describe an impossible motion
    }

    for (size_t i = 0; i < good.size(); ++i) {
        if (inlierMask[i]) {
            result.inlierMatches.push_back(good[i]);
        }
    }
    result.inliers = static_cast<int>(result.inlierMatches.size());
    const int smallerSet =
        std::min(candidate.descriptors.rows, reference.descriptors.rows);
    result.matchPercent = 100.0 * result.inliers / smallerSet;
    return result;
}

// Decision rules, in order:
//   1. rejected by diameter or colour            -> size-/color-rejected
//   2. SIFT accepts one type clearly (>= minInliers) -> that type (evidence SIFT)
//   3. SIFT accepts two types about equally      -> ambiguous
//   4. no SIFT match, but colour is decisive     -> that type   (size+color)
//      (one compatible type and no other reference has the same colour)
//   5. otherwise                                 -> unknown
Classification classifyCandidate(const CoinFeatures& candidate,
                                 const std::vector<CoinFeatures>& references,
                                 const Prefilter& prefilter,
                                 const std::vector<std::string>& names, int minInliers,
                                 double loweRatio)
{
    Classification result;
    if (prefilter.types.empty()) {
        result.label = prefilter.rejectedBy + "-rejected";
        return result;
    }

    // Compare only with the coin types allowed by the pre-filters.
    for (int type : prefilter.types) {
        result.comparisons.push_back(
            matchToReference(candidate, references[type], type, loweRatio));
    }
    std::sort(result.comparisons.begin(), result.comparisons.end(),
              [](const MatchResult& x, const MatchResult& y) { return x.inliers > y.inliers; });
    result.best = result.comparisons.front();

    const bool bestAccepted = result.best.inliers >= minInliers;
    const bool secondAccepted = result.comparisons.size() > 1 &&
                                result.comparisons[1].inliers >= minInliers;

    if (bestAccepted && secondAccepted &&
        result.best.inliers < params::AMBIGUITY_FACTOR * result.comparisons[1].inliers) {
        result.label = "ambiguous";  // two types fit about equally well
    } else if (bestAccepted) {
        result.type = result.best.type;
        result.evidence = "SIFT";
    } else if (prefilter.colorDecisive) {
        result.type = prefilter.types.front();
        result.evidence = "size+color";
    } else {
        result.label = "unknown";  // nothing identifies the type
    }

    if (result.type >= 0) {
        result.label = names[result.type];
    }
    return result;
}

cv::Mat drawMatchResult(const CoinPatch& candidatePatch, const CoinFeatures& candidate,
                        const CoinPatch& referencePatch, const CoinFeatures& reference,
                        const MatchResult& result)
{
    cv::Mat left;
    cv::Mat right;
    cv::cvtColor(normalizeContrast(candidatePatch.gray), left, cv::COLOR_GRAY2BGR);
    cv::cvtColor(normalizeContrast(referencePatch.gray), right, cv::COLOR_GRAY2BGR);

    cv::Mat canvas;
    cv::drawMatches(left, candidate.keypoints, right, reference.keypoints,
                    result.inlierMatches, canvas, cv::Scalar(0, 255, 0),
                    cv::Scalar(0, 140, 255));
    return canvas;
}
