#include "circle_detection.hpp"

#include "params.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

HoughSettings imageHoughSettings()
{
    return {params::HOUGH_DP,     params::HOUGH_MIN_DIST,   params::HOUGH_PARAM1,
            params::HOUGH_PARAM2, params::HOUGH_MIN_RADIUS, params::HOUGH_MAX_RADIUS};
}

std::vector<cv::Vec3f> detectCircles(const cv::Mat& smoothedGray)
{
    return detectCircles(smoothedGray, imageHoughSettings());
}

std::vector<cv::Vec3f> detectCircles(const cv::Mat& smoothedGray, const HoughSettings& settings)
{
    // Hough transform for circles, gradient variant (HOUGH_GRADIENT):
    //   1. Canny(param1 / 2, param1) finds edge pixels of the input.
    //   2. Every edge pixel votes for possible centers along its gradient
    //      direction (the center of a circle lies on the normal of its rim),
    //      for all radii in [minRadius, maxRadius].
    //   3. Accumulator cells with at least param2 votes, that are local maxima
    //      and lie at least minDist apart, become centers.
    //   4. For each center the radius supported by the most edge pixels wins.
    // The input is the smoothed intensity image, not our own Canny output:
    // OpenCV needs the gradient directions, which a binary edge map has lost.
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(smoothedGray, circles, cv::HOUGH_GRADIENT, settings.dp,
                     settings.minDist, settings.param1, settings.param2,
                     settings.minRadius, settings.maxRadius);
    return circles;
}

double rimSupport(const cv::Mat& edges, const cv::Vec3f& circle)
{
    // Accept an edge pixel up to 2 px away from the ideal rim (the Hough
    // radius is not exact and edges are 1 px thin).
    const int tolerance = 2;
    int onRim = 0;
    int inside = 0;
    for (int k = 0; k < 90; ++k) {
        const double angle = 2.0 * CV_PI * k / 90.0;
        const int x = cvRound(circle[0] + circle[2] * std::cos(angle));
        const int y = cvRound(circle[1] + circle[2] * std::sin(angle));
        if (x < 0 || y < 0 || x >= edges.cols || y >= edges.rows) {
            continue;  // this part of the coin is outside the image
        }
        ++inside;
        const cv::Rect window = cv::Rect(x - tolerance, y - tolerance, 2 * tolerance + 1,
                                         2 * tolerance + 1) &
                                cv::Rect(0, 0, edges.cols, edges.rows);
        onRim += cv::countNonZero(edges(window)) > 0 ? 1 : 0;
    }
    return inside == 0 ? 0.0 : static_cast<double>(onRim) / inside;
}

// True if circle `b` describes the same coin as circle `a`: their centers are
// closer than a fraction of the smaller radius.
static bool isSameCoin(const cv::Vec3f& a, const cv::Vec3f& b)
{
    const float distance = std::hypot(a[0] - b[0], a[1] - b[1]);
    const float smallerRadius = std::min(a[2], b[2]);
    return distance < params::DUPLICATE_CENTER_FRACTION * smallerRadius;
}

static bool isCutByBorder(const cv::Vec3f& circle, const cv::Size& imageSize)
{
    const float x = circle[0];
    const float y = circle[1];
    const float r = circle[2];
    return x - r < 0 || y - r < 0 || x + r > imageSize.width || y + r > imageSize.height;
}

std::vector<CoinCandidate> selectCandidates(const std::vector<cv::Vec3f>& circles,
                                            const cv::Size& imageSize, bool verbose)
{
    // minDist already prevents most duplicates; this explicit check makes the
    // "each physical coin counts once" rule visible. HoughCircles lists the
    // centers with more votes first, so the earlier (stronger) circle is kept.
    std::vector<cv::Vec3f> kept;
    for (const cv::Vec3f& circle : circles) {
        bool duplicate = false;
        for (const cv::Vec3f& accepted : kept) {
            if (isSameCoin(accepted, circle)) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            kept.push_back(circle);
        }
    }

    if (verbose) {
        std::cout << "  Hough circles: " << circles.size()
                  << ", duplicates removed: " << circles.size() - kept.size() << "\n";
    }

    std::vector<CoinCandidate> candidates;
    for (size_t i = 0; i < kept.size(); ++i) {
        const cv::Vec3f& c = kept[i];
        candidates.push_back({static_cast<int>(i) + 1, cv::Point2f(c[0], c[1]), c[2],
                              isCutByBorder(c, imageSize)});
    }
    return candidates;
}

cv::Mat drawCandidates(const cv::Mat& colorImage,
                       const std::vector<CoinCandidate>& candidates)
{
    cv::Mat canvas = colorImage.clone();  // never draw on the input itself
    const cv::Scalar green(0, 255, 0);
    const cv::Scalar red(0, 0, 255);
    const cv::Scalar yellow(0, 255, 255);
    const cv::Scalar black(0, 0, 0);

    for (const CoinCandidate& candidate : candidates) {
        const cv::Point center(cvRound(candidate.center.x), cvRound(candidate.center.y));
        cv::circle(canvas, center, cvRound(candidate.radius), green, 4, cv::LINE_AA);
        cv::circle(canvas, center, 6, red, cv::FILLED, cv::LINE_AA);

        // Black outline under yellow text keeps the id readable on any coin.
        const std::string label = std::to_string(candidate.id);
        const cv::Point textOrigin = center + cv::Point(-18, -16);
        cv::putText(canvas, label, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 1.4, black, 8,
                    cv::LINE_AA);
        cv::putText(canvas, label, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 1.4, yellow, 3,
                    cv::LINE_AA);
    }

    const std::string title =
        std::to_string(candidates.size()) + " geometric candidates (not coin classes)";
    cv::putText(canvas, title, cv::Point(15, 45), cv::FONT_HERSHEY_SIMPLEX, 1.2, black, 8,
                cv::LINE_AA);
    cv::putText(canvas, title, cv::Point(15, 45), cv::FONT_HERSHEY_SIMPLEX, 1.2, yellow, 3,
                cv::LINE_AA);
    return canvas;
}

void printCandidates(const std::vector<CoinCandidate>& candidates, const cv::Point& offset)
{
    std::printf("   id | center in search region | center in full image | radius | cut by border\n");
    for (const CoinCandidate& c : candidates) {
        std::printf("  %3d | (%5.0f, %5.0f)          | (%5.0f, %5.0f)       | %6.1f | %s\n",
                    c.id, c.center.x, c.center.y, c.center.x + offset.x,
                    c.center.y + offset.y, c.radius, c.cutByBorder ? "yes" : "no");
    }
    std::fflush(stdout);
}
