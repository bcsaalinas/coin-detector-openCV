#include "paper_view.hpp"

#include "params.hpp"

#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>  // approxPolyDP, contourArea, homographies in 5.x
#else
#include <opencv2/calib3d.hpp>
#endif

#include <algorithm>
#include <cmath>

// Four-sided outlines that are big enough to be one of the printed rectangles.
static std::vector<std::vector<cv::Point>> findQuadrilaterals(const cv::Mat& frame)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    // Dilation closes small gaps in the printed lines so each outline is one
    // connected contour.
    cv::dilate(edges, edges, cv::Mat::ones(3, 3, CV_8U));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const double minArea = params::PAPER_MIN_RECT_AREA_FRACTION * frame.total();
    std::vector<std::vector<cv::Point>> quads;
    for (const auto& contour : contours) {
        std::vector<cv::Point> polygon;
        const double perimeter = cv::arcLength(contour, true);
        cv::approxPolyDP(contour, polygon, params::PAPER_POLY_EPSILON * perimeter, true);
        if (polygon.size() == 4 && cv::isContourConvex(polygon) &&
            cv::contourArea(polygon) >= minArea) {
            quads.push_back(polygon);
        }
    }
    return quads;
}

// Orders 4 corners as top-left, top-right, bottom-right, bottom-left:
// top-left has the smallest x+y, bottom-right the largest; top-right has the
// smallest y-x, bottom-left the largest.
static std::vector<cv::Point2f> orderCorners(const std::vector<cv::Point>& quad)
{
    std::vector<cv::Point2f> p(quad.begin(), quad.end());
    auto sum = [](const cv::Point2f& q) { return q.x + q.y; };
    auto diff = [](const cv::Point2f& q) { return q.y - q.x; };
    const auto tl = *std::min_element(p.begin(), p.end(),
                                      [&](auto& a, auto& b) { return sum(a) < sum(b); });
    const auto br = *std::max_element(p.begin(), p.end(),
                                      [&](auto& a, auto& b) { return sum(a) < sum(b); });
    const auto tr = *std::min_element(p.begin(), p.end(),
                                      [&](auto& a, auto& b) { return diff(a) < diff(b); });
    const auto bl = *std::max_element(p.begin(), p.end(),
                                      [&](auto& a, auto& b) { return diff(a) < diff(b); });
    return {tl, tr, br, bl};
}

static double distance(const cv::Point2f& a, const cv::Point2f& b)
{
    return std::hypot(a.x - b.x, a.y - b.y);
}

std::vector<cv::Point2f> mapPoints(const cv::Mat& homography,
                                   const std::vector<cv::Point2f>& points)
{
    std::vector<cv::Point2f> mapped;
    if (!points.empty()) {
        cv::perspectiveTransform(points, mapped, homography);
    }
    return mapped;
}

// Warps the part of the frame inside `planeRect` (top-down coordinates),
// minus the inset, and returns the matrix that maps image pixels back.
static cv::Mat warpRegion(const cv::Mat& frame, const cv::Mat& frameToPlane,
                          const cv::Rect2f& planeRect, cv::Mat& imageToFrame)
{
    const float insetX = static_cast<float>(params::LIVE_RECT_INSET) * planeRect.width;
    const float insetY = static_cast<float>(params::LIVE_RECT_INSET) * planeRect.height;
    const cv::Size size(cvRound(planeRect.width - 2 * insetX),
                        cvRound(planeRect.height - 2 * insetY));

    // Shift so the inset corner of the rectangle becomes pixel (0, 0).
    cv::Mat shift = cv::Mat::eye(3, 3, CV_64F);
    shift.at<double>(0, 2) = -(planeRect.x + insetX);
    shift.at<double>(1, 2) = -(planeRect.y + insetY);
    const cv::Mat frameToImage = shift * frameToPlane;

    cv::Mat image;
    cv::warpPerspective(frame, image, frameToImage, size);
    imageToFrame = frameToImage.inv();
    return image;
}

// Coins are round, so in a correct top-down view a coin outline is as high
// as it is wide. Returns the median height / width of the coin outlines in the
// reference view, or 0 if no coin-like outline is found.
static double measureCoinAspect(const cv::Mat& referenceImage)
{
    // The paper is the most common colour in the reference rectangle: coins
    // are the pixels whose Lab colour differs from the median colour. Otsu
    // picks the threshold between "paper" and "different" automatically.
    cv::Mat smooth;
    cv::GaussianBlur(referenceImage, smooth, cv::Size(5, 5), 0);
    cv::Mat lab;
    cv::cvtColor(smooth, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    cv::Mat difference = cv::Mat::zeros(lab.size(), CV_32F);
    for (const cv::Mat& channel : channels) {
        std::vector<uchar> values(channel.begin<uchar>(), channel.end<uchar>());
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        cv::Mat delta;
        channel.convertTo(delta, CV_32F, 1.0, -values[values.size() / 2]);
        difference += delta.mul(delta);
    }
    cv::sqrt(difference, difference);
    cv::Mat difference8;
    cv::normalize(difference, difference8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat coins;
    cv::threshold(difference8, coins, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::morphologyEx(coins, coins, cv::MORPH_CLOSE, cv::Mat::ones(9, 9, CV_8U));
    cv::morphologyEx(coins, coins, cv::MORPH_OPEN, cv::Mat::ones(5, 5, CV_8U));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(coins, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    const int minSide = referenceImage.cols / 25;
    const cv::Rect inner(1, 1, referenceImage.cols - 2, referenceImage.rows - 2);
    std::vector<double> ratios;
    for (const auto& contour : contours) {
        const cv::Rect box = cv::boundingRect(contour);
        if (box.width < minSide || box.height < minSide || (box & inner) != box) {
            continue;  // too small, or touching the border (cut off)
        }
        // An ellipse fills pi/4 = 0.785 of its bounding box; broken or
        // irregular blobs don't, and are ignored.
        const double fill = cv::contourArea(contour) / box.area();
        if (fill > 0.70 && fill < 0.86) {
            ratios.push_back(static_cast<double>(box.height) / box.width);
        }
    }
    if (ratios.empty()) {
        return 0.0;
    }
    std::nth_element(ratios.begin(), ratios.begin() + ratios.size() / 2, ratios.end());
    return ratios[ratios.size() / 2];
}

PaperView findPaper(const cv::Mat& frame)
{
    PaperView view;
    const std::vector<std::vector<cv::Point>> quads = findQuadrilaterals(frame);

    // The printed pair = the two largest quadrilaterals that do not overlap.
    // (The edge of the sheet overlaps both; the inner and outer edge of one
    // printed line overlap each other.)
    int bestA = -1;
    int bestB = -1;
    double bestArea = 0.0;
    for (size_t i = 0; i < quads.size(); ++i) {
        for (size_t j = i + 1; j < quads.size(); ++j) {
            const cv::Rect a = cv::boundingRect(quads[i]);
            const cv::Rect b = cv::boundingRect(quads[j]);
            const double overlap = (a & b).area();
            if (overlap > 0.05 * std::min(a.area(), b.area())) {
                continue;
            }
            const double area = cv::contourArea(quads[i]) + cv::contourArea(quads[j]);
            if (area > bestArea) {
                bestArea = area;
                bestA = static_cast<int>(i);
                bestB = static_cast<int>(j);
            }
        }
    }
    if (bestA < 0) {
        return view;  // fewer than two separate rectangles visible
    }

    const bool aIsSmaller = cv::contourArea(quads[bestA]) < cv::contourArea(quads[bestB]);
    const int referenceIndex = (aIsSmaller == params::REFERENCE_IS_SMALLER_RECT) ? bestA : bestB;
    const int searchIndex = referenceIndex == bestA ? bestB : bestA;
    view.searchCorners = orderCorners(quads[searchIndex]);
    view.referenceCorners = orderCorners(quads[referenceIndex]);

    // One homography for the whole sheet, computed from the (bigger) search
    // rectangle: its corners go to an upright rectangle LIVE_SEARCH_WIDTH wide.
    // First guess of its height: the aspect ratio seen in the frame (average
    // side lengths), or the configured one.
    const std::vector<cv::Point2f>& s = view.searchCorners;
    const double widthSeen = (distance(s[0], s[1]) + distance(s[3], s[2])) / 2.0;
    const double heightSeen = (distance(s[0], s[3]) + distance(s[1], s[2])) / 2.0;
    const float w = static_cast<float>(params::LIVE_SEARCH_WIDTH);
    float h = static_cast<float>(w * (params::PAPER_SEARCH_ASPECT > 0
                                          ? params::PAPER_SEARCH_ASPECT
                                          : heightSeen / widthSeen));

    // With a tilted camera, the seen aspect ratio is wrong (foreshortening).
    // Since the four corners are fixed, the error is a pure vertical scaling,
    // which makes the round reference coins look like ellipses. Their height /
    // width tells the correction, so a second pass makes them round again.
    const int passes = params::PAPER_SEARCH_ASPECT > 0 ? 1 : 2;
    for (int pass = 0; pass < passes; ++pass) {
        const std::vector<cv::Point2f> upright = {{0, 0}, {w, 0}, {w, h}, {0, h}};
        const cv::Mat frameToPlane = cv::getPerspectiveTransform(s, upright);

        // The reference rectangle in the same top-down view.
        const std::vector<cv::Point2f> referenceInPlane =
            mapPoints(frameToPlane, view.referenceCorners);
        const cv::Rect2f referenceRect = cv::boundingRect(referenceInPlane);
        if (referenceRect.width < 20 || referenceRect.height < 20 ||
            referenceRect.width > 4 * w || referenceRect.height > 4 * h) {
            return view;  // implausible geometry (e.g. wrong pair of outlines)
        }
        view.referenceImage =
            warpRegion(frame, frameToPlane, referenceRect, view.referenceToFrame);

        if (pass + 1 < passes) {
            const double coinAspect = measureCoinAspect(view.referenceImage);
            if (coinAspect > 0.4 && coinAspect < 2.5) {
                h = static_cast<float>(h / coinAspect);
            }
            continue;
        }
        view.searchImage = warpRegion(frame, frameToPlane, cv::Rect2f(0, 0, w, h),
                                      view.searchToFrame);
    }
    view.found = true;
    return view;
}
