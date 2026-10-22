#include "coin_patch.hpp"

#include "params.hpp"
#include "preprocessing.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <vector>

// Disc of radius PATCH_MASK_FRACTION * D/2 in the middle of a D x D patch.
static cv::Mat makeCircleMask()
{
    const int d = params::PATCH_DIAMETER;
    cv::Mat mask = cv::Mat::zeros(d, d, CV_8U);
    const int radius = cvRound(params::PATCH_MASK_FRACTION * d / 2.0);
    cv::circle(mask, cv::Point(d / 2, d / 2), radius, cv::Scalar(255), cv::FILLED);
    return mask;
}

CoinPatch makeReferencePatch(const ReferenceCoin& reference, double& stretch)
{
    // The crop is exactly the coin's bounding box, so its width and height are
    // the two axes of the (possibly stretched) coin.
    stretch = static_cast<double>(reference.crop.rows) / reference.crop.cols;

    // Independent x / y scaling turns the ellipse back into a circle.
    CoinPatch patch;
    const cv::Size size(params::PATCH_DIAMETER, params::PATCH_DIAMETER);
    cv::resize(toGrayscale(reference.crop), patch.gray, size, 0, 0, cv::INTER_AREA);

    // Keep only pixels of this coin (not a neighbour entering the box).
    cv::Mat coinMask;
    cv::resize(reference.mask, coinMask, size, 0, 0, cv::INTER_NEAREST);
    cv::bitwise_and(makeCircleMask(), coinMask, patch.mask);
    return patch;
}

CoinPatch makeCandidatePatch(const cv::Mat& searchGray, const CoinCandidate& candidate)
{
    // Pad the image so squares around coins at the border can still be cut.
    // `valid` remembers which pixels are real image and which are padding.
    // The padding depends on this coin's radius (not a fixed maximum), so any
    // Hough radius range works, including larger live views.
    const int pad = cvCeil(candidate.radius) + 10;
    cv::Mat padded;
    cv::Mat valid;
    cv::copyMakeBorder(searchGray, padded, pad, pad, pad, pad, cv::BORDER_CONSTANT,
                       cv::Scalar(0));
    cv::copyMakeBorder(cv::Mat(searchGray.size(), CV_8U, cv::Scalar(255)), valid, pad, pad,
                       pad, pad, cv::BORDER_CONSTANT, cv::Scalar(0));

    const int side = cvRound(2.0f * candidate.radius);
    const cv::Rect square(cvRound(candidate.center.x - candidate.radius) + pad,
                          cvRound(candidate.center.y - candidate.radius) + pad, side, side);

    CoinPatch patch;
    const cv::Size size(params::PATCH_DIAMETER, params::PATCH_DIAMETER);
    cv::resize(padded(square), patch.gray, size, 0, 0, cv::INTER_AREA);

    cv::Mat validPatch;
    cv::resize(valid(square), validPatch, size, 0, 0, cv::INTER_NEAREST);
    // Shrink the valid area a little: the image border itself is a strong
    // artificial edge that must not create features.
    cv::erode(validPatch, validPatch, cv::Mat::ones(9, 9, CV_8U));

    cv::bitwise_and(makeCircleMask(), validPatch, patch.mask);
    return patch;
}

cv::Mat makePatchStrip(const std::vector<CoinPatch>& patches)
{
    std::vector<cv::Mat> tiles;
    for (const CoinPatch& patch : patches) {
        cv::Mat tile = cv::Mat::zeros(patch.gray.size(), CV_8U);
        patch.gray.copyTo(tile, patch.mask);
        tiles.push_back(tile);
    }
    cv::Mat strip;
    cv::hconcat(tiles, strip);
    return strip;
}
