#include "regions.hpp"

#include "params.hpp"

#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>  // contourArea, boundingRect moved here in 5.x
#endif

#include <algorithm>
#include <iostream>

bool isInsideImage(const cv::Rect& rect, const cv::Size& imageSize)
{
    const cv::Rect imageBounds(0, 0, imageSize.width, imageSize.height);
    // The intersection equals the rectangle only if no part of it sticks out.
    return rect.area() > 0 && (rect & imageBounds) == rect;
}

cv::Mat extractRegion(const cv::Mat& image, const NamedRegion& region)
{
    if (!isInsideImage(region.rect, image.size())) {
        std::cerr << "[error] ROI '" << region.name << "' " << region.rect
                  << " is not inside the " << image.cols << " x " << image.rows
                  << " image\n";
        return cv::Mat();
    }

    // image(rect) is only a view that shares its pixels with `image`.
    // clone() makes an independent copy, so nothing done to the crop later
    // can modify the original image.
    return image(region.rect).clone();
}

// Coins whose centres differ vertically by less than half the smaller height
// are on the same row.
void sortReadingOrder(std::vector<ReferenceCoin>& coins)
{
    auto centerY = [](const ReferenceCoin& c) { return c.outline.y + c.outline.height / 2.0; };
    std::sort(coins.begin(), coins.end(), [&](const ReferenceCoin& a, const ReferenceCoin& b) {
        return centerY(a) < centerY(b);
    });

    size_t rowStart = 0;
    for (size_t i = 1; i <= coins.size(); ++i) {
        const bool rowEnds =
            i == coins.size() ||
            centerY(coins[i]) - centerY(coins[rowStart]) >
                0.5 * std::min(coins[i].outline.height, coins[rowStart].outline.height);
        if (rowEnds) {
            std::sort(coins.begin() + rowStart, coins.begin() + i,
                      [](const ReferenceCoin& a, const ReferenceCoin& b) {
                          return a.outline.x < b.outline.x;
                      });
            rowStart = i;
        }
    }
}

std::vector<ReferenceCoin> findReferenceCoins(const cv::Mat& image,
                                              const cv::Rect& referenceRegion)
{
    std::vector<ReferenceCoin> coins;
    const cv::Mat region = extractRegion(image, {"reference region", referenceRegion});
    if (region.empty()) {
        return coins;
    }

    // Coin pixels are darker than the white background. Closing fills small
    // bright highlights inside a coin so each coin is one blob.
    cv::Mat gray;
    cv::cvtColor(region, gray, cv::COLOR_BGR2GRAY);
    cv::Mat coinPixels;
    cv::threshold(gray, coinPixels, params::REFERENCE_WHITE_THRESHOLD, 255,
                  cv::THRESH_BINARY_INV);
    cv::morphologyEx(coinPixels, coinPixels, cv::MORPH_CLOSE, cv::Mat::ones(15, 15, CV_8U));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(coinPixels, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    const double minArea = params::REFERENCE_MIN_AREA_FRACTION * region.total();
    for (size_t k = 0; k < contours.size(); ++k) {
        if (cv::contourArea(contours[k]) < minArea) {
            continue;  // a speck, not a coin
        }
        // Mask of this coin only (filled contour), so a neighbouring coin that
        // enters its bounding box is not counted as part of it.
        cv::Mat fullMask = cv::Mat::zeros(region.size(), CV_8U);
        cv::drawContours(fullMask, contours, static_cast<int>(k), cv::Scalar(255), cv::FILLED);

        const cv::Rect box = cv::boundingRect(contours[k]);
        ReferenceCoin coin;
        coin.outline = box + referenceRegion.tl();
        coin.crop = region(box).clone();
        coin.mask = fullMask(box).clone();
        coins.push_back(coin);
    }

    sortReadingOrder(coins);
    if (!assignReferenceNames(coins)) {
        std::cerr << "[warning] " << coins.size() << " reference coins found but "
                  << params::REFERENCE_NAMES.size() << " names configured; using type_k\n";
    }
    return coins;
}

bool assignReferenceNames(std::vector<ReferenceCoin>& coins)
{
    const bool namesFit = coins.size() == params::REFERENCE_NAMES.size();
    for (size_t k = 0; k < coins.size(); ++k) {
        coins[k].name = namesFit ? params::REFERENCE_NAMES[k] : "type_" + std::to_string(k + 1);
    }
    return namesFit;
}

cv::Mat drawRegionsOverview(const cv::Mat& image, const std::vector<NamedRegion>& regions)
{
    cv::Mat overview = image.clone();
    const cv::Scalar red(0, 0, 255);
    for (const NamedRegion& region : regions) {
        cv::rectangle(overview, region.rect, red, 3);
        cv::putText(overview, region.name, region.rect.tl() + cv::Point(0, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, red, 3, cv::LINE_AA);
    }
    return overview;
}
