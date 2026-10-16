#include "regions.hpp"

#include "params.hpp"

#include <opencv2/imgproc.hpp>

#include <iostream>

std::vector<NamedRegion> referenceRegions()
{
    return {
        {"penny", params::PENNY_REGION},
        {"nickel", params::NICKEL_REGION},
        {"dime", params::DIME_REGION},
        {"quarter", params::QUARTER_REGION},
    };
}

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
