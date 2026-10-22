#include "color_filter.hpp"

#include "params.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

const char* metalName(Metal metal)
{
    switch (metal) {
    case Metal::Copper:
        return "copper";
    case Metal::Gold:
        return "gold";
    default:
        return "silver";
    }
}

std::string describeMetals(const ColorSignature& signature)
{
    if (signature.center.metal == signature.ring.metal) {
        return metalName(signature.center.metal);
    }
    return std::string(metalName(signature.center.metal)) + "/" + metalName(signature.ring.metal);
}

static double median(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

// Median colour of the pixels whose normalised distance to the centre,
// sqrt((dx / rx)^2 + (dy / ry)^2), lies in [from, to). rx = ry for a circle;
// different radii describe the stretched (elliptical) reference coins.
// Pixels where `valid` is 0 are skipped (empty `valid` = all pixels valid).
static ColorSample sampleZone(const cv::Mat& lab, const cv::Mat& valid, cv::Point2d center,
                              double rx, double ry, double from, double to)
{
    std::vector<double> aValues;
    std::vector<double> bValues;
    const int x0 = std::max(0, static_cast<int>(center.x - rx));
    const int x1 = std::min(lab.cols - 1, static_cast<int>(center.x + rx));
    const int y0 = std::max(0, static_cast<int>(center.y - ry));
    const int y1 = std::min(lab.rows - 1, static_cast<int>(center.y + ry));
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const double d = std::hypot((x - center.x) / rx, (y - center.y) / ry);
            if (d < from || d >= to || (!valid.empty() && !valid.at<uchar>(y, x))) {
                continue;
            }
            // 8-bit Lab in OpenCV stores a* and b* with an offset of 128.
            const cv::Vec3b pixel = lab.at<cv::Vec3b>(y, x);
            aValues.push_back(pixel[1] - 128.0);
            bValues.push_back(pixel[2] - 128.0);
        }
    }

    ColorSample sample;
    sample.a = median(aValues);
    sample.b = median(bValues);
    sample.chroma = std::hypot(sample.a, sample.b);
    sample.hueDeg = std::atan2(sample.b, sample.a) * 180.0 / CV_PI;

    if (sample.chroma < params::MIN_CHROMA_COLORED) {
        sample.metal = Metal::Silver;
    } else if (sample.hueDeg < params::MAX_HUE_COPPER_DEG) {
        sample.metal = Metal::Copper;
    } else {
        sample.metal = Metal::Gold;
    }
    return sample;
}

static ColorSignature sampleSignature(const cv::Mat& lab, const cv::Mat& valid,
                                      cv::Point2d center, double rx, double ry)
{
    ColorSignature signature;
    signature.center = sampleZone(lab, valid, center, rx, ry, 0.0, params::COLOR_CENTER_TO);
    signature.ring = sampleZone(lab, valid, center, rx, ry, params::COLOR_RING_FROM,
                                params::COLOR_RING_TO);
    return signature;
}

ColorSignature measureCandidateColor(const cv::Mat& searchLab, const CoinCandidate& candidate)
{
    // Coins cut by the image border are measured on their visible part only.
    return sampleSignature(searchLab, cv::Mat(), candidate.center, candidate.radius,
                           candidate.radius);
}

ColorSignature measureReferenceColor(const ReferenceCoin& reference)
{
    cv::Mat lab;
    cv::cvtColor(reference.crop, lab, cv::COLOR_BGR2Lab);
    const double rx = reference.crop.cols / 2.0;
    const double ry = reference.crop.rows / 2.0;
    return sampleSignature(lab, reference.mask, cv::Point2d(rx, ry), rx, ry);
}

static bool sameMetals(const ColorSignature& x, const ColorSignature& y)
{
    return x.center.metal == y.center.metal && x.ring.metal == y.ring.metal;
}

std::vector<int> filterTypesByColor(const std::vector<int>& types,
                                    const ColorSignature& candidate,
                                    const std::vector<ColorSignature>& references)
{
    std::vector<int> kept;
    for (int type : types) {
        if (sameMetals(candidate, references[type])) {
            kept.push_back(type);
        }
    }
    return kept;
}

bool isDistinctiveColor(const ColorSignature& signature,
                        const std::vector<ColorSignature>& references)
{
    int sameCount = 0;
    for (const ColorSignature& reference : references) {
        sameCount += sameMetals(signature, reference) ? 1 : 0;
    }
    return sameCount == 1;
}
