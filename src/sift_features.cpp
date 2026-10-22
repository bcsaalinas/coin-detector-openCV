#include "sift_features.hpp"

#include "params.hpp"

#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/features.hpp>
#else
#include <opencv2/features2d.hpp>
#endif

cv::Mat normalizeContrast(const cv::Mat& gray)
{
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        params::CLAHE_CLIP_LIMIT, cv::Size(params::CLAHE_TILES, params::CLAHE_TILES));
    cv::Mat equalized;
    clahe->apply(gray, equalized);
    return equalized;
}

CoinFeatures extractSift(const CoinPatch& patch, double contrastThreshold, bool rootSift)
{
    // SIFT settings from Lowe's paper: 3 layers per octave, edge threshold 10,
    // sigma 1.6; contrast threshold 0.04 unless a lower one is asked for.
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(0, 3, contrastThreshold, 10, 1.6);

    CoinFeatures features;
    sift->detectAndCompute(normalizeContrast(patch.gray), patch.mask, features.keypoints,
                           features.descriptors);

    if (rootSift) {
        // RootSIFT: divide each descriptor by the sum of its values (L1 norm)
        // and take the square root. Comparing these with the usual Euclidean
        // distance equals the Hellinger distance between the gradient
        // histograms, which lets large single bins (e.g. a glare edge)
        // dominate less.
        for (int row = 0; row < features.descriptors.rows; ++row) {
            cv::Mat descriptor = features.descriptors.row(row);
            descriptor /= cv::sum(descriptor)[0] + 1e-7;
            cv::sqrt(descriptor, descriptor);
        }
    }
    return features;
}

cv::Mat drawSiftKeypoints(const CoinPatch& patch, const CoinFeatures& features)
{
    cv::Mat canvas;
    cv::cvtColor(normalizeContrast(patch.gray), canvas, cv::COLOR_GRAY2BGR);
    cv::drawKeypoints(canvas, features.keypoints, canvas, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_OVER_OUTIMG |
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return canvas;
}
