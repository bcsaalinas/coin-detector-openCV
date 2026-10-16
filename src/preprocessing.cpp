#include "preprocessing.hpp"

#include "params.hpp"

#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

cv::Mat toGrayscale(const cv::Mat& colorImage)
{
    // Weighted sum Y = 0.299 R + 0.587 G + 0.114 B. Edges and circles only need
    // intensity changes, so color is not used in H2-H3.
    cv::Mat gray;
    cv::cvtColor(colorImage, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat smoothGaussian(const cv::Mat& grayImage)
{
    // Step 1 of Canny's method: noise reduction. Convolving with a Gaussian
    // removes small, high-frequency details (wood grain, engraving, sensor
    // noise) that would otherwise produce many short edges.
    cv::Mat smoothed;
    const cv::Size kernel(params::GAUSSIAN_KERNEL_SIZE, params::GAUSSIAN_KERNEL_SIZE);
    cv::GaussianBlur(grayImage, smoothed, kernel, params::GAUSSIAN_SIGMA);
    return smoothed;
}

cv::Mat detectEdgesCanny(const cv::Mat& smoothedImage)
{
    // cv::Canny performs the remaining Canny steps on the smoothed image:
    //   2. Sobel gradients (magnitude and direction),
    //   3. non-maximum suppression (thin edges to 1 px along the gradient),
    //   4. double threshold + hysteresis (keep weak edges only when they
    //      connect to strong ones).
    // We smooth explicitly beforehand so that step 1 can be seen and tuned.
    cv::Mat edges;
    cv::Canny(smoothedImage, edges, params::CANNY_LOW_THRESHOLD,
              params::CANNY_HIGH_THRESHOLD);
    return edges;
}

// Converts a gray image to BGR (so it can sit next to color images), shrinks
// it and writes a caption in the top-left corner.
static cv::Mat makeTile(const cv::Mat& image, const std::string& caption, double scale)
{
    cv::Mat tile;
    if (image.channels() == 1) {
        cv::cvtColor(image, tile, cv::COLOR_GRAY2BGR);
    } else {
        tile = image.clone();
    }
    cv::resize(tile, tile, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::putText(tile, caption, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    return tile;
}

cv::Mat makeComparisonStrip(const cv::Mat& color, const cv::Mat& gray,
                            const cv::Mat& smoothed, const cv::Mat& edges,
                            double scale)
{
    const std::vector<cv::Mat> tiles = {
        makeTile(color, "color", scale),
        makeTile(gray, "grayscale", scale),
        makeTile(smoothed, "gaussian", scale),
        makeTile(edges, "canny", scale),
    };
    cv::Mat strip;
    cv::hconcat(tiles, strip);
    return strip;
}
