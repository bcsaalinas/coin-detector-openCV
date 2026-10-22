#include "image_io.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

cv::Mat loadColorImage(const std::string& path)
{
    if (!fs::exists(path)) {
        std::cerr << "[error] Input image not found: " << path << "\n";
        return cv::Mat();
    }

    // IMREAD_COLOR always returns 3 channels (BGR). The provided PNG has an
    // alpha channel, but it is fully opaque, so dropping it loses nothing.
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "[error] OpenCV could not decode the image: " << path << "\n";
    }
    return image;
}

void printImageInfo(const std::string& label, const cv::Mat& image)
{
    std::cout << label << ": " << image.cols << " x " << image.rows
              << " px, " << image.channels() << " channel(s), depth "
              << (image.depth() == CV_8U ? "8-bit unsigned" : "other") << "\n";
}

bool ensureDirectory(const std::string& directory)
{
    std::error_code error;
    fs::create_directories(directory, error);
    if (error || !fs::is_directory(directory)) {
        std::cerr << "[error] Cannot create output directory " << directory
                  << ": " << error.message() << "\n";
        return false;
    }
    return true;
}

bool saveImage(const std::string& directory, const std::string& fileName,
               const cv::Mat& image)
{
    const std::string path = (fs::path(directory) / fileName).string();
    bool written = false;
    try {
        written = cv::imwrite(path, image);
    } catch (const cv::Exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
    }

    if (!written) {
        std::cerr << "[error] Could not write " << path << "\n";
        return false;
    }
    std::cout << "  saved " << path << "\n";
    return true;
}

void showImage(const std::string& windowTitle, const cv::Mat& image)
{
    // Large images do not fit on screen: shrink the displayed copy only.
    const double maxSide = 900.0;
    const double scale = std::min(1.0, maxSide / std::max(image.cols, image.rows));

    cv::Mat display;
    cv::resize(image, display, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::imshow(windowTitle, display);
}

cv::Mat makeMosaic(const std::vector<cv::Mat>& tiles, const std::vector<std::string>& captions,
                   int columns)
{
    if (tiles.empty()) {
        return cv::Mat();
    }
    const cv::Size tileSize = tiles[0].size();
    const int rows = (static_cast<int>(tiles.size()) + columns - 1) / columns;
    cv::Mat mosaic(rows * tileSize.height, columns * tileSize.width, CV_8UC3,
                   cv::Scalar(40, 40, 40));

    for (size_t i = 0; i < tiles.size(); ++i) {
        const int row = static_cast<int>(i) / columns;
        const int column = static_cast<int>(i) % columns;
        cv::Mat cell = mosaic(cv::Rect(column * tileSize.width, row * tileSize.height,
                                       tileSize.width, tileSize.height));
        tiles[i].copyTo(cell);
        if (i < captions.size()) {
            cv::putText(cell, captions[i], cv::Point(8, 26), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
            cv::putText(cell, captions[i], cv::Point(8, 26), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
    }
    return mosaic;
}
