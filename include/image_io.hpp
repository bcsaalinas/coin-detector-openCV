#pragma once

// Small helpers to load, save and (optionally) display images.

#include <opencv2/core.hpp>

#include <string>

// Loads an image as 3-channel BGR. Returns an empty cv::Mat and prints the
// reason if the file is missing or cannot be decoded.
cv::Mat loadColorImage(const std::string& path);

// Prints size, channels and pixel type of an image.
void printImageInfo(const std::string& label, const cv::Mat& image);

// Creates the directory (and its parents) if it does not exist yet.
bool ensureDirectory(const std::string& directory);

// Writes `image` to `directory/fileName`. Returns false and prints the reason
// on failure.
bool saveImage(const std::string& directory, const std::string& fileName,
               const cv::Mat& image);

// Shows a (downscaled if needed) copy of the image in a window. Only used when
// the program runs with --show; the pipeline never depends on a GUI.
void showImage(const std::string& windowTitle, const cv::Mat& image);
