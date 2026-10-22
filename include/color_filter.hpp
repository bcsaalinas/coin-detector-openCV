#pragma once

// H4a' - Colour pre-filter.
//
// The metal colour is a cheap, rotation-independent cue. Each reference coin's
// colour is measured with the same rule as the candidates, so the program
// learns from the reference rectangle which types are copper, silver, gold or
// bimetallic; nothing is known in advance about any coin type.

#include "circle_detection.hpp"
#include "regions.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

enum class Metal { Silver, Copper, Gold };

const char* metalName(Metal metal);

struct ColorSample {
    double a = 0.0;       // median a* (green -> red), 0 = neutral
    double b = 0.0;       // median b* (blue -> yellow), 0 = neutral
    double chroma = 0.0;  // sqrt(a^2 + b^2)
    double hueDeg = 0.0;  // atan2(b, a) in degrees
    Metal metal = Metal::Silver;
};

// Centre and ring are measured separately (bimetallic coins).
struct ColorSignature {
    ColorSample center;
    ColorSample ring;
};

// "copper" for a one-metal coin, "silver/gold" (centre/ring) for bimetallic.
std::string describeMetals(const ColorSignature& signature);

// Colour of a candidate circle. `searchLab` is the search region converted
// with cv::COLOR_BGR2Lab.
ColorSignature measureCandidateColor(const cv::Mat& searchLab, const CoinCandidate& candidate);

// Colour of a reference coin (its outline is an ellipse; only its own pixels).
ColorSignature measureReferenceColor(const ReferenceCoin& reference);

// Keeps only the types whose reference has the same centre and ring metals.
std::vector<int> filterTypesByColor(const std::vector<int>& types,
                                    const ColorSignature& candidate,
                                    const std::vector<ColorSignature>& references);

// True if exactly one reference type has these metals, so the colour alone
// identifies the type (e.g. the only copper coin of the reference rectangle).
bool isDistinctiveColor(const ColorSignature& signature,
                        const std::vector<ColorSignature>& references);
