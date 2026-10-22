#include "report.hpp"

#include "params.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

std::vector<GroundTruthCoin> loadGroundTruth(const std::string& path)
{
    std::vector<GroundTruthCoin> coins;
    std::ifstream file(path);
    if (!file) {
        std::cerr << "[warning] Ground truth not found: " << path
                  << " (evaluation skipped)\n";
        return coins;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream fields(line);
        GroundTruthCoin coin;
        std::string x;
        std::string y;
        std::getline(fields, coin.type, ',');
        std::getline(fields, x, ',');
        std::getline(fields, y, ',');
        std::getline(fields, coin.note);
        coin.position = cv::Point2f(std::stof(x), std::stof(y));
        coins.push_back(coin);
    }
    return coins;
}

void printCounts(const std::vector<Classification>& results,
                 const std::vector<std::string>& names)
{
    std::vector<int> bySift(names.size(), 0);
    std::vector<int> byColor(names.size(), 0);
    int unknown = 0;
    int ambiguous = 0;
    int sizeRejected = 0;
    int colorRejected = 0;
    for (const Classification& result : results) {
        if (result.type >= 0) {
            ++(result.evidence == "SIFT" ? bySift : byColor)[result.type];
        } else if (result.label == "ambiguous") {
            ++ambiguous;
        } else if (result.label == "size-rejected") {
            ++sizeRejected;
        } else if (result.label == "color-rejected") {
            ++colorRejected;
        } else {
            ++unknown;
        }
    }

    int recognised = 0;
    for (size_t type = 0; type < names.size(); ++type) {
        const int count = bySift[type] + byColor[type];
        std::printf("  %-9s %d   (SIFT %d, size+color %d)\n", (names[type] + ":").c_str(),
                    count, bySift[type], byColor[type]);
        recognised += count;
    }
    std::printf("  Unknown:  %d  (+ %d ambiguous; discarded: %d by diameter, %d by colour)\n",
                unknown, ambiguous, sizeRejected, colorRejected);
    std::printf("  Total recognised: %d of %zu candidates\n", recognised, results.size());
    std::fflush(stdout);
}

static cv::Scalar labelColor(const Classification& result)
{
    // One colour per reference type (cycled if there are more types).
    const cv::Scalar palette[] = {cv::Scalar(0, 120, 255), cv::Scalar(255, 200, 0),
                                  cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 0),
                                  cv::Scalar(255, 120, 120), cv::Scalar(120, 255, 255)};
    if (result.type >= 0) return palette[result.type % 6];
    if (result.label == "ambiguous") return cv::Scalar(0, 255, 255);
    if (result.label == "size-rejected" || result.label == "color-rejected") {
        return cv::Scalar(0, 0, 255);
    }
    return cv::Scalar(235, 235, 235);  // unknown
}

static void putOutlinedText(cv::Mat& image, const std::string& text, cv::Point origin,
                            double scale, const cv::Scalar& color)
{
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0),
                7, cv::LINE_AA);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, 2, cv::LINE_AA);
}

cv::Mat drawFinalResult(const cv::Mat& searchColor,
                        const std::vector<CoinCandidate>& candidates,
                        const std::vector<Classification>& results,
                        const std::vector<std::string>& names)
{
    cv::Mat canvas = searchColor.clone();
    for (size_t i = 0; i < candidates.size(); ++i) {
        const CoinCandidate& candidate = candidates[i];
        const Classification& result = results[i];
        const cv::Point center(cvRound(candidate.center.x), cvRound(candidate.center.y));
        const cv::Scalar color = labelColor(result);
        const bool recognised = result.type >= 0;

        cv::circle(canvas, center, cvRound(candidate.radius), color, recognised ? 8 : 3,
                   cv::LINE_AA);

        std::string text = "#" + std::to_string(candidate.id) + " " + result.label;
        putOutlinedText(canvas, text, center + cv::Point(-75, -10), 1.0, color);
        if (!result.comparisons.empty()) {
            // Second line: what the decision rests on, and the best SIFT score.
            char details[64];
            if (recognised) {
                std::snprintf(details, sizeof(details), "%s, SIFT %.1f%%",
                              result.evidence.c_str(), result.best.matchPercent);
            } else {
                std::snprintf(details, sizeof(details), "SIFT %s %.1f%%",
                              names[result.best.type].c_str(), result.best.matchPercent);
            }
            putOutlinedText(canvas, details, center + cv::Point(-75, 28), 0.85, color);
        }
    }

    // Banner with the final counts.
    std::vector<int> counts(names.size(), 0);
    int others = 0;
    for (const Classification& result : results) {
        if (result.type >= 0) {
            ++counts[result.type];
        } else {
            ++others;
        }
    }
    std::string banner;
    for (size_t type = 0; type < names.size(); ++type) {
        banner += names[type] + " " + std::to_string(counts[type]) + "  ";
    }
    banner += "| not recognised " + std::to_string(others);
    cv::rectangle(canvas, cv::Rect(0, 0, canvas.cols, 60), cv::Scalar(30, 30, 30), cv::FILLED);
    putOutlinedText(canvas, banner, cv::Point(15, 42), 1.0, cv::Scalar(255, 255, 255));
    return canvas;
}

void evaluateAgainstGroundTruth(const std::vector<CoinCandidate>& candidates,
                                const std::vector<Classification>& results,
                                const std::vector<GroundTruthCoin>& groundTruth,
                                const std::vector<std::string>& names)
{
    if (groundTruth.empty()) {
        return;
    }

    // A ground-truth coin is "detected" by the candidate whose circle contains
    // its centre (within half a radius).
    std::vector<int> candidateOfTruth(groundTruth.size(), -1);
    for (size_t g = 0; g < groundTruth.size(); ++g) {
        for (size_t c = 0; c < candidates.size(); ++c) {
            const cv::Point2f d = candidates[c].center - groundTruth[g].position;
            if (std::hypot(d.x, d.y) < 0.5 * candidates[c].radius) {
                candidateOfTruth[g] = static_cast<int>(c);
            }
        }
    }

    int truePositives = 0;
    int detectionErrors = 0;
    int classificationErrors = 0;
    std::printf("  ground truth                 | candidate | predicted     | result\n");
    for (size_t g = 0; g < groundTruth.size(); ++g) {
        const GroundTruthCoin& truth = groundTruth[g];
        const int c = candidateOfTruth[g];
        std::string predicted = "-";
        std::string verdict;
        if (c < 0) {
            verdict = "DETECTION error (no Hough circle)";
            ++detectionErrors;
        } else {
            predicted = results[c].label;
            if (predicted == truth.type) {
                verdict = "correct";
                ++truePositives;
            } else {
                verdict = "CLASSIFICATION error";
                ++classificationErrors;
            }
        }
        std::printf("  %-8s at (%4.0f, %4.0f)     | %9s | %-13s | %s\n", truth.type.c_str(),
                    truth.position.x, truth.position.y,
                    c < 0 ? "-" : ("#" + std::to_string(candidates[c].id)).c_str(),
                    predicted.c_str(), verdict.c_str());
    }

    // Every candidate labelled with a reference type that is not a true
    // positive is a false positive (a wrong type, or a coin of no reference type).
    int predictedUs = 0;
    for (const Classification& result : results) {
        if (result.type >= 0) {
            ++predictedUs;
        }
    }
    const int falsePositives = predictedUs - truePositives;

    std::printf("\n  coins in ground truth: %zu, recognised correctly: %d, false positives: %d\n",
                groundTruth.size(), truePositives, falsePositives);
    std::printf("  detection errors (H3): %d, classification errors (H5): %d\n",
                detectionErrors, classificationErrors);
    if (predictedUs > 0) {
        std::printf("  precision: %.1f%% (%d of %d recognised coins are right)\n",
                    100.0 * truePositives / predictedUs, truePositives, predictedUs);
    } else {
        std::printf("  precision: undefined (no coin was accepted as any reference type)\n");
    }
    std::printf("  recall:    %.1f%% (%d of %zu reference-type coins found)\n",
                100.0 * truePositives / groundTruth.size(), truePositives,
                groundTruth.size());

    // Same measures per coin type.
    std::printf("\n  type     | truth | predicted | correct | precision | recall\n");
    for (size_t type = 0; type < names.size(); ++type) {
        const std::string& name = names[type];
        int truth = 0;
        int correct = 0;
        for (size_t g = 0; g < groundTruth.size(); ++g) {
            if (groundTruth[g].type == name) {
                ++truth;
                const int c = candidateOfTruth[g];
                correct += (c >= 0 && results[c].type == static_cast<int>(type)) ? 1 : 0;
            }
        }
        int predicted = 0;
        for (const Classification& result : results) {
            predicted += result.type == static_cast<int>(type) ? 1 : 0;
        }
        char precision[16] = "n/a";
        char recall[16] = "n/a";
        if (predicted > 0) {
            std::snprintf(precision, sizeof(precision), "%.1f%%", 100.0 * correct / predicted);
        }
        if (truth > 0) {
            std::snprintf(recall, sizeof(recall), "%.1f%%", 100.0 * correct / truth);
        }
        std::printf("  %-8s | %5d | %9d | %7d | %9s | %6s\n", name.c_str(), truth, predicted,
                    correct, precision, recall);
    }
    std::fflush(stdout);
}
