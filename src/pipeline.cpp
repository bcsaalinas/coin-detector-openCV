#include "pipeline.hpp"

#include "coin_patch.hpp"
#include "color_filter.hpp"
#include "image_io.hpp"
#include "params.hpp"
#include "preprocessing.hpp"
#include "regions.hpp"
#include "report.hpp"
#include "sift_features.hpp"
#include "size_filter.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>  // getRotationMatrix2D moved here in 5.x
#endif

#include <cstdio>
#include <iostream>

static void waitForKey()
{
    std::cout << "  (press any key in an image window to continue)\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// The ROI coordinates are only valid for the resolution they were measured on.
static bool hasExpectedSize(const cv::Mat& image)
{
    if (image.cols == params::EXPECTED_IMAGE_WIDTH &&
        image.rows == params::EXPECTED_IMAGE_HEIGHT) {
        return true;
    }
    std::cerr << "[error] Expected a " << params::EXPECTED_IMAGE_WIDTH << " x "
              << params::EXPECTED_IMAGE_HEIGHT << " image but got " << image.cols
              << " x " << image.rows << ".\n"
              << "        The ROI rectangles in include/params.hpp were measured on the\n"
              << "        expected resolution; measure them again for this image.\n";
    return false;
}

// ==================================================================== H0 ==
bool runH0LoadImage(const Options& options, cv::Mat& original)
{
    std::cout << "[H0] Loading image\n";
    original = loadColorImage(options.imagePath);
    if (original.empty()) {
        return false;
    }
    printImageInfo("  original", original);
    if (!hasExpectedSize(original) || !ensureDirectory(options.outputDir)) {
        return false;
    }

    if (options.showWindows) {
        showImage("H0 - original", original);
        waitForKey();
    }
    return true;
}

// ==================================================================== H1 ==
bool runH1Regions(const Options& options, const cv::Mat& original, cv::Mat& searchRegion,
                  std::vector<ReferenceCoin>& references)
{
    std::cout << "\n[H1] Extracting regions of interest\n";
    const NamedRegion search{"search region", params::SEARCH_REGION};
    const NamedRegion referenceArea{"reference region", params::REFERENCE_REGION};

    searchRegion = extractRegion(original, search);
    if (searchRegion.empty()) {
        return false;
    }
    printImageInfo("  search region", searchRegion);

    // The coin types are whatever coins lie in the reference rectangle.
    references = findReferenceCoins(original, params::REFERENCE_REGION);
    if (references.empty()) {
        std::cerr << "[error] No coins found in the reference region\n";
        return false;
    }
    std::cout << "  reference coins found: " << references.size() << " (reading order)\n";
    for (const ReferenceCoin& reference : references) {
        std::printf("    %-8s outline at (%d, %d), %d x %d px\n", reference.name.c_str(),
                    reference.outline.x, reference.outline.y, reference.outline.width,
                    reference.outline.height);
    }

    std::vector<NamedRegion> allRegions = {search, referenceArea};
    for (const ReferenceCoin& reference : references) {
        allRegions.push_back({reference.name, reference.outline});
    }
    bool saved = saveImage(options.outputDir, "00_roi_overview.png",
                           drawRegionsOverview(original, allRegions));
    saved = saveImage(options.outputDir, "01_search_region.png", searchRegion) && saved;
    for (size_t i = 0; i < references.size(); ++i) {
        const std::string fileName =
            "02_reference_" + std::to_string(i + 1) + "_" + references[i].name + ".png";
        saved = saveImage(options.outputDir, fileName, references[i].crop) && saved;
    }

    if (options.showWindows) {
        showImage("H1 - search region", searchRegion);
        for (const ReferenceCoin& reference : references) {
            showImage("H1 - " + reference.name, reference.crop);
        }
        waitForKey();
    }
    return saved;
}

// ==================================================================== H2 ==
bool runH2Preprocessing(const Options& options, const cv::Mat& searchRegion, cv::Mat& smoothed)
{
    // Only the search region is preprocessed; the references are used in H4.
    std::cout << "\n[H2] Grayscale -> Gaussian smoothing -> Canny\n"
              << "  Gaussian kernel " << params::GAUSSIAN_KERNEL_SIZE << "x"
              << params::GAUSSIAN_KERNEL_SIZE << ", sigma " << params::GAUSSIAN_SIGMA
              << "; Canny thresholds low " << params::CANNY_LOW_THRESHOLD << " / high "
              << params::CANNY_HIGH_THRESHOLD << "\n";

    const cv::Mat gray = toGrayscale(searchRegion);
    smoothed = smoothGaussian(gray);
    const cv::Mat edges = detectEdgesCanny(smoothed);

    const double edgePercent = 100.0 * cv::countNonZero(edges) / edges.total();
    std::cout << "  edge pixels: " << cv::countNonZero(edges) << " (" << edgePercent
              << "% of the search region)\n";

    bool saved = saveImage(options.outputDir, "06_grayscale.png", gray);
    saved = saveImage(options.outputDir, "07_gaussian.png", smoothed) && saved;
    saved = saveImage(options.outputDir, "08_canny_edges.png", edges) && saved;
    saved = saveImage(options.outputDir, "10_preprocessing_comparison.png",
                      makeComparisonStrip(searchRegion, gray, smoothed, edges, 0.5)) &&
            saved;

    if (options.showWindows) {
        showImage("H2 - grayscale", gray);
        showImage("H2 - gaussian", smoothed);
        showImage("H2 - canny", edges);
        waitForKey();
    }
    return saved;
}

// ==================================================================== H3 ==
bool runH3Hough(const Options& options, const cv::Mat& searchRegion, const cv::Mat& smoothed,
                std::vector<CoinCandidate>& candidates)
{
    std::cout << "\n[H3] Hough circle transform\n"
              << "  dp " << params::HOUGH_DP << ", minDist " << params::HOUGH_MIN_DIST
              << ", param1 " << params::HOUGH_PARAM1 << ", param2 " << params::HOUGH_PARAM2
              << ", radius [" << params::HOUGH_MIN_RADIUS << ", "
              << params::HOUGH_MAX_RADIUS << "]\n";

    const std::vector<cv::Vec3f> circles = detectCircles(smoothed);
    candidates = selectCandidates(circles, smoothed.size());
    printCandidates(candidates, params::SEARCH_REGION.tl());

    // A circle is a geometric candidate, not a recognised coin type.
    std::cout << "\n  Geometric coin candidates: " << candidates.size() << "\n";

    const cv::Mat detections = drawCandidates(searchRegion, candidates);
    if (options.showWindows) {
        showImage("H3 - Hough candidates", detections);
        waitForKey();
    }
    return saveImage(options.outputDir, "09_hough_circles.png", detections);
}

// ================================================================= H4/H5 ==

// Control experiment: a reference against a rotated copy of itself. It shows
// what a real match looks like when both images share lighting and condition,
// so the scores of the photo coins can be judged against it.
static void runSelfCheck(const Options& options, const std::string& name,
                         const CoinPatch& patch, const CoinFeatures& features)
{
    const double angle = 137.0;  // any non-trivial rotation
    const cv::Point2f center(params::PATCH_DIAMETER / 2.0f, params::PATCH_DIAMETER / 2.0f);
    const cv::Mat rotation = cv::getRotationMatrix2D(center, angle, 1.0);

    CoinPatch rotated;
    cv::warpAffine(patch.gray, rotated.gray, rotation, patch.gray.size());
    cv::warpAffine(patch.mask, rotated.mask, rotation, patch.mask.size(), cv::INTER_NEAREST);
    const CoinFeatures rotatedFeatures = extractSift(rotated);

    const MatchResult result = matchToReference(rotatedFeatures, features, 0);
    std::printf("  self-check (%s reference vs itself rotated %.0f deg): %d inliers, "
                "match %.1f%%, scale %.2f\n",
                name.c_str(), angle, result.inliers, result.matchPercent, result.scale);
    saveImage(options.outputDir, "14_sift_self_check.png",
              drawMatchResult(rotated, rotatedFeatures, patch, features, result));
}

static void printClassification(const CoinCandidate& candidate, const Classification& result,
                                const std::vector<std::string>& names)
{
    if (result.comparisons.empty()) {
        std::printf("  %3d | %-8s | %5s | %7s | %7s | %5s | %s\n", candidate.id, "-", "-", "-",
                    "-", "-", result.label.c_str());
        return;
    }
    const std::string label =
        result.label + (result.evidence.empty() ? "" : " (" + result.evidence + ")");
    for (size_t k = 0; k < result.comparisons.size(); ++k) {
        const MatchResult& m = result.comparisons[k];
        std::printf("  %3s | %-8s | %5d | %7d | %6.1f%% | %5s | %s\n",
                    k == 0 ? std::to_string(candidate.id).c_str() : "",
                    names[m.type].c_str(), m.goodMatches, m.inliers, m.matchPercent,
                    m.geometryOk ? "ok" : "bad", k == 0 ? label.c_str() : "");
    }
}

static std::vector<std::string> namesOf(const std::vector<ReferenceCoin>& references)
{
    std::vector<std::string> names;
    for (const ReferenceCoin& reference : references) {
        names.push_back(reference.name);
    }
    return names;
}

static cv::Scalar metalDrawColor(Metal metal)
{
    if (metal == Metal::Copper) return cv::Scalar(0, 110, 255);
    if (metal == Metal::Gold) return cv::Scalar(0, 230, 255);
    return cv::Scalar(235, 235, 235);
}

// H4a' - measures the colour (centre and ring) of the references and of every
// candidate, and keeps only the types whose reference has the same metals.
static std::vector<Prefilter> runColorFilter(const Options& options, const cv::Mat& searchRegion,
                                             const std::vector<ReferenceCoin>& references,
                                             const std::vector<CoinCandidate>& candidates,
                                             const std::vector<SizeCheck>& sizeChecks,
                                             bool& saved)
{
    std::cout << "\n[H4a'] Colour pre-filter (Lab median of centre < "
              << params::COLOR_CENTER_TO << " r and ring " << params::COLOR_RING_FROM << "-"
              << params::COLOR_RING_TO << " r)\n"
              << "  rule: chroma < " << params::MIN_CHROMA_COLORED << " -> silver; else hue < "
              << params::MAX_HUE_COPPER_DEG << " deg -> copper; else gold\n";

    std::vector<ColorSignature> referenceColors;
    for (const ReferenceCoin& reference : references) {
        const ColorSignature s = measureReferenceColor(reference);
        referenceColors.push_back(s);
        std::printf("  reference %-8s centre chroma %5.1f hue %6.1f | ring chroma %5.1f hue "
                    "%6.1f -> %s\n",
                    reference.name.c_str(), s.center.chroma, s.center.hueDeg, s.ring.chroma,
                    s.ring.hueDeg, describeMetals(s).c_str());
    }

    cv::Mat searchLab;
    cv::cvtColor(searchRegion, searchLab, cv::COLOR_BGR2Lab);

    std::printf("   id | centre chroma/hue | ring chroma/hue | metal  | compared with SIFT\n");
    std::vector<Prefilter> prefilters;
    cv::Mat canvas = searchRegion.clone();
    int comparisons = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const ColorSignature color = measureCandidateColor(searchLab, candidates[i]);
        Prefilter prefilter;
        prefilter.types =
            filterTypesByColor(sizeChecks[i].compatibleTypes, color, referenceColors);
        if (sizeChecks[i].compatibleTypes.empty()) {
            prefilter.rejectedBy = "size";
        } else if (prefilter.types.empty()) {
            prefilter.rejectedBy = "color";
        }
        prefilter.colorDecisive =
            prefilter.types.size() == 1 && isDistinctiveColor(color, referenceColors);
        comparisons += static_cast<int>(prefilter.types.size());

        std::string kept;
        for (int type : prefilter.types) {
            kept += (kept.empty() ? "" : ", ") + references[type].name;
        }
        if (kept.empty()) {
            kept = prefilter.rejectedBy + "-rejected, no SIFT";
        } else if (prefilter.colorDecisive) {
            kept += "  (colour alone identifies it)";
        }
        const std::string metals = describeMetals(color);
        std::printf("  %3d | %6.1f / %5.1f    | %6.1f / %5.1f  | %-6s | %s\n", candidates[i].id,
                    color.center.chroma, color.center.hueDeg, color.ring.chroma,
                    color.ring.hueDeg, metals.c_str(), kept.c_str());

        // Circle in the ring colour; thick if the candidate goes on to SIFT.
        const cv::Point center(cvRound(candidates[i].center.x), cvRound(candidates[i].center.y));
        const cv::Scalar drawColor = metalDrawColor(color.ring.metal);
        cv::circle(canvas, center, cvRound(candidates[i].radius), drawColor,
                   prefilter.types.empty() ? 2 : 6, cv::LINE_AA);
        const std::string text = "#" + std::to_string(candidates[i].id) + " " + metals +
                                 (prefilter.types.empty() ? " X" : "");
        cv::putText(canvas, text, center + cv::Point(-70, 8), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 0, 0), 6, cv::LINE_AA);
        cv::putText(canvas, text, center + cv::Point(-70, 8), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    drawColor, 2, cv::LINE_AA);
        prefilters.push_back(prefilter);
    }

    const int allComparisons = static_cast<int>(candidates.size() * references.size());
    std::printf("  SIFT comparisons after diameter + colour: %d instead of %d (%.0f%% avoided)\n",
                comparisons, allComparisons,
                100.0 * (allComparisons - comparisons) / allComparisons);
    saved = saveImage(options.outputDir, "11b_color_filter.png", canvas) && saved;
    return prefilters;
}

bool runH4H5Recognition(const Options& options, const cv::Mat& searchRegion,
                        const std::vector<ReferenceCoin>& references,
                        const std::vector<CoinCandidate>& candidates,
                        std::vector<Classification>& results)
{
    const std::vector<std::string> names = namesOf(references);

    // ---- H4a: diameter pre-filter --------------------------------------
    std::cout << "\n[H4a] Diameter pre-filter in pixels (reference -> search scale "
              << params::REFERENCE_TO_SEARCH_SCALE << ", tolerance +/-"
              << cvRound(100 * params::SIZE_TOLERANCE) << "%)\n";
    printReferenceSizes(references);

    std::vector<SizeCheck> sizeChecks;
    for (const CoinCandidate& candidate : candidates) {
        sizeChecks.push_back(checkDiameter(candidate, references));
    }
    printSizeChecks(candidates, sizeChecks, references);
    bool saved = saveImage(options.outputDir, "11_size_filter.png",
                           drawSizeChecks(searchRegion, candidates, sizeChecks, references));

    // ---- H4a': colour pre-filter -----------------------------------------
    const std::vector<Prefilter> prefilters =
        runColorFilter(options, searchRegion, references, candidates, sizeChecks, saved);

    // ---- H4b: SIFT on the reference coins -------------------------------
    std::cout << "\n[H4b] SIFT on the reference coins (stretch corrected, "
              << params::PATCH_DIAMETER << " px diameter, CLAHE)\n";
    std::vector<CoinPatch> referencePatches;
    std::vector<CoinFeatures> referenceFeatures;
    std::vector<cv::Mat> tiles;
    std::vector<std::string> captions;
    for (const ReferenceCoin& reference : references) {
        double stretch = 1.0;
        referencePatches.push_back(makeReferencePatch(reference, stretch));
        referenceFeatures.push_back(extractSift(referencePatches.back()));
        const CoinFeatures& f = referenceFeatures.back();
        std::printf("  %-8s stretch h/w %.3f -> corrected | %4d keypoints | descriptors %d x %d\n",
                    reference.name.c_str(), stretch, static_cast<int>(f.keypoints.size()),
                    f.descriptors.rows, f.descriptors.cols);
        if (f.descriptors.empty()) {
            std::cerr << "[error] No SIFT descriptors for reference " << reference.name << "\n";
            return false;
        }
        tiles.push_back(drawSiftKeypoints(referencePatches.back(), f));
        captions.push_back(reference.name + " " + std::to_string(f.keypoints.size()) + " kp");
    }
    saved = saveImage(options.outputDir, "12_reference_keypoints.png",
                      makeMosaic(tiles, captions, static_cast<int>(references.size()))) &&
            saved;

    // ---- H4c: SIFT on the candidates that passed the pre-filters --------
    std::cout << "\n[H4c] SIFT on the candidates kept by both pre-filters\n";
    const cv::Mat searchGray = toGrayscale(searchRegion);
    std::vector<CoinPatch> candidatePatches(candidates.size());
    std::vector<CoinFeatures> candidateFeatures(candidates.size());
    tiles.clear();
    captions.clear();
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (prefilters[i].types.empty()) {
            continue;  // discarded by diameter or colour: no SIFT is computed
        }
        candidatePatches[i] = makeCandidatePatch(searchGray, candidates[i]);
        candidateFeatures[i] = extractSift(candidatePatches[i]);
        tiles.push_back(drawSiftKeypoints(candidatePatches[i], candidateFeatures[i]));
        captions.push_back("#" + std::to_string(candidates[i].id) + " " +
                           std::to_string(candidateFeatures[i].keypoints.size()) + " kp");
    }
    std::cout << "  SIFT computed for " << tiles.size() << " of " << candidates.size()
              << " candidates\n";
    saved = saveImage(options.outputDir, "13_candidate_keypoints.png",
                      makeMosaic(tiles, captions, 6)) && saved;

    // ---- H5: matching and classification --------------------------------
    std::cout << "\n[H5] Matching: Lowe ratio " << params::LOWE_RATIO << ", RANSAC "
              << params::RANSAC_REPROJECTION_PX << " px, accept with >= "
              << params::MIN_INLIERS << " inliers\n";
    runSelfCheck(options, names[0], referencePatches[0], referenceFeatures[0]);

    std::printf("   id | compared | ratio | RANSAC  | match   | geom  | label\n"
                "      | with     | match | inliers |         |       |\n");
    const std::string matchDir = options.outputDir + "/15_matches";
    saved = ensureDirectory(matchDir) && saved;
    results.clear();
    for (size_t i = 0; i < candidates.size(); ++i) {
        results.push_back(
            classifyCandidate(candidateFeatures[i], referenceFeatures, prefilters[i], names));
        printClassification(candidates[i], results.back(), names);

        // Evidence: the best comparison of every candidate that was matched.
        const Classification& result = results.back();
        if (!result.comparisons.empty()) {
            const int type = result.best.type;
            char number[8];
            std::snprintf(number, sizeof(number), "%02d", candidates[i].id);
            const std::string fileName =
                "candidate_" + std::string(number) + "_vs_" + names[type] + ".png";
            saved = saveImage(matchDir, fileName,
                              drawMatchResult(candidatePatches[i], candidateFeatures[i],
                                              referencePatches[type], referenceFeatures[type],
                                              result.best)) &&
                    saved;
        }
    }

    if (options.showWindows) {
        showImage("H4 - candidate keypoints", makeMosaic(tiles, captions, 6));
        waitForKey();
    }
    return saved;
}

// ==================================================================== H6 ==
bool runH6Report(const Options& options, const cv::Mat& searchRegion,
                 const std::vector<ReferenceCoin>& references,
                 const std::vector<CoinCandidate>& candidates,
                 const std::vector<Classification>& results)
{
    const std::vector<std::string> names = namesOf(references);
    std::cout << "\n[H6] Coins recognised per reference type (each candidate counts once)\n";
    printCounts(results, names);

    const cv::Mat finalImage = drawFinalResult(searchRegion, candidates, results, names);
    const bool saved =
        saveImage(options.outputDir, "16_final_classification.png", finalImage);

    // The ground truth is only used here, to score the result; the
    // classification above never reads it.
    std::cout << "\n[H6] Evaluation against the manual ground truth\n  "
              << options.groundTruthPath << "\n";
    evaluateAgainstGroundTruth(candidates, results, loadGroundTruth(options.groundTruthPath),
                               names);

    if (options.showWindows) {
        showImage("H6 - final classification", finalImage);
        waitForKey();
    }
    return saved;
}
