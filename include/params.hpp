#pragma once

// All tunable values of the pipeline live in this file, so an experiment only
// needs to change one place. Record every change in the project log
// (PLAN_PROYECTO_CONTEO_MONEDAS.md) together with the observed result.

#include <opencv2/core.hpp>

#include <string>
#include <vector>

// Everything here is either measured on the input image or a tuning value.
// No outside knowledge about real coins (sizes in mm, metals, designs) is
// used: the reference rectangle is the only source of what a coin type is.

namespace params {

// ---------------------------------------------------------------------------
// H0 - Input image
// ---------------------------------------------------------------------------
// The ROI coordinates below were measured on this exact resolution. If a
// different version of the image is used they must be measured again.
constexpr int EXPECTED_IMAGE_WIDTH = 2477;
constexpr int EXPECTED_IMAGE_HEIGHT = 1929;

// ---------------------------------------------------------------------------
// H1 - Regions of interest: cv::Rect(x, y, width, height) in full-image pixels
// ---------------------------------------------------------------------------
// Search region: the wooden photo on the left, just inside its dark frame
// (frame columns 92-98 / 1169-1175, frame row 126-127, photo ends at row 1722).
// These match the plan's (86, 106, 880, 1317) scaled by 2477 / 2048 = 1.2095.
const cv::Rect SEARCH_REGION(104, 128, 1064, 1595);

// Reference region: the white panel inside the top-right box. Every coin found
// in it becomes one coin type, so any set of references works (US, Mexican,
// ...) without changing the code.
const cv::Rect REFERENCE_REGION(1550, 207, 850, 723);

// The reference coins lie on a white background: darker pixels are coin.
constexpr int REFERENCE_WHITE_THRESHOLD = 235;

// Blobs smaller than this fraction of the reference region are ignored (specks).
constexpr double REFERENCE_MIN_AREA_FRACTION = 0.005;

// Labels for the reference coins in reading order (top row left -> right,
// then the next row). They are only names for the output; if the number of
// coins found is different, generic names "type_1", "type_2", ... are used.
const std::vector<std::string> REFERENCE_NAMES = {"penny", "nickel", "dime", "quarter"};

// ---------------------------------------------------------------------------
// H2 - Grayscale, Gaussian smoothing and Canny
// ---------------------------------------------------------------------------
// Gaussian low-pass filter. Larger sigma removes more wood-grain texture but
// also blurs the coin rims. Kernel size must be odd (about 4.5 * sigma here).
// Plan start (5x5, sigma 1.5) kept too much wood grain and coin engraving.
constexpr int GAUSSIAN_KERNEL_SIZE = 9;
constexpr double GAUSSIAN_SIGMA = 2.0;

// Canny hysteresis: gradients above HIGH start an edge, gradients above LOW
// continue it. LOW = HIGH / 2 on purpose: HoughCircles (HOUGH_GRADIENT) runs
// Canny(param1 / 2, param1) internally, so with HOUGH_PARAM1 = CANNY_HIGH the
// saved edge map is exactly the one the Hough transform votes with.
constexpr double CANNY_HIGH_THRESHOLD = 120.0;
constexpr double CANNY_LOW_THRESHOLD = CANNY_HIGH_THRESHOLD / 2.0;

// ---------------------------------------------------------------------------
// H3 - Hough transform for circles (cv::HOUGH_GRADIENT)
// ---------------------------------------------------------------------------
// dp: inverse resolution of the center accumulator (1 = same as the image,
//     1.2 = slightly coarser cells, so votes of a rim gather more easily).
//     dp >= 1.5 produced dozens of false circles on this image.
constexpr double HOUGH_DP = 1.2;

// Minimum distance between two detected centers, in pixels. Two different
// coins cannot have centers closer than roughly the smallest radius.
// 60 px produced duplicate circles on the same coin; 80-150 px gave none.
constexpr double HOUGH_MIN_DIST = 100.0;

// param1: high threshold of the internal Canny (low = param1 / 2).
constexpr double HOUGH_PARAM1 = CANNY_HIGH_THRESHOLD;

// param2: minimum number of accumulator votes for a center. Lower values find
// more (partial) coins but also circles in the wood grain.
// Tested on this image: <= 32 -> false positives, 35 -> 18 circles,
// 40 -> 17 circles, 45-50 -> 15 circles, 55-60 -> 14 circles (no false
// positives from 33 upwards). 40 keeps a safety margin from the noisy range.
constexpr double HOUGH_PARAM2 = 40.0;

// Radius range in pixels. Coins in the search region measure ~115-200 px.
constexpr int HOUGH_MIN_RADIUS = 95;
constexpr int HOUGH_MAX_RADIUS = 220;

// Explicit duplicate check after Hough: a circle whose center lies closer than
// this fraction of the smaller radius to a stronger circle is the same coin.
constexpr double DUPLICATE_CENTER_FRACTION = 0.5;

// ---------------------------------------------------------------------------
// H4a - Diameter pre-filter (runs before any SIFT computation)
// ---------------------------------------------------------------------------
// Candidate and reference diameters are compared in pixels, assuming both
// rectangles show coins at the same zoom. Change this factor (search pixels
// per reference pixel) if the reference is drawn bigger or smaller.
constexpr double REFERENCE_TO_SEARCH_SCALE = 1.0;

// The reference coins are drawn as ellipses (e.g. 205 x 250 px) and the image
// alone cannot tell which axis is the real diameter, so a candidate fits a
// reference if its diameter lies between the two axes, widened by this
// relative tolerance (covers the Hough radius error and small zoom changes).
constexpr double SIZE_TOLERANCE = 0.10;

// ---------------------------------------------------------------------------
// H4a' - Colour pre-filter (silver / copper / gold, centre and ring)
// ---------------------------------------------------------------------------
// Colour is measured in CIE Lab: a* = green..red, b* = blue..yellow.
// chroma = sqrt(a*^2 + b*^2) is how coloured the metal is, and
// hue = atan2(b*, a*) says which colour (0 deg = red, 90 deg = yellow).
// Two zones are measured separately so bimetallic coins (e.g. Mexican pesos:
// silver centre, gold ring) are described correctly. Radii are fractions of
// the coin radius; the median ignores neighbours lying on top of the coin.
constexpr double COLOR_CENTER_TO = 0.45;
constexpr double COLOR_RING_FROM = 0.60;
constexpr double COLOR_RING_TO = 0.85;

// Below this chroma a coin is silver/neutral. Measured on this photo: silver
// coins 4-13, coloured coins 24-47 (threshold in the gap).
constexpr double MIN_CHROMA_COLORED = 18.0;

// Coloured coins with a hue below this are copper/bronze, above it brass/gold.
// Measured: hues 65-75 deg vs 83-90 deg (threshold in the largest gap).
constexpr double MAX_HUE_COPPER_DEG = 79.0;

// ---------------------------------------------------------------------------
// H4b - Coin patches for SIFT
// ---------------------------------------------------------------------------
// Every coin (reference or candidate) is resampled to this diameter so that
// matching features have the same scale, and centred in a square patch.
constexpr int PATCH_DIAMETER = 300;

// Only pixels inside this fraction of the radius are used, so the background
// around the coin cannot produce features.
constexpr double PATCH_MASK_FRACTION = 0.96;

// CLAHE (contrast-limited adaptive histogram equalisation) before SIFT. The
// photo coins are washed out compared with the studio references; equalising
// local contrast in 8 x 8 tiles brings both to a similar range.
constexpr double CLAHE_CLIP_LIMIT = 2.0;
constexpr int CLAHE_TILES = 8;

// ---------------------------------------------------------------------------
// H5 - Matching and classification
// ---------------------------------------------------------------------------
// Lowe's ratio test: keep a match only if the best reference descriptor is
// clearly closer than the second best (distance1 < RATIO * distance2).
constexpr double LOWE_RATIO = 0.8;

// RANSAC fits a similarity transform (rotation + uniform scale + shift)
// between reference and candidate keypoints. A match is an inlier if the
// transform maps it within this many pixels (2 % of the patch diameter).
constexpr double RANSAC_REPROJECTION_PX = 6.0;

// Both coins are centred and normalised to the same diameter, so the correct
// transform is (almost) a pure rotation about the centre. Transforms outside
// these limits are geometrically impossible and are rejected.
constexpr double MIN_TRANSFORM_SCALE = 0.8;
constexpr double MAX_TRANSFORM_SCALE = 1.25;
constexpr double MAX_CENTER_SHIFT_PX = 18.0;

// Acceptance threshold, chosen from measured controls (see the plan, H5):
//   coins with different designs                 -> at most 6 consistent matches
//   two photo coins with the same design (#10/#11) -> 13 consistent matches
constexpr int MIN_INLIERS = 10;

// If two coin types are accepted, the best must have this many times the
// inliers of the second; otherwise the candidate is labelled "ambiguous".
constexpr double AMBIGUITY_FACTOR = 1.5;

// ---------------------------------------------------------------------------
// H7 - Live camera mode (coin_camera)
// ---------------------------------------------------------------------------
// The camera looks at a sheet of paper with two printed rectangles (dark
// outlines). The reference coins are placed in one, the coins to identify in
// the other. Only what the camera sees is used.

// A printed rectangle must cover at least this fraction of the frame.
constexpr double PAPER_MIN_RECT_AREA_FRACTION = 0.02;

// approxPolyDP tolerance (fraction of the contour perimeter) when deciding
// whether an outline is a quadrilateral.
constexpr double PAPER_POLY_EPSILON = 0.02;

// Which of the two rectangles holds the reference coins.
constexpr bool REFERENCE_IS_SMALLER_RECT = true;

// Both rectangles are warped to one top-down view of the paper (a single
// homography), in which the search rectangle is this wide. Same view = same
// scale, so pixel diameters of reference and search coins are comparable.
constexpr int LIVE_SEARCH_WIDTH = 800;

// Height / width of the printed search rectangle. 0 = automatic: the program
// makes the reference coins round in the top-down view (coins are circles),
// which corrects the foreshortening of a tilted camera. Set it only if you
// prefer a fixed value for your printout.
constexpr double PAPER_SEARCH_ASPECT = 0.0;

// Fraction cut from each side of a rectified rectangle to drop its printed line.
constexpr double LIVE_RECT_INSET = 0.03;

// Hough radius range for the REFERENCE rectangle, as a fraction of
// LIVE_SEARCH_WIDTH (independent of how far the camera is). The search
// rectangle does not use it: there the range comes from the reference coins
// found (their radii +/- SIZE_TOLERANCE), so only coins of a possible size
// are looked for.
constexpr double LIVE_MIN_RADIUS_FRACTION = 0.04;
constexpr double LIVE_MAX_RADIUS_FRACTION = 0.20;

// Minimum fraction of a circle's rim lying on Canny edges (rimSupport).
// Reference coins lie apart on plain paper, so their rims are complete;
// search coins may overlap, so less of the rim is required there.
constexpr double LIVE_REFERENCE_MIN_RIM_SUPPORT = 0.6;
constexpr double LIVE_SEARCH_MIN_RIM_SUPPORT = 0.5;

// Canny high threshold (= HoughCircles param1) in the live view. Camera images
// have much less contrast than the still image: with 120 the rims of silver
// coins on paper were not found at all. In a simulated camera view 40-50
// found all reference coins, 30 and 60 missed some. Tune it with the slider.
constexpr double LIVE_HOUGH_PARAM1 = 45.0;

// Votes needed for a circle centre in the live view (smaller images give
// fewer votes than the 2477 px still image, hence lower than HOUGH_PARAM2).
constexpr double LIVE_HOUGH_PARAM2 = 35.0;

// SIFT contrast threshold in live mode. Coins seen by a camera have little
// texture; 0.02 (instead of the default 0.04) gave 4-5x more keypoints on
// silver coins in the simulated camera view without creating false matches.
constexpr double LIVE_SIFT_CONTRAST_THRESHOLD = 0.02;

// Inliers needed to accept a type in live mode. With the geometry checks (the
// match must be a rotation about the coin centre), wrong pairs had 0
// consistent inliers in the simulation; a different penny than the reference
// had 6, exact copies 16-49. 5 accepts the real case with a margin.
constexpr int LIVE_MIN_INLIERS = 5;

// Lowe ratio in live mode (still image: 0.8). Real coins under a camera
// produce fewer clear-cut matches; a looser ratio lets more through, and the
// RANSAC + "rotation about the centre" check still rejects wrong ones.
// Simulated camera, other penny (#11), share of frames recognised:
//   0.80 -> 19 %, 0.85 -> 74-81 %, 0.90 -> 93-96 %; wrong types: 0 in all.
// Wrong types only appeared when LIVE_MIN_INLIERS was lowered to 3.
constexpr double LIVE_LOWE_RATIO = 0.85;

// RootSIFT (Arandjelovic & Zisserman 2012): L1-normalise each descriptor and
// take the square root, so Euclidean distance compares histograms more
// fairly. Small gain in the simulation (#11: 74 % -> 81 % at ratio 0.85).
constexpr bool LIVE_ROOT_SIFT = true;

// The decision in live mode is Hough + SIFT. The colour pre-filter is optional
// (its thresholds were measured on the still image, not on your camera).
constexpr bool LIVE_USE_COLOR_FILTER = false;

// Reference SIFT features are cached; they are recomputed only when the
// reference coins change (count, or a centre moving more than this many px).
constexpr double LIVE_REFERENCE_MOVE_PX = 8.0;

// A changed set of reference coins must be seen in this many consecutive
// frames before the references are re-learned (a coin missed in one frame
// should not reset everything). Press 'r' to re-learn immediately.
constexpr int LIVE_REFERENCE_CONFIRM_FRAMES = 5;

// Each search coin is followed from frame to frame by its position in the
// top-down view (which is fixed to the paper). Its CONFIDENCE for a type is
// the share of its last LIVE_VOTE_FRAMES frames in which SIFT recognised it
// as that type (frames not seen yet count as "not recognised").
constexpr int LIVE_VOTE_FRAMES = 10;

// A type is shown when its confidence reaches this value. Real coins under a
// camera are recognised in some frames only; 0.3 = 3 of the last 10 frames.
constexpr double LIVE_MIN_CONFIDENCE = 0.3;

} // namespace params
