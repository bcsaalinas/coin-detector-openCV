# Coin detector — OpenCV / C++

Undergraduate Computer Vision project. There are two parts: a
**reference rectangle** with the coin types to look for, and a bigger
**search rectangle** with the coins to identify. The program counts how many
coins of each reference type are in the search rectangle. It uses methods from
the course: Canny, Hough, SIFT, plus simple colour analysis.

There are two programs:

| Program | Input | Milestones |
|---|---|---|
| `coin_detector` | the supplied image `assets/coins.png` | H0–H6 |
| `coin_camera` | **a live camera** looking at a printed sheet with the two rectangles | H7 ([Live camera mode](#live-camera-mode-coin_camera)) |

**Only information from the image is used.** The coin types are whatever
coins the program finds in the reference rectangle: US, Mexican or any other.
It uses no real-world sizes, metals or designs. Diameters are compared in
pixels, and each type's colour is measured on its reference coin.

**State: the full pipeline H0–H6 runs.** On the supplied image (references:
penny, nickel, dime, quarter) it finds **both pennies (penny recall 100 %)**,
but also labels 5 foreign copper coins of similar size as pennies. It misses
the quarter, which shows its back side. Overall: **precision 28.6 %,
recall 66.7 %**. SIFT alone recognised nothing; the pennies come from the
size + colour cue (see [Results](#results-on-the-supplied-image)).
[`PLAN_PROYECTO_CONTEO_MONEDAS.md`](PLAN_PROYECTO_CONTEO_MONEDAS.md) holds the
milestones and the full test log.

```text
image ─► H1 search rectangle ─► H2 gray/Gauss/Canny ─► H3 HoughCircles ─► candidates
     └─► H1 reference rectangle ─► coins found automatically (reading order) ─► types
                                                                                │
     H4a  candidate diameter vs reference diameters (pixels) ◄─────────────────┘
     H4a' centre + ring metal colour vs each reference's colour
           │  no compatible type left: discarded, no SIFT computed
           ▼
     H4   SIFT on normalised patches (references + kept candidates)
           ▼
     H5   ratio test ─► RANSAC ─► SIFT label; else size+colour if decisive
           ▼
     H6   counts, final image, precision / recall vs manual ground truth
```

## Requirements

| Tool | Tested version | Install (macOS / Homebrew) |
|---|---|---|
| C++17 compiler | Apple clang 21.0 | `xcode-select --install` |
| CMake ≥ 3.21 | 4.4.2 | `brew install cmake` |
| OpenCV (core, imgproc, imgcodecs, highgui, videoio, features, geometry) | 5.0.0 | `brew install opencv` |

OpenCV 4.x also works: CMake then uses `features2d` and `calib3d` instead of
`features` and `geometry`. On Debian/Ubuntu,
`sudo apt install build-essential cmake libopencv-dev` gives the same setup.

## Build and run

```bash
cmake --preset release
cmake --build --preset release
./build/release/coin_detector
```

For a debug build, use the `debug` preset instead (binary in
`build/debug/`). The program runs in about 1.5 s.

| Option | Meaning |
|---|---|
| `--image <path>` | input image (default `assets/coins.png`) |
| `--truth <csv>` | manual ground truth for scoring (default `assets/ground_truth.csv`) |
| `--output <dir>` | folder for generated images (default `output/`, created if missing) |
| `--show` | also open a window per stage; press a key to continue |

Default paths are fixed by CMake at configure time, so the program works from
any working directory; configure again if you move the project.

### Using other reference coins (e.g. Mexican)

Everything is configured in [`include/params.hpp`](include/params.hpp):

1. `SEARCH_REGION` and `REFERENCE_REGION`: the two rectangles. The
   reference rectangle needs a plain light background around the coins.
2. `REFERENCE_NAMES`: labels in reading order (top row left → right, then
   the next row). If the count doesn't match, the types are called
   `type_1`, `type_2`, …
3. `REFERENCE_TO_SEARCH_SCALE`: change this only if the reference coins are
   drawn at a different zoom than the photo.
4. `EXPECTED_IMAGE_WIDTH/HEIGHT`: the size of the new image.

Bimetallic coins are supported: centre and ring colours are compared
separately (e.g. a silver centre with a gold ring). Removing one reference
coin from the image was tested: the program finds 3 types and runs without
any code change.

## Live camera mode (`coin_camera`)

Print a sheet with **two rectangles drawn with a thick dark line** (the same
layout as the supplied image). Place the **reference coins in the smaller
rectangle**, apart from each other, and the coins to identify in the bigger
one. Point the camera at the sheet: overhead is best, but a tilt is fine.

```bash
./build/release/coin_camera
```

| Option | Meaning |
|---|---|
| `--camera <n>` | camera index (default 0) |
| `--video <file>` | process a recorded video instead |
| `--image <file>` | process one photo; saves `output/camera_result.png` and the two top-down views |
| `--frames <n>` | stop after n frames |
| `--no-gui` | no window: print a line when the result changes; save the last frame |

| Key | Action |
|---|---|
| `q` / `Esc` | quit |
| `r` | re-learn the reference coins now (also after changing a slider) |
| `v` | show / hide the top-down views of both rectangles |
| `s` | save a snapshot to `output/` |

Every coin in the search rectangle shows its **confidence**, e.g.
`penny 80%`: the share of the last 10 frames in which SIFT recognised it as
that type. The type is shown from 30 % (3 of 10 frames). Below that, the best
guess is still shown in grey (`? penny 20%`). With `--image`, there is only
one frame, so the confidence is 100 % if SIFT recognised the coin in that
photo and 0 % otherwise.

A **controls window** has sliders (like the course's `InteractiveDetector`)
for the Canny high threshold, Hough votes, rim support, minimum SIFT inliers
and the Lowe ratio. Tune them on your own camera, then copy the values into the `LIVE_*`
constants of [`include/params.hpp`](include/params.hpp).

On macOS, the first run asks to allow camera access for the Terminal app.

### How the live mode works (every frame)

1. **Find the sheet.** Canny + contours + `approxPolyDP` find four-sided
   outlines. The two largest that don't overlap are the printed rectangles;
   the sheet's own edge contains both, so it is never chosen. The smaller one
   is the reference (`REFERENCE_IS_SMALLER_RECT`).
2. **Top-down view.** One homography (`getPerspectiveTransform`) maps the sheet
   to a view from directly above, so both rectangles share the same scale. A
   tilted camera foreshortens the rectangle, and only the image is available,
   so the program corrects the vertical scale until the reference coins are
   **round** (coins are circles, the same assumption Hough makes).
3. **Reference coins: Hough** in the reference view. Each circle must have
   most of its rim on Canny edges (rim support ≥ 60 %), and circles may not
   overlap, since reference coins lie apart. Names come in reading order.
   **Their SIFT features are cached** and only re-learned when a different
   set of reference coins is seen for 5 frames in a row, or when you press `r`.
4. **Search coins: Hough**, searching only for radii that some reference has
   (±10 %), with rim support ≥ 50 %.
5. **Diameter pre-filter, then SIFT** (RootSIFT descriptors, ratio test 0.85,
   RANSAC + "rotation about the centre" check) against the size-compatible
   references. Accepted at ≥ 5 inliers. The size+colour shortcut of the still
   image is off: **Hough + SIFT decide**.
6. **Confidence.** Each coin is followed by its position in the top-down view.
   Its confidence is the share of its last 10 frames recognised as a type,
   and the label is shown from 30 %. With `--no-gui`, a per-coin report
   (frames seen, how often recognised) is printed at the end.

### What was tested (no camera was available during development)

* `coins.png` as a "camera frame": both rectangles are found and the views are
  corrected. Its stock reference coins are stretched ellipses, so few pass the
  rim check; real coins under the camera are round.
* **A simulated camera.** A synthetic sheet whose reference rectangle holds 3
  round coins cut from the photo (a US penny, an Austrian coin, a Dutch
  25 cent) was viewed tilted at 1280 × 720 with blur and noise, both as a
  still and as a 40-frame moving video.
  - The coin aspect correction was within 1.8 % of the true rectangle shape.
  - All 3 reference coins were found.
  - The three copies were matched (53 / 18 / 20 inliers) with 0 false
    positives.
  - The *other* US penny (#11), which is not a copy, is the case most like
    real coins. With the first settings it was recognised in 19 % of frames.
    With RootSIFT + ratio 0.85 it reaches **81 %**, and the overlay shows
    `penny 80%`. No coin ever got a wrong type; wrong types only appear if
    the inlier threshold is lowered to 3.
  - Speed: about 110–150 ms per frame (**7–9 fps**) on this Mac.
* Opening the real camera from this development session was refused by macOS
  (no permission), so the live run with your camera is still to be done.

## Outputs (`output/`)

| File | Stage | Content |
|---|---|---|
| `00_roi_overview.png` | H1 | both rectangles and every reference coin found, with names |
| `01_search_region.png` | H1 | search rectangle |
| `02_reference_<k>_<name>.png` | H1 | each reference coin (its outline box) |
| `06_grayscale.png`, `07_gaussian.png`, `08_canny_edges.png` | H2 | preprocessing |
| `10_preprocessing_comparison.png` | H2 | color / gray / smoothed / edges |
| `09_hough_circles.png` | H3 | numbered circle candidates |
| `11_size_filter.png` | H4a | diameter in px per candidate; green = kept (with compatible types), red = discarded |
| `11b_color_filter.png` | H4a' | measured metal per candidate (white = silver, orange = copper, yellow = gold); thin circle and `X` = discarded |
| `12_reference_keypoints.png` | H4 | SIFT keypoints of the references (stretch corrected) |
| `13_candidate_keypoints.png` | H4 | SIFT keypoints of every kept candidate |
| `14_sift_self_check.png` | H5 | control: first reference vs a rotated copy of itself |
| `15_matches/candidate_XX_vs_<type>.png` | H5 | RANSAC inlier matches of each candidate's best comparison |
| `16_final_classification.png` | H6 | label, evidence (`SIFT` or `size+color`), SIFT match %, counts banner |

## How it works

1. **H1 References.** In the reference rectangle, pixels darker than the
   white background are grouped into blobs (threshold, closing, contours).
   Each blob is one reference coin with its own mask, sorted in reading order.
2. **H3 Hough** gives each search coin's centre and radius (17 candidates).
3. **H4a Diameter pre-filter (pixels).** The reference coins are drawn as
   ellipses (e.g. 205 × 250 px), and the image cannot tell which axis is the
   true diameter. So a candidate fits a type if its diameter lies between
   that reference's two axes, widened by ±10 %. Both rectangles are assumed
   to use the same zoom (`REFERENCE_TO_SEARCH_SCALE = 1`).
4. **H4a' Colour pre-filter.** The median CIE Lab colour is measured in two
   zones: centre (< 0.45 r) and ring (0.60–0.85 r). Each zone gets a metal:
   - chroma √(a\*² + b\*²) < 18 → **silver**
   - otherwise hue atan2(b\*, a\*) < 79° → **copper**
   - otherwise → **gold**

   References are measured with the same rule (here: penny → copper; nickel,
   dime, quarter → silver). A candidate keeps only the types whose centre and
   ring metals match its own.
5. **H4 SIFT.** Each coin is resampled to a centred 300 px circle. For the
   references, this also undoes their ~20 % vertical stretch. CLAHE equalises
   local contrast, a circular mask keeps the background out, and
   `SIFT::detectAndCompute` extracts 128-value descriptors. SIFT runs only on
   candidates that survive both pre-filters.
6. **H5 Matching.** `BFMatcher(NORM_L2)`, Lowe's ratio test (0.8) and RANSAC
   (`estimateAffinePartial2D`) with plausibility checks (scale 0.8–1.25,
   centre within 18 px). **Match % = inliers / keypoints of the smaller set.**
   Decision:
   1. SIFT accepts a type (≥ 10 inliers) → that type, evidence `SIFT`.
   2. Two types accepted about equally → `ambiguous`.
   3. Otherwise, if exactly one type is left and no other reference has the
      same colour → that type, evidence `size+color`.
   4. Otherwise → `unknown`.
7. **H6** counts each candidate once, draws the final image, and scores the
   result against `assets/ground_truth.csv`, an annotation made by eye that is
   used only for scoring. It reports overall and per-type precision/recall,
   with detection errors (H3) kept separate from classification errors (H5).

## Results on the supplied image

Ground truth (annotated by eye, scoring only): **2 pennies (#10, #11)** and
**1 quarter (#8, back side)**; no nickels or dimes.

| | Value |
|---|---|
| Reference coins found | 4: penny, nickel, dime, quarter (reading order) |
| Diameter pre-filter | 0 discarded; 45 of 68 comparisons left (34 % avoided) |
| + colour pre-filter | 2 discarded (gold); **24 of 68 comparisons left (65 % avoided)** |
| SIFT self-check (reference vs itself, rotated) | 728 inliers, 71.2 % match |
| SIFT on photo coins | at most 3 inliers: no SIFT acceptance |
| Recognised | 7 pennies (#10 ✔, #11 ✔; #13, #14, #15, #16, #17 ✘), all by `size+color` |
| **Precision / recall** | **28.6 % (2/7) / 66.7 % (2/3)** |
| Penny precision / recall | 28.6 % / 100 % |
| Quarter recall | 0 % (back side visible) |

* **False positives.** The five are copper coins of similar size: copper is
  the penny reference's colour, and the pixel diameter range is wide. Without
  outside knowledge of real coin sizes, size and colour cannot separate them;
  that would need the design, which is SIFT's job.
* **SIFT.** Its own controls work: a reference against a rotated copy of
  itself gives 728 inliers, and two photo coins with the same design (#10,
  #11) give 13. Between the stock references and the photo coins, though, it
  never gets past 6 inliers. Lighting, wear, sharpness and contrast differ.
  Many variants were tried; see the plan, H5.
* **Sensitivity.** The thresholds come from gaps in the values measured on
  this photo: chroma 13 → 24, hue 75° → 83°. Two alternatives:
  - Size tolerance ±5 % instead of ±10 %: #15 leaves the penny range, but
    the real quarter #8 (306 px) also falls below the quarter's range
    (309–404 px).
  - Hue threshold 73.5° instead of 79°: #13 becomes gold, giving 33 %
    precision (2/6).

## Code layout

```text
include/params.hpp       all settings (rectangles, names, thresholds)
src/main.cpp             coin_detector: command line + pipeline H0 → H6
src/pipeline.cpp         one function per milestone (prints, saves images)
src/image_io.cpp         load / save / show / mosaic helpers
src/regions.cpp          H1 ROI cropping + automatic reference-coin detection
src/preprocessing.cpp    H2 grayscale, Gaussian, Canny
src/circle_detection.cpp H3 HoughCircles, duplicate check
src/size_filter.cpp      H4a diameter pre-filter (pixels)
src/color_filter.cpp     H4a' centre/ring metal colour (Lab) pre-filter
src/coin_patch.cpp       H4 normalised coin patches (stretch fix, masks)
src/sift_features.cpp    H4 CLAHE + SIFT
src/coin_matching.cpp    H5 ratio test, RANSAC, decision rules
src/report.cpp           H6 counts, final image, ground-truth scoring
src/paper_view.cpp       H7 find the printed rectangles, top-down view (homography)
src/live_recognizer.cpp  H7 Hough in both rectangles, cached references, SIFT, voting
src/camera_main.cpp      H7 camera loop, sliders, keys, FPS
assets/ground_truth.csv  manual annotation (scoring only)
```

## Milestones

| Milestone | Status |
|---|---|
| H0–H3 | Verified (the `--show` window is not yet confirmed on screen) |
| H4 references, pre-filters, SIFT keypoints/descriptors | Verified |
| H5 Matching and classification | Implemented and verified. SIFT finds no true match on this image; pennies are recognised by size + colour |
| H6 Counting and evaluation | Verified: counts, final image, overall and per-type precision/recall |
| H7 Live camera mode | Implemented; verified on simulated camera frames and video. **Not yet run with a real camera** |

## Known limitations

* The rectangles, the zoom factor and the colour thresholds are set for this
  image; another image needs them checked.
* With only image information, size + colour cannot tell a reference coin
  from a foreign coin of similar size and metal (5 false positives here).
* Hough misses about 9 partial or hidden coins; checked by eye, none of them
  is of a reference type.
* To make SIFT useful, the references would need to be photographed under the
  same conditions as the search photo, and both sides of each coin would need
  to be shown. The live mode does exactly that: real reference coins under the
  same light and camera.
* Live mode: coins showing their other side can't match. Low-texture
  (smooth, worn) coins give few SIFT keypoints, and a different coin of the
  same design reached only ~5 inliers in the simulation. Better resolution
  (a closer camera, 1080p) gives SIFT more detail.
