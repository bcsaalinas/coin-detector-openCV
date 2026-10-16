# Coin detector — OpenCV / C++

Undergraduate Computer Vision project. Final goal: count how many coins of each
reference type (**penny, nickel, dime, quarter**) appear in a photo of mixed
international coins, using methods from the course (Canny, Hough, SIFT).

**Current state: milestones H0–H3 (50 % of the plan).** The program finds
*geometric coin candidates* (circles). It does **not** recognise or count
denominations yet. That work (SIFT matching, classification, final counting)
is H4–H6 in [`PLAN_PROYECTO_CONTEO_MONEDAS.md`](PLAN_PROYECTO_CONTEO_MONEDAS.md),
which also holds the progress log.

Pipeline:

```text
image ─► H1 ROI crop ─► H2 grayscale ─► Gaussian blur ─► Canny ─► H3 HoughCircles ─► candidates
            └─► 4 reference coins (kept for H4 / SIFT)
```

## Requirements

| Tool | Tested version | Install (macOS / Homebrew) |
|---|---|---|
| C++17 compiler | Apple clang 21.0 | `xcode-select --install` |
| CMake ≥ 3.21 | 4.4.2 | `brew install cmake` |
| OpenCV (core, imgproc, imgcodecs, highgui) | 5.0.0 | `brew install opencv` |

OpenCV 4.x also provides every function used so far. On Debian/Ubuntu,
`sudo apt install build-essential cmake libopencv-dev` gives the same setup.
Ninja is not required; the presets use CMake's default generator.

## Build

```bash
cmake --preset release
cmake --build --preset release
```

For a debug build, use `debug` instead of `release` (binary in `build/debug/`).
Without presets:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

## Run

```bash
./build/release/coin_detector
```

| Option | Meaning |
|---|---|
| `--image <path>` | input image (default `assets/coins.png`) |
| `--output <dir>` | folder for generated images (default `output/`, created if missing) |
| `--show` | also open a window per stage; press a key to continue |

Default paths are absolute paths to this project folder, set by CMake at
configure time, so the program works from any working directory. If you move
the project folder, configure again.

The input must be the provided **2477 × 1929** image: the ROI rectangles were
measured on that resolution, and the program stops with an explanation if
the size differs.

## Outputs (`output/`)

| File | Milestone | Content |
|---|---|---|
| `00_roi_overview.png` | H1 | original image with all ROI rectangles drawn |
| `01_search_region.png` | H1 | left search region (1064 × 1595) |
| `02…05_reference_<coin>.png` | H1 | penny, nickel, dime, quarter crops |
| `06_grayscale.png` | H2 | search region in grayscale |
| `07_gaussian.png` | H2 | after Gaussian smoothing |
| `08_canny_edges.png` | H2 | Canny edge map (same edges HoughCircles uses internally) |
| `09_hough_circles.png` | H3 | numbered circle candidates on the color search region |
| `10_preprocessing_comparison.png` | H2 | color / gray / smoothed / edges side by side |

The console prints image sizes, parameters and a table of candidates (id,
center in search-region and full-image coordinates, radius, cut by border).

## Parameters

All tunable values are in [`include/params.hpp`](include/params.hpp), with
comments explaining each value and what was tried.

| Stage | Value |
|---|---|
| Gaussian | kernel 9 × 9, σ = 2.0 |
| Canny | low 60, high 120 |
| HoughCircles | `HOUGH_GRADIENT`, dp 1.2, minDist 100, param1 120, param2 40, radius 95–220 |

## Code layout

```text
include/params.hpp           all constants (ROIs, thresholds)
src/main.cpp                 pipeline H0 → H3, command-line options
src/image_io.cpp             load / save / show helpers
src/regions.cpp              H1: ROI validation and cropping
src/preprocessing.cpp        H2: grayscale, Gaussian, Canny
src/circle_detection.cpp     H3: HoughCircles, duplicate check, drawing
```

## Milestones

| Milestone | Status |
|---|---|
| H0 Setup and image loading | Verified (the `--show` window is not yet confirmed on screen) |
| H1 ROI extraction | Verified |
| H2 Preprocessing and Canny | Verified |
| H3 Hough circle candidates | Verified: 17 candidates, reviewed by hand |
| H4 SIFT keypoints/descriptors | Not started |
| H5 Matching and classification | Not started |
| H6 Counting and final report | Not started |

## Known limitations (H3)

* **17 circles ≠ 17 US coins.** Most coins in the search region are not US
  coins; classification only starts in H4.
* Reviewed by hand: all 13 coins that are whole or almost whole (ids 1–12
  and 14) get a good circle, and so do 3 partial coins at the borders (13, 16,
  17). There are no false positives on the wood and no duplicates.
* About 9 partially visible coins are missed: coins cut by the right or bottom
  border, a silver coin mostly hidden under two others, and one of two stacked
  gold coins on the right.
* Candidate 15 follows the visible rim of a gold coin whose lower half is
  covered. It is plausible but cannot be confirmed by eye.
* Candidate 3 fits the inner beaded rim of its coin, so its radius is about
  12–15 px too small.
* Lowering `param2` to 35 finds 18 circles; 32 or less starts detecting wood
  grain. The chosen 40 keeps a margin.
* The reference coins are vertically stretched (ellipses about 205 × 250 px),
  but the coins in the photo are round. SIFT is not affine-invariant, so this
  may weaken H4 matching.

## Next step (H4)

Add the OpenCV features module (`features` in OpenCV 5, `features2d` in 4)
to `find_package`. Then, for each reference and each non-border candidate,
crop the coin with a circular or elliptical mask and run
`cv::SIFT::create()->detectAndCompute()`. Save keypoint images and check that
no descriptor set is empty.
