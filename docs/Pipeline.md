# UAV Road Inspection Preprocessing Pipeline

This document provides a detailed technical and mathematical explanation of the image enhancement, color-space transformation, and resolution normalization pipeline implemented for aerial UAV road inspection.

---

## 1. Contrast Limited Adaptive Histogram Equalization (CLAHE)

### 1.1 Motivation in Aerial UAV Imagery
Aerial images captured by Unmanned Aerial Vehicles (UAVs) present severe radiometric challenges that impair computer vision algorithms:
1. **Dynamic Shadow Obstructions:** Tall roadside trees, utility poles, and adjacent infrastructure cast deep, sharp shadows across road corridors.
2. **Specular Solar Reflections:** High solar angles produce localized glare and overexposed patches on light-colored concrete or fresh asphalt.
3. **Low Contrast Asphalt Boundaries:** Aging road surfaces, wet patches, and weathered bitumen share nearly identical pixel intensity distributions with shallow depressions and hairline cracks.

Standard Global Histogram Equalization (GHE) computes a single transformation for the entire image based on global pixel frequencies. In aerial photography, this approach typically causes catastrophic overexposure in bright road sections and amplifies high-frequency sensor noise across flat asphalt regions. To overcome these limitations, our pipeline integrates **Contrast Limited Adaptive Histogram Equalization (CLAHE)**.

---

### 1.2 Color Space Conversion: CIE LAB Space Transformation
Standard RGB/BGR image representations interleave chromaticity (color) and luminance (perceived brightness) across all three color channels ($R, G, B$). Applying histogram transformations directly to RGB channels causes severe chromatic distortion (color shifting, unnatural tinting of asphalt, and loss of edge fidelity).

To preserve the true photometric properties of the road surface, our pipeline converts the input imagery into the **CIE $L^*a^*b^*$ color space**:
- **$L^*$ Channel (Luminance):** Represents lightness ranging from 0 (pure black) to 100 (diffuse white). This channel models the human visual system's perception of brightness.
- **$a^*$ Channel (Chrominance Green-Red):** Represents color coordinates along the green (negative) to red (positive) axis.
- **$b^*$ Channel (Chrominance Blue-Yellow):** Represents color coordinates along the blue (negative) to yellow (positive) axis.

$$\text{BGR} \xrightarrow{\text{Nonlinear Matrix Conversion}} \text{CIE } L^*a^*b^*$$

**Processing Strategy:**
1. Split the image into individual channels: $L^*$, $a^*$, and $b^*$.
2. Apply CLAHE **strictly to the $L^*$ channel**, enhancing surface details, depth cues inside potholes, and crack edges.
3. Keep the $a^*$ and $b^*$ chrominance channels entirely unchanged to ensure 100% natural road pigmentation.
4. Merge the enhanced $L^*$ channel with the original $a^*$ and $b^*$ channels.
5. Convert back to BGR space for deep learning ingestion:

$$\text{CIE } [L^*_{\text{enhanced}}, a^*, b^*] \xrightarrow{} \text{BGR}_{\text{enhanced}}$$

---

### 1.3 Mathematical Formulation & Parameter Configuration

CLAHE divides the image into small non-overlapping contextual regions called **tiles** or **grids** (default: $8 \times 8$). Within each tile, the local histogram is computed and clipped to prevent noise amplification.

#### 1. Clip Limit ($\beta = 2.0$)
The clip limit sets an upper bound on the height of the local histogram bins. Any histogram values exceeding this threshold are clipped and redistributed uniformly across all gray-level bins before computing the Cumulative Distribution Function (CDF):

$$H_{\text{clipped}}(k) = \begin{cases} \beta_{\text{actual}}, & \text{if } H(k) > \beta_{\text{actual}} \\ H(k), & \text{otherwise} \end{cases}$$

$$\Delta = \frac{1}{N} \sum_{k=0}^{N-1} \max(0, H(k) - \beta_{\text{actual}})$$

$$H_{\text{final}}(k) = H_{\text{clipped}}(k) + \Delta$$

* **Selected Value (`clipLimit = 2.0`):** Prevents over-amplification of sensor noise and pavement aggregate grain, while providing sufficient dynamic range expansion inside shadowed potholes.

#### 2. Tile Grid Size ($M \times N = 8 \times 8$)
* The image is partitioned into $8 \times 8 = 64$ sub-regions.
* Histogram equalization is evaluated locally within each tile.
* **Bilinear Interpolation across Tile Boundaries:** To eliminate artificial grid boundaries (blocking artifacts), adjacent tile mappings are blended using bilinear interpolation:

$$T(x, y) = (1-s)(1-t) T_1 + s(1-t) T_2 + (1-s)t T_3 + st T_4$$

where $s$ and $t$ are normalized intra-tile coordinates, and $T_1 \dots T_4$ are the transformation functions of the four surrounding contextual tiles.

---

## 2. Image Resizing and Resolution Normalization

### 2.1 Motivation for 640x640 Normalization
UAV camera payloads commonly acquire imagery at **4K Ultra-High Definition ($3840 \times 2160$ pixels)**. Directly feeding 4K imagery into deep convolutional or transformer-based networks introduces severe computational bottlenecks:
- Exponential increase in GPU VRAM consumption ($O(H \times W)$ memory scaling).
- Extreme reduction in batch size, leading to unstable batch normalization gradients.
- Severe latency penalties incompatible with real-time UAV flight monitoring.

Downscaling to **$640 \times 640$ pixels** provides an optimal trade-off between structural defect visibility, receptive field coverage, and inference throughput ($>30$ FPS).

---

### 2.2 Interpolation Method: Bilinear Interpolation (`cv2.INTER_LINEAR`)
When downsampling from $3840 \times 2160$ to $640 \times 640$, the pipeline utilizes **bilinear interpolation**:
- **Why not Nearest Neighbor (`INTER_NEAREST`)?** Produces jagged, aliased edges that distort thin crack boundaries.
- **Why Bilinear over Bicubic (`INTER_CUBIC`)?** Bicubic computes a $4 \times 4$ neighborhood with cubic splines, which is computationally heavier and prone to overshoot artifacts (ringing) along high-contrast shadow boundaries. Bilinear interpolation computes a weighted average of the $2 \times 2$ nearest pixels, preserving continuous asphalt gradients at maximum downsampling speed.

---

### 2.3 YOLO Coordinate Invariance under Resizing
A major advantage of the YOLO annotation format is that bounding boxes are represented as **normalized relative coordinates**:

$$\text{Annotation Format: } [c, x_{\text{center}}, y_{\text{center}}, w, h]$$

Where:
- $c \in \{0, 1\}$ is the class index (`0: agujero`, `1: grietas`).
- $x_{\text{center}} = \frac{X_{\text{pixel}}}{W_{\text{original}}} \in [0, 1]$
- $y_{\text{center}} = \frac{Y_{\text{pixel}}}{H_{\text{original}}} \in [0, 1]$
- $w = \frac{\text{Width}_{\text{pixel}}}{W_{\text{original}}} \in [0, 1]$
- $h = \frac{\text{Height}_{\text{pixel}}}{H_{\text{original}}} \in [0, 1]$

**Mathematical Proof of Coordinate Invariance:**
When the image is scaled from original dimensions $(W_{\text{orig}}, H_{\text{orig}})$ to new dimensions $(W_{\text{new}}, H_{\text{new}})$, the absolute pixel coordinates scale linearly:

$$X'_{\text{pixel}} = X_{\text{pixel}} \times \left(\frac{W_{\text{new}}}{W_{\text{orig}}}\right), \quad Y'_{\text{pixel}} = Y_{\text{pixel}} \times \left(\frac{H_{\text{new}}}{H_{\text{orig}}}\right)$$

Re-normalizing relative to the new dimensions yields:

$$x'_{\text{center}} = \frac{X'_{\text{pixel}}}{W_{\text{new}}} = \frac{X_{\text{pixel}} \cdot (W_{\text{new}} / W_{\text{orig}})}{W_{\text{new}}} = \frac{X_{\text{pixel}}}{W_{\text{orig}}} = x_{\text{center}}$$

$$\therefore \quad [x'_{\text{center}}, y'_{\text{center}}, w', h'] \equiv [x_{\text{center}}, y_{\text{center}}, w, h]$$

**Engineering Consequence:** Annotation text files do not require any mathematical alteration during resizing. The existing `.txt` label files can be copied directly to the resized output directory without loss of ground-truth spatial accuracy.

---

## 3. Preprocessing Execution Guide

The preprocessing pipeline is completely modular and can be run via CLI commands with custom arguments:

### Step 1: Run CLAHE Contrast Enhancement
```bash
# Default execution (clip limit = 2.0, tile grid = 8x8)
python preprocessing/clahe.py

# Custom parameters for high-glare or deep shadow conditions
python preprocessing/clahe.py --clip-limit 3.0 --grid-size 16 --output-dir preprocessing/output_enhanced
```

### Step 2: Run Resolution Normalization
```bash
# Resize CLAHE-preprocessed images to 640x640
python preprocessing/resize.py --input-dir preprocessing/output --output-dir preprocessing/output_resized --width 640 --height 640
```
