from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch

from model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    Path(r"C:\Users\Ninad\Desktop\OceanShield-data")
    / "oil_spill_23scenes"
    / "oil_spill_23scenes"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "models"
    / "unet_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
)

TEST_SCENE = "2018_09_26.tif"

PATCH_SIZE = 256
STRIDE = 128

THRESHOLD = 0.50

# Ignore extremely small predicted regions in visualization
MIN_REGION_AREA = 100


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = UNet(
    in_channels=1,
    out_channels=1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# SAR NORMALIZATION FOR MODEL
# ============================================================

def normalize_patch(image):
    """
    Normalize one SAR patch using robust percentile scaling.
    """

    image = image.astype(np.float32)

    valid = np.isfinite(image)

    if not np.any(valid):
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    values = image[valid]

    p1 = np.percentile(values, 1)
    p99 = np.percentile(values, 99)

    if p99 <= p1:
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    normalized = (
        image - p1
    ) / (
        p99 - p1
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    normalized[~valid] = 0

    return normalized.astype(
        np.float32
    )


# ============================================================
# SAR DISPLAY ENHANCEMENT
# ============================================================

def enhance_sar_for_display(image):
    """
    Creates a high-contrast visualization of SAR imagery.

    Uses percentile stretching followed by CLAHE
    to make ocean structures and dark features clearer.
    """

    image = image.astype(np.float32)

    valid = np.isfinite(image)

    if not np.any(valid):
        return np.zeros(
            image.shape,
            dtype=np.uint8
        )

    values = image[valid]

    # Robust contrast limits
    low = np.percentile(values, 2)
    high = np.percentile(values, 98)

    if high <= low:
        high = low + 1

    display = (
        image - low
    ) / (
        high - low
    )

    display = np.clip(
        display,
        0,
        1
    )

    display[~valid] = 0

    display = (
        display * 255
    ).astype(np.uint8)

    # CLAHE improves local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(16, 16)
    )

    display = clahe.apply(display)

    return display


# ============================================================
# CREATE PATCH LOCATIONS
# ============================================================

def create_patch_locations(
    width,
    height,
    patch_size,
    stride
):

    patches = []

    for y in range(
        0,
        height,
        stride
    ):

        for x in range(
            0,
            width,
            stride
        ):

            y2 = min(
                y + patch_size,
                height
            )

            x2 = min(
                x + patch_size,
                width
            )

            y1 = max(
                0,
                y2 - patch_size
            )

            x1 = max(
                0,
                x2 - patch_size
            )

            patches.append(
                (x1, y1, x2, y2)
            )

    return list(
        dict.fromkeys(patches)
    )


# ============================================================
# LOAD TEST IMAGE
# ============================================================

image_path = (
    DATASET_ROOT
    / "test"
    / "images"
    / TEST_SCENE
)

if not image_path.exists():

    raise FileNotFoundError(
        f"Test image not found:\n{image_path}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


with rasterio.open(image_path) as src:

    image = src.read(1).astype(
        np.float32
    )

    height, width = image.shape

    crs = src.crs

    transform = src.transform


print()
print("=" * 60)
print("OCEANSHIELD AI - OIL SPILL DETECTION")
print("=" * 60)

print(
    f"Scene: {TEST_SCENE}"
)

print(
    f"Resolution: {width} x {height}"
)

print(
    f"CRS: {crs}"
)


# ============================================================
# CREATE PATCHES
# ============================================================

patches = create_patch_locations(
    width,
    height,
    PATCH_SIZE,
    STRIDE
)

print(
    f"Total patches: {len(patches)}"
)


# ============================================================
# PROBABILITY ACCUMULATION
# ============================================================

probability_sum = np.zeros(
    (height, width),
    dtype=np.float32
)

prediction_count = np.zeros(
    (height, width),
    dtype=np.float32
)


# ============================================================
# RUN MODEL
# ============================================================

print("Running U-Net inference...")

with torch.no_grad():

    for index, (
        x1,
        y1,
        x2,
        y2
    ) in enumerate(
        patches,
        start=1
    ):

        patch = image[
            y1:y2,
            x1:x2
        ]

        normalized = normalize_patch(
            patch
        )

        patch_height, patch_width = (
            normalized.shape
        )

        # Pad boundary patches
        padded = np.zeros(
            (
                PATCH_SIZE,
                PATCH_SIZE
            ),
            dtype=np.float32
        )

        padded[
            :patch_height,
            :patch_width
        ] = normalized

        tensor = torch.from_numpy(
            padded
        )

        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)

        output = model(tensor)

        probability = torch.sigmoid(
            output
        )

        probability = (
            probability
            .squeeze()
            .cpu()
            .numpy()
        )

        probability = probability[
            :patch_height,
            :patch_width
        ]

        probability_sum[
            y1:y2,
            x1:x2
        ] += probability

        prediction_count[
            y1:y2,
            x1:x2
        ] += 1

        if (
            index % 100 == 0
            or index == len(patches)
        ):

            print(
                f"Processed "
                f"{index}/{len(patches)}"
            )


# ============================================================
# BLEND OVERLAPPING PATCHES
# ============================================================

probability_map = (
    probability_sum
    / np.maximum(
        prediction_count,
        1
    )
)

probability_map = np.clip(
    probability_map,
    0,
    1
)


# ============================================================
# BINARY PREDICTION
# ============================================================

prediction_mask = (
    probability_map >= THRESHOLD
).astype(np.uint8)


# ============================================================
# REMOVE SMALL NOISE
# ============================================================

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    prediction_mask,
    connectivity=8
)

clean_mask = np.zeros_like(
    prediction_mask
)

for label in range(
    1,
    num_labels
):

    area = stats[
        label,
        cv2.CC_STAT_AREA
    ]

    if area >= MIN_REGION_AREA:

        clean_mask[
            labels == label
        ] = 1


prediction_mask = clean_mask


# ============================================================
# STATISTICS
# ============================================================

predicted_pixels = int(
    prediction_mask.sum()
)

total_pixels = prediction_mask.size

predicted_percentage = (
    predicted_pixels
    / total_pixels
) * 100

# Dataset has 10m pixels
pixel_area_m2 = 10 * 10

area_m2 = (
    predicted_pixels
    * pixel_area_m2
)

area_km2 = (
    area_m2 / 1_000_000
)

max_probability = (
    float(probability_map.max())
)

mean_probability = (
    float(
        probability_map[
            prediction_mask == 1
        ].mean()
    )
    if predicted_pixels > 0
    else 0
)


print()
print("=" * 60)
print("DETECTION RESULTS")
print("=" * 60)

print(
    f"Predicted spill pixels : "
    f"{predicted_pixels:,}"
)

print(
    f"Predicted spill        : "
    f"{predicted_percentage:.2f}%"
)

print(
    f"Estimated spill area   : "
    f"{area_km2:.2f} km²"
)

print(
    f"Maximum confidence     : "
    f"{max_probability:.3f}"
)

print(
    f"Mean spill confidence : "
    f"{mean_probability:.3f}"
)


# ============================================================
# 1. ORIGINAL ENHANCED SAR IMAGE
# ============================================================

print()
print("Creating visualization...")


enhanced_sar = (
    enhance_sar_for_display(
        image
    )
)

original_bgr = cv2.cvtColor(
    enhanced_sar,
    cv2.COLOR_GRAY2BGR
)

original_path = (
    OUTPUT_DIR
    / "01_enhanced_sar.png"
)

cv2.imwrite(
    str(original_path),
    original_bgr
)


# ============================================================
# 2. CLEAN BINARY MASK
# ============================================================

mask_image = (
    prediction_mask * 255
).astype(np.uint8)

mask_path = (
    OUTPUT_DIR
    / "02_prediction_mask.png"
)

cv2.imwrite(
    str(mask_path),
    mask_image
)


# ============================================================
# 3. CONFIDENCE HEATMAP
# ============================================================

probability_uint8 = (
    probability_map * 255
).astype(np.uint8)

heatmap = cv2.applyColorMap(
    probability_uint8,
    cv2.COLORMAP_JET
)

heatmap_path = (
    OUTPUT_DIR
    / "03_confidence_heatmap.png"
)

cv2.imwrite(
    str(heatmap_path),
    heatmap
)


# ============================================================
# 4. SIH DEMO OVERLAY
# ============================================================

# Start with enhanced SAR
overlay = original_bgr.copy()

# Darken the entire background slightly
overlay = (
    overlay.astype(np.float32)
    * 0.70
).clip(
    0,
    255
).astype(np.uint8)


# ------------------------------------------------------------
# Create strong red spill layer
# ------------------------------------------------------------

spill_layer = np.zeros_like(
    overlay
)

# BGR = RED
spill_layer[
    prediction_mask == 1
] = (
    0,
    0,
    255
)


# Blend red with SAR
overlay = cv2.addWeighted(
    overlay,
    1.0,
    spill_layer,
    0.65,
    0
)


# ============================================================
# DRAW YELLOW SPILL BOUNDARIES
# ============================================================

contours, _ = cv2.findContours(
    prediction_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

valid_contours = []

for contour in contours:

    area = cv2.contourArea(
        contour
    )

    if area >= MIN_REGION_AREA:

        valid_contours.append(
            contour
        )

# Yellow boundary
cv2.drawContours(
    overlay,
    valid_contours,
    -1,
    (0, 255, 255),
    4
)


# ============================================================
# CREATE HEADER
# ============================================================

header_height = 150

canvas = cv2.copyMakeBorder(
    overlay,
    header_height,
    0,
    0,
    0,
    cv2.BORDER_CONSTANT,
    value=(15, 15, 15)
)


# ============================================================
# HEADER TEXT
# ============================================================

cv2.putText(
    canvas,
    "OCEANSHIELD AI",
    (35, 42),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.15,
    (255, 255, 255),
    3,
    cv2.LINE_AA
)

cv2.putText(
    canvas,
    "AI-BASED SATELLITE OIL SPILL DETECTION",
    (35, 82),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (220, 220, 220),
    2,
    cv2.LINE_AA
)

cv2.putText(
    canvas,
    f"Scene: {TEST_SCENE}",
    (35, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.60,
    (200, 200, 200),
    2,
    cv2.LINE_AA
)


# ============================================================
# RIGHT SIDE STATISTICS
# ============================================================

right_x = max(
    700,
    width - 700
)

cv2.putText(
    canvas,
    f"SPILL: {predicted_percentage:.2f}%",
    (right_x, 42),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (0, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    canvas,
    f"AREA: {area_km2:.2f} km2",
    (right_x, 82),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    canvas,
    f"CONFIDENCE: {mean_probability:.2f}",
    (right_x, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# ============================================================
# LEGEND
# ============================================================

legend_y = canvas.shape[0] - 45

# Red square
cv2.rectangle(
    canvas,
    (30, legend_y),
    (60, legend_y + 25),
    (0, 0, 255),
    -1
)

cv2.putText(
    canvas,
    "AI detected spill",
    (75, legend_y + 21),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

# Yellow square
yellow_x = 270

cv2.rectangle(
    canvas,
    (yellow_x, legend_y),
    (yellow_x + 30, legend_y + 25),
    (0, 255, 255),
    -1
)

cv2.putText(
    canvas,
    "Spill boundary",
    (yellow_x + 45, legend_y + 21),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# ============================================================
# SAVE DEMO IMAGE
# ============================================================

demo_path = (
    OUTPUT_DIR
    / "04_SIH_DEMO_OIL_SPILL.png"
)

cv2.imwrite(
    str(demo_path),
    canvas
)


# ============================================================
# OPTIONAL: SAVE HIGH-CONTRAST SPILL-ONLY IMAGE
# ============================================================

spill_only = np.zeros_like(
    original_bgr
)

spill_only[
    prediction_mask == 1
] = (
    0,
    0,
    255
)

spill_only_path = (
    OUTPUT_DIR
    / "05_spill_highlight.png"
)

cv2.imwrite(
    str(spill_only_path),
    spill_only
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("OUTPUT FILES")
print("=" * 60)

print(
    f"1. Enhanced SAR:\n"
    f"   {original_path}"
)

print(
    f"2. Binary mask:\n"
    f"   {mask_path}"
)

print(
    f"3. Confidence heatmap:\n"
    f"   {heatmap_path}"
)

print(
    f"4. SIH DEMO IMAGE:\n"
    f"   {demo_path}"
)

print(
    f"5. Spill highlight:\n"
    f"   {spill_only_path}"
)

print()
print("Inference complete.")