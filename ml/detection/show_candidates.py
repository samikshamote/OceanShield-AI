import json

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "candidate_assessment.json"
)


with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


print()
print("TOP 10 CANDIDATE DETAILS")
print("=" * 100)

for rank, region in enumerate(
    data["top_candidates"],
    start=1
):

    print(
        f"#{rank} | "
        f"ID={region['region_id']} | "
        f"Area={region['area_pixels']:,} | "
        f"Mean={region['mean_confidence']:.3f} | "
        f"Max={region['maximum_confidence']:.3f} | "
        f"High80={region['high_confidence_pixels_percent']:.2f}% | "
        f"Size={region['width_pixels']}x{region['height_pixels']} | "
        f"Border={region['touches_image_border']} | "
        f"Score={region['priority_score']:.2f} | "
        f"{region['classification']}"
    )

print("=" * 100)