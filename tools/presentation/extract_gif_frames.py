"""Extract real frames from the retained simulated-grasp GIF for the portfolio deck."""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/20260816_140448_414159_run_seed10/demo.gif"
DESTINATION = ROOT / "deliverables/portfolio/assets"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with Image.open(SOURCE) as image:
        frame_count = image.n_frames
        for label, index in (("approach", max(0, frame_count // 5)), ("lift", max(0, (frame_count * 4) // 5))):
            image.seek(index)
            image.convert("RGB").save(DESTINATION / f"seed10_{label}.png")
    print(f"Extracted two frames from {SOURCE.relative_to(ROOT)} ({frame_count} total frames).")


if __name__ == "__main__":
    main()
