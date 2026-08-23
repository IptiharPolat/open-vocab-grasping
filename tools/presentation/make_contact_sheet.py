"""Build a labeled contact sheet from locally rendered portfolio slides."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RENDERED = ROOT / "deliverables/portfolio/rendered/final"
OUT = ROOT / "deliverables/portfolio/contact_sheet.png"


def main() -> None:
    files = sorted(RENDERED.glob("slide-*.png"))
    if len(files) != 8:
        raise RuntimeError(f"Expected 8 rendered slides, found {len(files)} in {RENDERED}")
    thumb_w, thumb_h, gap, label_h = 480, 270, 24, 38
    canvas = Image.new("RGB", (thumb_w * 2 + gap * 3, (thumb_h + label_h) * 4 + gap * 5), "#071522")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)
    for index, file in enumerate(files):
        image = Image.open(file).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        col, row = index % 2, index // 2
        x = gap + col * (thumb_w + gap)
        y = gap + row * (thumb_h + label_h + gap)
        canvas.paste(image, (x, y))
        draw.text((x, y + thumb_h + 7), f"{index + 1:02d}  {file.name}", fill="#DDEAF0", font=font)
    canvas.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
