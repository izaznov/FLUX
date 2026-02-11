import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageColor, ImageFilter, ImageOps

MAX_IMAGE_SIZE = 1600
ART_LIBRARY_DIR = Path("art_library")
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class ArtItem:
    path: str
    title: str
    tags: Tuple[str, ...]


WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def title_from_filename(image_path: Path) -> str:
    return image_path.stem.replace("_", " ").replace("-", " ").title()


def tags_from_filename(image_path: Path) -> Tuple[str, ...]:
    tokens = [token.lower() for token in WORD_RE.findall(image_path.stem)]
    return tuple(sorted(set(tokens)))


def build_art_library(folder: Path) -> List[ArtItem]:
    if not folder.exists():
        return []

    items: List[ArtItem] = []
    for image_path in sorted(folder.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            items.append(
                ArtItem(
                    path=str(image_path),
                    title=title_from_filename(image_path),
                    tags=tags_from_filename(image_path),
                )
            )

    return items


ART_LIBRARY = build_art_library(ART_LIBRARY_DIR)


def load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_stats(image: Image.Image) -> Dict[str, np.ndarray]:
    arr = np.asarray(image.resize((128, 128)), dtype=np.float32) / 255.0
    mean_rgb = arr.reshape(-1, 3).mean(axis=0)

    # simple saturation/brightness proxies
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = np.where(maxc == 0, 0, (maxc - minc) / maxc).mean()
    brightness = maxc.mean()

    return {
        "mean_rgb": mean_rgb,
        "saturation": np.array([saturation], dtype=np.float32),
        "brightness": np.array([brightness], dtype=np.float32),
    }


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def color_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def choose_frame_palette(wall_stats: Dict[str, np.ndarray]) -> Dict[str, str]:
    sat = float(wall_stats["saturation"][0])
    bright = float(wall_stats["brightness"][0])

    if bright > 0.72 and sat < 0.22:
        return {"frame": "#2f2f2f", "mat": "#f3efe8", "name": "gallery black"}
    if bright < 0.40:
        return {"frame": "#c7a76a", "mat": "#f5f0e6", "name": "warm gold"}
    if sat > 0.40:
        return {"frame": "#faf7f2", "mat": "#ece6de", "name": "soft white"}
    return {"frame": "#7a5332", "mat": "#f3ede4", "name": "classic walnut"}


def score_art(prompt: str, wall_image: Image.Image) -> Tuple[ArtItem, Dict[str, float]]:
    if not ART_LIBRARY:
        raise gr.Error(f"No artworks found in '{ART_LIBRARY_DIR}'. Add image files to continue.")

    wall_stats = image_stats(wall_image)
    prompt_tokens = tokenize(prompt)

    best_item = ART_LIBRARY[0]
    best_score = -1e9
    best_breakdown: Dict[str, float] = {}

    for item in ART_LIBRARY:
        art = load_rgb_image(item.path)
        art_stats = image_stats(art)

        tag_tokens = set(tokenize(item.title) + list(item.tags))
        token_overlap = sum(1 for t in prompt_tokens if t in tag_tokens)
        text_score = token_overlap / max(3, len(set(prompt_tokens)))

        wall_art_dist = color_distance(wall_stats["mean_rgb"], art_stats["mean_rgb"])
        # prefer moderate contrast so piece stands out but still fits
        color_score = max(0.0, 1.0 - abs(wall_art_dist - 0.45))

        sat_diff = abs(float(wall_stats["saturation"][0]) - float(art_stats["saturation"][0]))
        style_score = max(0.0, 1.0 - sat_diff)

        final_score = 0.55 * text_score + 0.30 * color_score + 0.15 * style_score

        if final_score > best_score:
            best_score = final_score
            best_item = item
            best_breakdown = {
                "text": text_score,
                "color": color_score,
                "style": style_score,
                "total": final_score,
            }

    return best_item, best_breakdown


def build_framed_art(art: Image.Image, frame_hex: str, mat_hex: str, target_size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = target_size
    art_fit = ImageOps.contain(art, (int(target_w * 0.78), int(target_h * 0.78)))

    mat_pad = max(14, int(min(target_w, target_h) * 0.07))
    frame_thickness = max(18, int(min(target_w, target_h) * 0.09))

    mat_canvas = Image.new("RGB", (art_fit.width + mat_pad * 2, art_fit.height + mat_pad * 2), ImageColor.getrgb(mat_hex))
    mat_canvas.paste(art_fit, (mat_pad, mat_pad))

    framed = ImageOps.expand(mat_canvas, border=frame_thickness, fill=frame_hex)

    # light bevel to make the frame feel less flat
    overlay = Image.new("RGBA", framed.size, (0, 0, 0, 0))
    ow, oh = framed.size
    highlight = Image.new("RGBA", (ow, oh), (255, 255, 255, 0))
    shadow = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))

    for y in range(oh):
        alpha = int(40 * (1 - y / oh))
        if alpha > 0:
            highlight.paste((255, 255, 255, alpha), (0, y, ow, y + 1))
        sa = int(38 * (y / oh))
        if sa > 0:
            shadow.paste((0, 0, 0, sa), (0, y, ow, y + 1))

    overlay = Image.alpha_composite(overlay, highlight)
    overlay = Image.alpha_composite(overlay, shadow)

    framed_rgba = framed.convert("RGBA")
    framed_rgba = Image.alpha_composite(framed_rgba, overlay)
    return framed_rgba


def place_on_wall(wall: Image.Image, framed_art: Image.Image) -> Image.Image:
    wall = wall.convert("RGB")
    wall_w, wall_h = wall.size

    max_w = int(wall_w * 0.44)
    max_h = int(wall_h * 0.56)
    framed_fit = ImageOps.contain(framed_art, (max_w, max_h))

    x = (wall_w - framed_fit.width) // 2
    y = int(wall_h * 0.33) - framed_fit.height // 2
    y = max(20, min(y, wall_h - framed_fit.height - 20))

    canvas = wall.convert("RGBA")

    shadow = Image.new("RGBA", framed_fit.size, (0, 0, 0, 130))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(6, framed_fit.width // 80)))
    canvas.alpha_composite(shadow, (x + 10, y + 14))
    canvas.alpha_composite(framed_fit, (x, y))

    return canvas.convert("RGB")


def infer(prompt: str, wall_photo: Image.Image):
    if wall_photo is None:
        raise gr.Error("Please upload a photo of your wall.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please describe what you want in the artwork.")

    wall = wall_photo.convert("RGB")
    wall = ImageOps.contain(wall, (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

    selected_item, score = score_art(prompt, wall)
    art = load_rgb_image(selected_item.path)

    wall_stats = image_stats(wall)
    frame_palette = choose_frame_palette(wall_stats)

    framed_art = build_framed_art(
        art,
        frame_hex=frame_palette["frame"],
        mat_hex=frame_palette["mat"],
        target_size=(int(wall.width * 0.42), int(wall.height * 0.52)),
    )

    placed = place_on_wall(wall, framed_art)

    details = (
        f"### Selected artwork: **{selected_item.title}**\n"
        f"- Frame style: **{frame_palette['name']}**\n"
        f"- Match score: **{score['total']:.2f}** (text {score['text']:.2f}, color {score['color']:.2f}, style {score['style']:.2f})\n"
        f"- Source file: `{selected_item.path}`"
    )

    return placed, art, details


EXAMPLES = [
    ["I want a cozy pet-themed picture that feels warm and homely", "art_library/cat_window.webp"],
    ["Give me a tropical wildlife piece with vibrant nature colors", "art_library/bird.webp"],
    ["I need an elegant modern portrait for a stylish interior wall", "art_library/woman2.webp"],
]


css = """
#col-container {max-width: 1100px; margin: 0 auto;}
"""


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
# Smart Wall Art Matcher
Upload a wall photo and describe the mood/subject you want. The app picks the best artwork from its local library,
generates a fitting frame style, and places it on your wall preview.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Describe the artwork you want",
                    placeholder="Example: modern portrait with soft neutral tones",
                )
                wall_photo = gr.Image(label="Upload wall photo", type="pil")
                run_button = gr.Button("Match and Place Artwork", variant="primary")

            with gr.Column(scale=1):
                result = gr.Image(label="Framed artwork on your wall", type="pil")
                selected_art = gr.Image(label="Selected library artwork", type="pil")
                details = gr.Markdown(label="Selection details")

        gr.Examples(
            examples=EXAMPLES,
            inputs=[prompt, wall_photo],
            outputs=[result, selected_art, details],
            fn=infer,
            cache_examples=False,
        )

        run_button.click(
            fn=infer,
            inputs=[prompt, wall_photo],
            outputs=[result, selected_art, details],
        )

demo.launch()

