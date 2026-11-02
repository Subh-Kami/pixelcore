"""Render only recorded model-output pixels as a labeled, accelerated replay clip."""
import argparse
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from qgpu.pixelcore.runtime import verify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trace", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--font", type=Path, help="Optional proportional TrueType font")
parser.add_argument("--mono-font", type=Path, help="Optional monospace TrueType font")
args = parser.parse_args()
trace = json.loads(args.trace.read_text())
result = verify(trace)
if trace["meta"]["kind"] != "merged":
    raise SystemExit("This Qwen-branded clip requires an actual merged-model recording")
def font(size, code=False):
    explicit = args.mono_font if code else args.font
    candidates = ([explicit] if explicit else []) + ([
        Path("/System/Library/Fonts/Menlo.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
    ] if code else [
        Path("/System/Library/Fonts/SFNS.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ])
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)
palette = ["#0c140f", "#4b5d43", "#ffb47b", "#c5f888"]
fps, acceleration = 30, 4
args.output.parent.mkdir(parents=True, exist_ok=True)
process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1080", "-r", str(fps),
    "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
    "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)],
    stdin=subprocess.PIPE)
frames = 0
try:
    for index, record in enumerate(trace["records"]):
        image = Image.new("RGB", (1080, 1080), "#0c100e")
        draw = ImageDraw.Draw(image)
        draw.text((64, 45), "PIXELCORE / WEIGHTPONG", font=font(20, True), fill="#c5f888")
        draw.text((64, 96), "The model is", font=font(65), fill="#e9eee6")
        draw.text((64, 166), "the game engine.", font=font(65), fill="#c5f888")
        draw.text((64, 256), "Qwen3.5-2B + 8 compiled game layers", font=font(25), fill="#a5b19f")
        draw.text((778, 45), "VERIFIED MODEL RUN", font=font(17, True), fill="#ffb47b")
        draw.text((778, 75), "REPLAY / 4x SPEED", font=font(17, True), fill="#ffb47b")
        for cell, value in enumerate(record["pixels"]):
            x, y = 64 + cell % 8 * 76, 310 + cell // 8 * 76
            draw.rectangle((x + 2, y + 2, x + 72, y + 72), fill=palette[value])
            if value == 2:
                draw.rectangle((x + 12, y + 12, x + 27, y + 27), fill="#ffe8ce")
        packed = record["after"]
        score = packed >> 11 & 3
        draw.text((720, 317), "SAVES", font=font(18, True), fill="#8d998d")
        draw.text((716, 349), f"{score} / 3", font=font(65), fill="#c5f888")
        draw.text((720, 465), "18 TOKENS / TICK", font=font(21, True), fill="#e9eee6")
        draw.text((720, 503), "14 BITS OF STATE", font=font(21, True), fill="#8d998d")
        draw.text((720, 541), "8 x 8 PIXELS", font=font(21, True), fill="#8d998d")
        draw.text((720, 628), f"{result['tokens']} / {result['tokens']} TOKENS", font=font(20, True), fill="#c5f888")
        draw.text((720, 662), "MATCHED REFERENCE", font=font(18, True), fill="#8d998d")
        draw.text((720, 738), f"TICK {max(0,index):02d} / {result['ticks']}", font=font(21, True), fill="#e9eee6")
        draw.text((720, 778), "BUTTON: " + (record["action"] or "BOOT").upper(), font=font(19, True), fill="#8d998d")
        draw.text((720, 839), f"{record['seconds']:.2f}s compute", font=font(22), fill="#ffb47b")
        draw.text((64, 952), "State, physics and pixels come from model weights.", font=font(26), fill="#e9eee6")
        draw.text((64, 997), "Host: external paddle assist + independent output checker.", font=font(19), fill="#8d998d")
        draw.text((64, 1035), "Compiler: Qwenputer   /   MPS   /   4x recording, not live speed", font=font(17, True), fill="#8d998d")
        count = max(1, math.ceil(record["seconds"] * fps / acceleration))
        if index == len(trace["records"]) - 1:
            count += fps
        raw = image.tobytes()
        for _ in range(count):
            process.stdin.write(raw)
        frames += count
    process.stdin.close()
    if process.wait():
        raise SystemExit("ffmpeg failed")
except BaseException:
    process.kill()
    raise
print(f"{args.output}: {frames/fps:.2f}s, 1080x1080, 4x replay")
