#!/usr/bin/env python3
"""Quick test of stop_motion renderer using existing test image."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from PIL import Image
from stop_motion import (
    remove_bg, normalize_pose, render_frame, compute_landing,
    generate_morph_frames, quantize, MOTION_FPS, TARGET_W, TARGET_H,
    _HAS_REMBG,
)

TEST_DIR = Path(__file__).parent
IMG_PATH = str(TEST_DIR / "dialogue_test.png")
BG_PATH = str(TEST_DIR / "dialogue_test.png")  # reuse as bg

def main():
    if not os.path.exists(IMG_PATH):
        print("ERROR: dialogue_test.png not found")
        sys.exit(1)

    print(f"1. Removing background (rembg: {_HAS_REMBG})...")
    raw = Image.open(IMG_PATH)
    print(f"   Raw size: {raw.size}")
    alpha = remove_bg(raw)
    # Count transparent pixels
    pixels = list(alpha.getdata())
    transparent = sum(1 for p in pixels if p[3] == 0)
    total = len(pixels)
    print(f"   Transparent: {transparent}/{total} ({100*transparent/total:.1f}%)")

    print("2. Normalizing pose...")
    normalized = normalize_pose(alpha)
    print(f"   Normalized size: {normalized.size}")

    print("3. Testing landing transform at various times...")
    for t in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        landing = compute_landing(t, direction=1)
        print(f"   t={t:.1f}s: scale={landing['scale']:.4f} x={landing['x']:.1f} y={landing['y']:.1f} rot={landing['rotation']:.2f}")

    print("4. Testing quantize (motion_fps=8)...")
    for t in [0.0, 0.05, 0.1, 0.125, 0.13, 0.25, 0.375, 0.5]:
        qt = quantize(t)
        print(f"   t={t:.3f} -> quantized={qt:.3f}")

    print("5. Rendering test frame...")
    bg = Image.open(BG_PATH).convert("RGBA").resize((TARGET_W, TARGET_H))
    landing = compute_landing(0.1, direction=1)
    scale = 1.0 + landing["scale"]
    frame = render_frame(bg, normalized, TARGET_W/2 + landing["x"], 
                         700 + landing["y"], scale, landing["rotation"])
    frame.save(str(TEST_DIR / "stop_motion_test_frame.png"))
    print(f"   Saved: stop_motion_test_frame.png ({frame.size})")

    print("6. Testing morph (if cv2 available)...")
    # Create a second "pose" by flipping the first
    pose_b = normalized.transpose(Image.FLIP_LEFT_RIGHT)
    morph_frames = generate_morph_frames(normalized, pose_b, n_frames=5)
    print(f"   Generated {len(morph_frames)} morph frames")
    for i, f in enumerate(morph_frames):
        f.save(str(TEST_DIR / f"morph_test_{i}.png"))
    print(f"   Saved morph frames: morph_test_0..4.png")

    print("\nAll tests passed!")

if __name__ == "__main__":
    main()
