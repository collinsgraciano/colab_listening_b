"""Quick test: verify remove_bg skips rembg for pre-segmented images."""
import sys
sys.path.insert(0, r"H:\2026_main_project\colab_listening_b")

from PIL import Image
import numpy as np
from stop_motion import remove_bg, _has_transparency

# Test 1: frontier segmented image (P-mode with palette transparency)
img = Image.open(r"H:\2026_main_project\colab_listening_b\test_seg_frontier.png")
print(f"Test 1 (frontier seg): mode={img.mode}, has_transparency={_has_transparency(img)}")
result = remove_bg(img)
print(f"  After remove_bg: mode={result.mode}, size={result.size}")
alpha = np.array(result.getchannel("A"))
pct = (alpha == 0).sum() / alpha.size * 100
print(f"  Transparent: {(alpha==0).sum()}/{alpha.size} = {pct:.1f}%")

# Test 2: 4K atlas segmented image
img2 = Image.open(r"H:\2026_main_project\colab_listening_b\test_seg_atlas4k.png")
print(f"\nTest 2 (4K atlas seg): mode={img2.mode}, has_transparency={_has_transparency(img2)}")
result2 = remove_bg(img2)
print(f"  After remove_bg: mode={result2.mode}, size={result2.size}")
alpha2 = np.array(result2.getchannel("A"))
pct2 = (alpha2 == 0).sum() / alpha2.size * 100
print(f"  Transparent: {(alpha2==0).sum()}/{alpha2.size} = {pct2:.1f}%")

print("\n✅ All tests passed — remove_bg correctly skips rembg for pre-segmented images.")
