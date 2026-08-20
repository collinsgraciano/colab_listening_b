from PIL import Image
for name in ['frame_welcome', 'frame_hook', 'frame_dialogue', 'frame_outro']:
    img = Image.open(f'test5/{name}.png').convert('RGBA')
    w, h = img.size
    sub = img.crop((0, h-150, w, h))
    white = 0
    for y in range(sub.height):
        for x in range(0, sub.width, 2):
            p = sub.getpixel((x, y))
            if p[0] > 200 and p[1] > 200 and p[2] > 200:
                white += 1
    tag = "HAS SUBTITLE" if white > 50 else "NO SUBTITLE"
    print(f"{name}: {w}x{h} white={white} {tag}")
