"""Make windows/monitor/kindi.ico: the Kindi mark for the tray, the windows, the exe and the installer.

A lime K (#CAF31C, the Kindi page's accent) on an ink tile (#080808, its background), drawn for each size on its
own pixel grid (16, 20, 24, 32, 48, 64, 256) so that the stem and arms stay sharp at 16 px. The tile keeps the K
readable on light and dark taskbars. Also writes a preview (--preview FILE): every size on a light and a dark strip.

    python windows/monitor/make_icon.py [--preview preview.png]

Needs Pillow. Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.
"""
import os
import sys

from PIL import Image, ImageDraw

INK = (8, 8, 8, 255)
LIME = (202, 243, 28, 255)
EDGE = (58, 58, 58, 255)
SIZES = (16, 20, 24, 32, 48, 64, 256)
SS = 8   # supersampling


def mark(n):
    big = n * SS
    im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = max(1, round(n * 0.18)) * SS
    d.rounded_rectangle((0, 0, big - 1, big - 1), radius=r, fill=INK)
    if n >= 32:   # a hairline edge, so the tile shows on a black taskbar too
        d.rounded_rectangle((0, 0, big - 1, big - 1), radius=r, outline=EDGE, width=max(1, n // 32) * SS)
    # the K on the pixel grid of this size: a stem and two arms of the same weight
    w = max(2, round(n * 0.19))                     # stroke weight in pixels (3 at 16 px)
    top, bot = round(n * 0.19), n - round(n * 0.19)
    sx = round(n * 0.25)
    d.rectangle((sx * SS, top * SS, (sx + w) * SS - 1, bot * SS - 1), fill=LIME)
    # the arms: two strokes of the same weight, drawn long and then cut to the K's box (top..bot, stem..rx)
    rx = n - round(n * 0.22)
    arms = Image.new("L", (big, big), 0)
    a = ImageDraw.Draw(arms)
    mid = (top + bot) / 2.0
    jx = sx + w * 0.6                               # the arms start inside the stem
    a.line([(jx * SS, (mid + w * 0.35) * SS), ((rx + w) * SS, (top - w * 1.1) * SS)], fill=255, width=round(w * 1.1 * SS))
    kx, ky = sx + w * 1.55, mid - w * 0.15          # the lower arm leaves the upper one here
    a.line([(kx * SS, ky * SS), ((rx + w * 0.5) * SS, (bot + w * 1.0) * SS)], fill=255, width=round(w * 1.1 * SS))
    box = Image.new("L", (big, big), 0)
    ImageDraw.Draw(box).rectangle((sx * SS, top * SS, rx * SS - 1, bot * SS - 1), fill=255)
    from PIL import ImageChops
    im.paste(Image.new("RGBA", (big, big), LIME), (0, 0), ImageChops.multiply(arms, box))
    return im.resize((n, n), Image.LANCZOS)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    images = [mark(n) for n in SIZES]
    out = os.path.join(here, "kindi.ico")
    images[-1].save(out, format="ICO", sizes=[(n, n) for n in SIZES], append_images=images[:-1])
    print("wrote", out, os.path.getsize(out), "bytes")
    if "--preview" in sys.argv:
        path = sys.argv[sys.argv.index("--preview") + 1]
        width = sum(n + 24 for n in SIZES[:-1]) + 24
        sheet = Image.new("RGBA", (width + 8 * 0, 2 * 96), (243, 243, 243, 255))
        dark = Image.new("RGBA", (width, 96), (32, 32, 32, 255))
        sheet.paste(dark, (0, 96))
        x = 24
        for im in images[:-1]:
            sheet.alpha_composite(im, (x, 48 - im.size[1] // 2))
            sheet.alpha_composite(im, (x, 144 - im.size[1] // 2))
            x += im.size[0] + 24
        big = sheet.resize((sheet.size[0] * 3, sheet.size[1] * 3), Image.NEAREST)   # 3x: the pixels are visible
        big.alpha_composite(images[-1].resize((192, 192), Image.LANCZOS), (big.size[0] - 0, 0)) if False else None
        big.save(path)
        images[-1].save(os.path.splitext(path)[0] + "-256.png")
        print("preview", path)


if __name__ == "__main__":
    main()
