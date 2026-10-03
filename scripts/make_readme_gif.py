"""Record the README hero GIF from the running dashboard's landing page ("RiskPulse in 60 seconds").

Needs the dashboard on http://localhost:8501 (`python -m riskpulse demo --fast`) and Playwright, a
local recording tool that is not a project dependency:
    pip install playwright && python -m playwright install chromium
    python scripts/make_readme_gif.py    # writes docs/riskpulse_60s.gif (must stay under 3 MB)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image

URL = "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "docs" / "riskpulse_60s.gif"
WIDTH, HEIGHT, OUT_WIDTH, COLORS = 1280, 760, 900, 96
MAX_BYTES = 3_000_000


def main() -> None:
    from playwright.sync_api import sync_playwright

    frames: list[tuple[Image.Image, int]] = []  # (frame, duration ms)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(URL, wait_until="networkidle")
        page.get_by_text("RiskPulse in 60 seconds").first.wait_for(timeout=60_000)
        page.locator(".js-plotly-plot").first.wait_for(timeout=60_000)
        page.wait_for_timeout(2500)
        # Collapse the sidebar for a cleaner frame, if the control is present.
        btn = page.locator('[data-testid="stSidebarCollapseButton"] button')
        try:
            btn.first.click(timeout=2000, force=True)
            page.wait_for_timeout(600)
        except Exception:  # control hidden in this Streamlit version: keep the sidebar
            pass

        def snap(ms: int) -> None:
            img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
            img = img.resize((OUT_WIDTH, round(img.height * OUT_WIDTH / img.width)), Image.LANCZOS)
            frames.append((img, ms))

        snap(1800)  # title and the timeline
        point = page.locator(".js-plotly-plot .scatterlayer .point").first
        box = point.bounding_box() if point.count() else None
        if box:  # Plotly hovers via its drag layer: move the mouse to the point's coordinates
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.wait_for_timeout(700)
            snap(2800)  # tooltip on the 22 Feb 00:00 UTC trigger
        page.mouse.move(5, 5)
        for heading, hold in (
            ("2 · The 24 February stress run", 2600),
            ("3 · Predicted vs realised", 2800),
            ("4 · Headline results", 2800),
        ):
            target = page.get_by_text(heading).first
            # Scroll in a few steps so the motion reads as a scroll, not a cut.
            box = target.bounding_box()
            steps = 4
            for _ in range(steps):
                page.mouse.wheel(0, (box["y"] - 70) / steps if box else 150)
                page.wait_for_timeout(120)
                snap(140)
            target.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            snap(hold)
        browser.close()

    pal = [f.quantize(colors=COLORS, method=Image.Quantize.MEDIANCUT) for f, _ in frames]
    pal[0].save(
        OUT,
        save_all=True,
        append_images=pal[1:],
        duration=[d for _, d in frames],
        loop=0,
        optimize=True,
    )
    size, total = OUT.stat().st_size, sum(d for _, d in frames) / 1000
    print(f"{OUT.name}: {len(frames)} frames, {total:.1f} s, {size / 1e6:.2f} MB")
    if size > MAX_BYTES:
        sys.exit("GIF over 3 MB: lower OUT_WIDTH or COLORS")


if __name__ == "__main__":
    main()
