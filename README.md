# AthleteRise – Real-Time Cover Drive Analysis

Processes the **entire** YouTube video in real time (no keyframe exports), runs **MediaPipe Pose** per frame, overlays live biomechanical metrics, and produces:
- `output/annotated_video.mp4`
- `output/evaluation.json` (scores + actionable feedback)
- (optional) `output/metrics_per_frame.csv` and PNG plots

Tested with input: 2025-08-17 – YouTube Short `https://youtube.com/shorts/vSX3IRxGnNY`

## Quickstart

```bash
# 1) Create & activate venv (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run (right-handed batter by default)
python cover_drive_analysis_realtime.py --url "https://youtube.com/shorts/vSX3IRxGnNY" --save_metrics_csv --plot_metrics
```

Outputs are saved in `output/`.

### Notes
- The script first tries **yt-dlp** to download; if unavailable, falls back to **pytube**.
- If yt-dlp outputs `.webm`, we attempt to **remux to .mp4** with `ffmpeg` (optional). If you don't have ffmpeg, most players still read `.webm`.
- To assume left-handed batter:
```bash
python cover_drive_analysis_realtime.py --left_handed
```

## What the script computes (per frame)
- **Front elbow angle** (shoulder–elbow–wrist)
- **Spine lean** vs vertical (mid-hip → mid-shoulder line)
- **Head-over-knee** horizontal offset (normalized by frame width)
- **Front foot direction** (foot-index to ankle angle vs x-axis)

### Live Overlays
- Pose skeleton
- Metric readouts (e.g., `Elbow: 115°`)
- ✅ / ❌ feedback cues using thresholds in-code (editable)

## Final Evaluation
Generates 1–10 scores for:
- Footwork
- Head Position
- Swing Control
- Balance
- Follow-through

Plus short actionable feedback, and runtime stats.

## Troubleshooting
- If MediaPipe fails to import on Apple Silicon, install with: `pip install mediapipe-silicon` (community build) or use Python 3.10+.
- If download fails, ensure `yt-dlp` is installed, or try `pip install pytube`.
- For performance, use `--resize_width 640` (default) or 480.
- To log average FPS, check console output at the end.

## License
For assignment/demo use.
