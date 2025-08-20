import os
import sys
import json
import math
import time
import argparse
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


def download_video(url: str, out_path: str) -> str:
    """
    Download the video at url to out_path directory.
    Returns the local file path. Raises RuntimeError on failure.
    """
    os.makedirs(out_path, exist_ok=True)
    
    try:
        from yt_dlp import YoutubeDL
        ydl_opts = {
            "outtmpl": os.path.join(out_path, "%(title)s.%(ext)s"),
            "format": "mp4/best",
            "quiet": True,
            "noplaylist": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if not filename.endswith(".mp4"):
                
                mp4_name = os.path.splitext(filename)[0] + ".mp4"
                try:
                    cmd = f'ffmpeg -y -i "{filename}" -c copy "{mp4_name}"'
                    os.system(cmd)
                    if os.path.exists(mp4_name):
                        return mp4_name
                except Exception:
                    pass
            return filename
    except Exception as e:
        print(f"[download] yt-dlp failed: {e}")

    
    try:
        from pytube import YouTube
        yt = YouTube(url)
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution").desc().first()
        )
        if stream is None:
            raise RuntimeError("No suitable mp4 stream found via pytube.")
        fp = stream.download(output_path=out_path)
        return fp
    except Exception as e:
        raise RuntimeError(f"Failed to download video. Install yt-dlp or pytube. Error: {e}")


def angle_three_points(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Optional[float]:
    """Returns angle ABC in degrees given 2D points (A-B-C). Returns None if invalid."""
    try:
        ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
        bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
        nba = ba / (np.linalg.norm(ba) + 1e-9)
        nbc = bc / (np.linalg.norm(bc) + 1e-9)
        cosang = np.clip(np.dot(nba, nbc), -1.0, 1.0)
        ang = math.degrees(math.acos(cosang))
        return ang
    except Exception:
        return None

def angle_with_vertical(p_top: Tuple[float, float], p_bottom: Tuple[float, float]) -> Optional[float]:
    """Angle (in deg) between the line (bottom->top) and the vertical axis (negative y direction)."""
    try:
        dx = p_top[0] - p_bottom[0]
        dy = p_top[1] - p_bottom[1]
        v = np.array([dx, dy], dtype=float)
        v_norm = v / (np.linalg.norm(v) + 1e-9)
        cosang = np.clip(np.dot(v_norm, np.array([0.0, -1.0])), -1.0, 1.0)
        ang = math.degrees(math.acos(cosang))
        return ang
    except Exception:
        return None

def angle_with_horizontal(p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[float]:
    """Absolute angle (deg) of the line p1->p2 relative to +x axis (0..180)."""
    try:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        ang = math.degrees(math.atan2(dy, dx))
        return abs(ang)
    except Exception:
        return None

def dist_horizontal(p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[float]:
    try:
        return abs(p1[0] - p2[0])
    except Exception:
        return None


def put_text(img, text, org, scale=0.6, color=(255,255,255), thickness=2, bg=True):
    if bg:
        (w,h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(img, (org[0]-4, org[1]-h-4), (org[0]+w+4, org[1]+4), (0,0,0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_skeleton(img, kps, visibility_thresh=0.5):
    
    PAIRS = [
        ("LEFT_SHOULDER","RIGHT_SHOULDER"),
        ("LEFT_SHOULDER","LEFT_ELBOW"), ("LEFT_ELBOW","LEFT_WRIST"),
        ("RIGHT_SHOULDER","RIGHT_ELBOW"), ("RIGHT_ELBOW","RIGHT_WRIST"),
        ("LEFT_SHOULDER","LEFT_HIP"), ("RIGHT_SHOULDER","RIGHT_HIP"),
        ("LEFT_HIP","RIGHT_HIP"),
        ("LEFT_HIP","LEFT_KNEE"), ("LEFT_KNEE","LEFT_ANKLE"),
        ("RIGHT_HIP","RIGHT_KNEE"), ("RIGHT_KNEE","RIGHT_ANKLE"),
    ]
    for a,b in PAIRS:
        if a in kps and b in kps:
            pa = kps[a]; pb = kps[b]
            if pa[2] >= visibility_thresh and pb[2] >= visibility_thresh:
                cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0,255,0), 2)
    for name, (x,y,v) in kps.items():
        if v >= visibility_thresh:
            cv2.circle(img, (int(x), int(y)), 4, (0,128,255), -1)


def mediapipe_pose():
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError("mediapipe is required. Please `pip install mediapipe`") from e
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False, min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose

def extract_keypoints(results, width, height) -> Dict[str, Tuple[float,float,float]]:
    """Return dict name -> (x,y,visibility). Names aligned with MediaPipe's PoseLandmark."""
    kps = {}
    if not results or not results.pose_landmarks:
        return kps
    try:
        import mediapipe as mp
        lm = results.pose_landmarks.landmark
        def p(name):
            idx = mp.solutions.pose.PoseLandmark[name].value
            return (lm[idx].x * width, lm[idx].y * height, lm[idx].visibility)
        names = [
            "NOSE",
            "LEFT_SHOULDER","RIGHT_SHOULDER",
            "LEFT_ELBOW","RIGHT_ELBOW",
            "LEFT_WRIST","RIGHT_WRIST",
            "LEFT_HIP","RIGHT_HIP",
            "LEFT_KNEE","RIGHT_KNEE",
            "LEFT_ANKLE","RIGHT_ANKLE",
            "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX",
        ]
        for n in names:
            kps[n] = p(n)
        if "NOSE" in kps:
            kps["HEAD"] = kps["NOSE"]
    except Exception:
        pass
    return kps


def compute_metrics(kps: Dict[str, Tuple[float,float,float]], img_w: int, img_h: int, right_handed=True) -> Dict[str, Optional[float]]:
    """Compute per-frame biomechanical metrics. Assumes camera approx side-on."""
    front = "LEFT" if right_handed else "RIGHT"
    

    def g(name):
        return kps.get(name)

    metrics = {}
    
    S = g(f"{front}_SHOULDER")
    E = g(f"{front}_ELBOW")
    W = g(f"{front}_WRIST")
    if S and E and W:
        metrics["front_elbow_angle"] = angle_three_points(S[:2], E[:2], W[:2])
    else:
        metrics["front_elbow_angle"] = None

    
    LH, RH = g("LEFT_HIP"), g("RIGHT_HIP")
    LS, RS = g("LEFT_SHOULDER"), g("RIGHT_SHOULDER")
    if LH and RH and LS and RS:
        hip_mid = ((LH[0]+RH[0])/2.0, (LH[1]+RH[1])/2.0)
        sh_mid  = ((LS[0]+RS[0])/2.0, (LS[1]+RS[1])/2.0)
        metrics["spine_lean_deg"] = angle_with_vertical(sh_mid, hip_mid)  
    else:
        metrics["spine_lean_deg"] = None

    
    head = g("HEAD")
    FK = g(f"{front}_KNEE")
    if head and FK:
        metrics["head_knee_horiz_norm"] = dist_horizontal(head[:2], FK[:2]) / max(1.0, float(img_w))
    else:
        metrics["head_knee_horiz_norm"] = None

    
    F_ANK = g(f"{front}_ANKLE")
    F_FI  = g(f"{front}_FOOT_INDEX")
    if F_ANK and F_FI:
        metrics["front_foot_angle_deg"] = angle_with_horizontal(F_ANK[:2], F_FI[:2])
    else:
        metrics["front_foot_angle_deg"] = None

    return metrics

def feedback_from_metrics(metrics: Dict[str, Optional[float]], cfg: Dict) -> List[str]:
    cues = []
    thr = cfg["thresholds"]
    elbow_min = thr.get("front_elbow_min_deg", 100)
    elbow_max = thr.get("front_elbow_max_deg", 170)
    spine_max = thr.get("spine_lean_max_deg", 25)
    head_knee_max = thr.get("head_knee_horiz_max_norm", 0.05)
    foot_angle_max = thr.get("front_foot_angle_max_deg", 35)

    e = metrics.get("front_elbow_angle")
    s = metrics.get("spine_lean_deg")
    h = metrics.get("head_knee_horiz_norm")
    f = metrics.get("front_foot_angle_deg")

    if e is not None:
        cues.append("✅ Good elbow elevation" if elbow_min <= e <= elbow_max
                    else f"❌ Adjust front elbow (target ~{elbow_min}–{elbow_max}°)")
    if s is not None:
        cues.append("✅ Upright spine" if s <= spine_max else "❌ Excessive spine lean")
    if h is not None:
        cues.append("✅ Head over front knee" if h <= head_knee_max else "❌ Head not over front knee")
    if f is not None:
        cues.append("✅ Front foot aligned" if f <= foot_angle_max else "❌ Front foot too open")
    return cues

def score_summary(all_metrics: List[Dict[str, Optional[float]]], cfg: Dict) -> Dict:
    
    def avg(key):
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    e = avg("front_elbow_angle")
    s = avg("spine_lean_deg")
    h = avg("head_knee_horiz_norm")
    f = avg("front_foot_angle_deg")

    
    def band_score(val, good_range=None, inverse=False, max_val=None):
        if val is None:
            return 5
        if good_range:
            lo, hi = good_range
            if lo <= val <= hi:
                return 9
            if val < lo:
                return max(1, int(9 - (lo - val) / max(1e-6, lo) * 6))
            else:
                return max(1, int(9 - (val - hi) / max(1e-6, hi) * 6))
        if inverse and max_val:
            frac = min(1.0, val / max_val)  
            return max(1, int(10 - 9 * frac))
        return 5

    thr = cfg["thresholds"]
    scores = {}
    scores["Swing Control"] = band_score(e, (thr.get("front_elbow_min_deg",100), thr.get("front_elbow_max_deg",170)))
    scores["Balance"] = band_score(s, inverse=True, max_val=thr.get("spine_lean_max_deg",25))
    scores["Head Position"] = band_score(h, inverse=True, max_val=thr.get("head_knee_horiz_max_norm",0.05))
    scores["Footwork"] = band_score(f, inverse=True, max_val=thr.get("front_foot_angle_max_deg",35))

    
    def var(key):
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        return float(np.var(vals)) if len(vals) >= 2 else None
    smooth_penalty = 0.0
    v1, v2 = var("spine_lean_deg"), var("front_elbow_angle")
    for v in [v1, v2]:
        if v is not None:
            smooth_penalty += min(4.0, v/50.0)
    scores["Follow-through"] = max(1, int(9 - smooth_penalty))

    
    feedback = {
        "Footwork": "Aim to keep the front foot closer to the target line; reduce opening angle."
                    if (f is not None and f > thr.get("front_foot_angle_max_deg",35))
                    else "Front foot alignment looks controlled.",
        "Head Position": "Bring the head more over the front knee at release/impact frames."
                         if (h is not None and h > thr.get("head_knee_horiz_max_norm",0.05))
                         else "Head position is solid over the base.",
        "Swing Control": "Maintain front elbow within the target window for a freer arc."
                         if (e is not None and not (thr.get('front_elbow_min_deg',100) <= e <= thr.get('front_elbow_max_deg',170)))
                         else "Elbow elevation supports a clean arc.",
        "Balance": "Reduce side bend; try staying more upright through contact."
                   if (s is not None and s > thr.get("spine_lean_max_deg",25))
                   else "Spine angle is well-managed.",
        "Follow-through": "Work on smooth deceleration; minimize abrupt elbow/spine changes."
                          if smooth_penalty > 2.0 else "Follow-through flow is consistent."
    }

    return {
        "scores": scores,
        "feedback": feedback,
        "averages": {
            "front_elbow_angle": e,
            "spine_lean_deg": s,
            "head_knee_horiz_norm": h,
            "front_foot_angle_deg": f
        }
    }


def grade_skill(scores: dict) -> str:
    avg_score = sum(scores.values()) / max(1, len(scores))
    if avg_score < 4:
        return "Beginner"
    elif avg_score < 8:
        return "Intermediate"
    else:
        return "Advanced"

def safe_metric(m: Dict[str, Optional[float]], key: str, default: float) -> float:
    v = m.get(key)
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return default
    return float(v)

def segment_phases(metrics_per_frame: List[Dict[str, Optional[float]]]) -> Dict[str, int]:
    if not metrics_per_frame:
        return {"stance": 0, "impact": 0, "follow_through": 0}
   
    distances = [abs(safe_metric(m, "head_knee_horiz_norm", 1e6)) for m in metrics_per_frame]
    impact_idx = int(np.argmin(distances))
    
    elbows = [safe_metric(m, "front_elbow_angle", -1e6) for m in metrics_per_frame]
    if len(elbows) > impact_idx + 1:
        post_elbows = elbows[impact_idx:]
        follow_rel = int(np.argmax(post_elbows))
        follow_idx = impact_idx + follow_rel
    else:
        follow_idx = int(np.argmax(elbows))
    stance_idx = 0
    return {"stance": stance_idx, "impact": impact_idx, "follow_through": follow_idx}


def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)

def save_phase_thumbnail(src_path: str, frame_index: int, out_png: str, resize_w: int):
    cap = cv2.VideoCapture(src_path)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = resize_w / max(1, in_w)
    out_w = int(in_w * scale)
    out_h = int(in_h * scale)

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index-1))
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_png, frame)
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="https://youtube.com/shorts/vSX3IRxGnNY", help="Input video URL (YouTube/Shorts).")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs.")
    parser.add_argument("--right_handed", action="store_true", help="Assume batter is right-handed (front side = LEFT). Default True unless --left_handed used.")
    parser.add_argument("--left_handed", action="store_true", help="Assume batter is left-handed (front side = RIGHT).")
    parser.add_argument("--resize_width", type=int, default=640, help="Resize width for processing (keep aspect).")
    parser.add_argument("--target_fps", type=float, default=30.0, help="FPS for annotated video output.")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON to override thresholds.")
    parser.add_argument("--save_metrics_csv", action="store_true", help="Save per-frame metrics CSV.")
    parser.add_argument("--plot_metrics", action="store_true", help="Save elbow/spine vs frame PNGs.")
    args = parser.parse_args()

    
    cfg = {
        "thresholds": {
            "front_elbow_min_deg": 110,
            "front_elbow_max_deg": 170,
            "spine_lean_max_deg": 25,
            "head_knee_horiz_max_norm": 0.045,  
            "front_foot_angle_max_deg": 35
        }
    }
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            user_cfg = json.load(f)
        for k,v in user_cfg.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v

    ensure_dirs(args.output_dir)

    handed_right = True
    if args.left_handed:
        handed_right = False
    elif args.right_handed:
        handed_right = True

    
    print("[info] Downloading video...")
    src_path = download_video(args.url, args.output_dir)
    print(f"[info] Video saved at: {src_path}")

   
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print("[error] Could not open video.")
        sys.exit(1)

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    scale = args.resize_width / max(1, in_w)
    out_w = int(in_w * scale)
    out_h = int(in_h * scale)

    
    pose, mp_pose = mediapipe_pose()
    cap = cv2.VideoCapture(src_path)

    frame_idx = 0
    t0 = time.time()
    all_metrics: List[Dict[str, Optional[float]]] = []
    csv_rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        kps = extract_keypoints(results, out_w, out_h)

        m = compute_metrics(kps, out_w, out_h, right_handed=handed_right)
        all_metrics.append(m)

        if args.save_metrics_csv:
            csv_rows.append([
                frame_idx,
                m.get('front_elbow_angle'),
                m.get('spine_lean_deg'),
                m.get('head_knee_horiz_norm'),
                m.get('front_foot_angle_deg')
            ])

    cap.release()
    if 'pose' in locals():
        pose.close()

    elapsed = time.time() - t0
    fps = max(1e-6, frame_idx / elapsed)
    print(f"[info] Analyzed {frame_idx} frames in {elapsed:.2f}s (avg {fps:.2f} FPS).")

    # Final evaluation
    summary = score_summary(all_metrics, cfg)
    scores = summary["scores"]
    feedback = summary["feedback"]
    averages = summary["averages"]

    
    phases = segment_phases(all_metrics)
    skill_level = grade_skill(scores)

    
    if args.save_metrics_csv and csv_rows:
        import csv
        csv_path = os.path.join(args.output_dir, "metrics_per_frame.csv")
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["frame","front_elbow_angle_deg","spine_lean_deg","head_knee_horiz_norm","front_foot_angle_deg"])
            writer_csv.writerows(csv_rows)
        print(f"[info] Saved CSV: {csv_path}")

    
    if args.plot_metrics:
        try:
            import matplotlib.pyplot as plt
            frames = [i+1 for i in range(len(all_metrics))]
            elbow = [m.get("front_elbow_angle") for m in all_metrics]
            spine = [m.get("spine_lean_deg") for m in all_metrics]

            plt.figure()
            plt.plot(frames, elbow, label="Front Elbow (deg)")
            plt.xlabel("Frame")
            plt.ylabel("Degrees")
            plt.legend()
            plot1 = os.path.join(args.output_dir, "elbow_vs_frame.png")
            plt.savefig(plot1, dpi=160, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.plot(frames, spine, label="Spine Lean (deg)")
            plt.xlabel("Frame")
            plt.ylabel("Degrees")
            plt.legend()
            plot2 = os.path.join(args.output_dir, "spine_vs_frame.png")
            plt.savefig(plot2, dpi=160, bbox_inches="tight")
            plt.close()
            print(f"[info] Saved plots: {plot1}, {plot2}")
        except Exception as e:
            print(f"[warn] Failed to plot metrics: {e}")

    
    results = {
        "scores": scores,
        "feedback": feedback,
        "averages": averages,
        "runtime": {
            "frames": frame_idx,
            "elapsed_sec": round(elapsed, 2),
            "avg_fps": round(fps, 2)
        },
        "skill_level": skill_level,
        "phases": phases
    }
    eval_path = os.path.join(args.output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[info] Saved evaluation to {eval_path}")

    
    stance_png = os.path.join(args.output_dir, f"phase_stance_f{phases['stance']+1}.png")
    impact_png = os.path.join(args.output_dir, f"phase_impact_f{phases['impact']+1}.png")
    follow_png = os.path.join(args.output_dir, f"phase_follow_through_f{phases['follow_through']+1}.png")
    try:
        save_phase_thumbnail(src_path, phases["stance"]+1, stance_png, args.resize_width)
        save_phase_thumbnail(src_path, phases["impact"]+1, impact_png, args.resize_width)
        save_phase_thumbnail(src_path, phases["follow_through"]+1, follow_png, args.resize_width)
        print(f"[info] Saved phase thumbnails: {stance_png}, {impact_png}, {follow_png}")
    except Exception as e:
        print(f"[warn] Could not save phase thumbnails: {e}")

    
    pose, _ = mediapipe_pose()
    cap = cv2.VideoCapture(src_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = os.path.join(args.output_dir, "annotated_video.mp4")
    writer = cv2.VideoWriter(out_video_path, fourcc, args.target_fps, (out_w, out_h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(rgb)
        kps = extract_keypoints(results_pose, out_w, out_h)
        m = all_metrics[idx-1] if 0 <= idx-1 < len(all_metrics) else {}

        draw_skeleton(frame, kps)

        
        y = 24
        def fmt(x, p=1): return "NA" if (x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))) else f"{x:.{p}f}"
        put_text(frame, f"Frame: {idx}", (10, y)); y += 22
        put_text(frame, f"Elbow: {fmt(m.get('front_elbow_angle'))} deg", (10, y)); y += 22
        put_text(frame, f"Spine lean: {fmt(m.get('spine_lean_deg'))} deg", (10, y)); y += 22
        put_text(frame, f"Head-Knee (norm): {fmt(m.get('head_knee_horiz_norm'),3)}", (10, y)); y += 22
        put_text(frame, f"Front foot angle: {fmt(m.get('front_foot_angle_deg'))} deg", (10, y)); y += 22

        
        if idx-1 == phases["stance"]:
            put_text(frame, "[STANCE]", (int(out_w*0.65), 30), scale=0.9)
        if idx-1 == phases["impact"]:
            put_text(frame, "[IMPACT]", (int(out_w*0.65), 30), scale=0.9, color=(0,255,255))
        if idx-1 == phases["follow_through"]:
            put_text(frame, "[FOLLOW-THROUGH]", (int(out_w*0.55), 30), scale=0.9, color=(255,255,0))

        
        put_text(frame, f"Skill: {skill_level}", (int(out_w*0.55), 60), scale=0.7)

        cues = feedback_from_metrics(m, cfg)
        y2 = 90
        for c in cues:
            put_text(frame, c, (int(out_w*0.55), y2), scale=0.6)
            y2 += 22

        writer.write(frame)

    cap.release()
    writer.release()
    if 'pose' in locals():
        pose.close()

    print(f"[info] Annotated video: {out_video_path}")

if __name__ == "__main__":
    main()

