"""Entry point for the Forza Horizon 4 webcam hand-tracking Xbox 360 controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2

from controller import ViGEmDriverError, XboxController
from gestures import ControlInterpolator, ControlState, GestureProcessor, resolve_game_profile
from hud import draw_hud
from tracker import HandTracker

if sys.platform.startswith("win"):
    import ctypes
    import msvcrt
else:  # pragma: no cover - non-Windows fallback
    ctypes = None
    msvcrt = None


def _apply_window_icon(window_title: str, icon_path: Path) -> None:
    """Best-effort Windows titlebar/taskbar icon setter for the OpenCV preview window."""
    if ctypes is None or not icon_path.is_file():
        return
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        LR_DEFAULTSIZE = 0x0040
        WM_SETICON = 0x0080
        hicon = user32.LoadImageW(
            None,
            str(icon_path),
            IMAGE_ICON,
            0,
            0,
            LR_LOADFROMFILE | LR_DEFAULTSIZE,
        )
        if hicon:
            user32.SendMessageW(hwnd, WM_SETICON, 0, hicon)
            user32.SendMessageW(hwnd, WM_SETICON, 1, hicon)
    except Exception:
        pass


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration parameters from a JSON file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root in {config_path} must be a JSON object.")
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the FH4 hand controller."""
    parser = argparse.ArgumentParser(
        description="Low-latency webcam hand-tracking Xbox 360 controller for Forza Horizon 4."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.json",
        help="Path to config.json (default: ./config.json).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run tracking, gestures, and HUD without activating vgamepad.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the OpenCV preview window (recommended once calibrated for Steam/FH4).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Override webcam index from config.json.",
    )
    parser.add_argument(
        "--game",
        choices=["fh4", "fh5", "fh6", "motorsport"],
        default=None,
        help="Forza game profile: fh4 (default), fh5, fh6, or motorsport.",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Hide hand skeleton landmarks on the HUD overlay.",
    )
    return parser.parse_args(argv)


def _headless_quit_requested() -> bool:
    if msvcrt is None:
        return False
    while msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b"q", b"Q", b"\x1b"):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Initialize virtual Xbox 360 controller, camera tracker, and run the main loop."""
    args = parse_args(argv)
    base_dir = Path(__file__).resolve().parent

    try:
        config = load_config(args.config)
        game_key, profile = resolve_game_profile(config, game_override=args.game)
        config["active_game"] = game_key
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[Error] Failed to load config: {exc}", file=sys.stderr)
        return 1

    # Initialize virtual Xbox 360 controller BEFORE opening Forza (Steam / Xbox App)
    # so Steam/Windows enumerates the controller prior to game launch.
    try:
        controller = XboxController(debug=args.debug)
    except ViGEmDriverError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    if args.debug:
        print("[Controller] Running in --debug mode (vgamepad disabled).")
    else:
        print("[Controller] Virtual Xbox 360 controller active (ViGEmBus connected).")
        print(
            f"[Steam Tip] Launch {profile.get('label', game_key.upper())} now with Steam Input disabled."
        )

    try:
        tracker = HandTracker.from_config(
            config,
            base_dir=base_dir,
            camera_override=args.camera,
            open_camera=True,
        )
    except Exception as exc:
        controller.close()
        print(f"[Error] Failed to initialize HandTracker / webcam: {exc}", file=sys.stderr)
        return 1

    print(f"[Tracker] Ready ({tracker.backend_name}, {tracker.frame_width}x{tracker.frame_height}).")
    if args.headless:
        print("[HUD] Headless mode enabled — preview window disabled. Press 'q' or Ctrl+C to quit.")
    else:
        print("[HUD] Press 'q' in the preview window (or Ctrl+C in terminal) to exit cleanly.")

    processor = GestureProcessor(config, game=game_key)
    interpolator = ControlInterpolator(config, game=game_key)
    smooth_cfg = config.get("smoothing", {})
    control_hz = max(30, min(240, int(smooth_cfg.get("control_loop_hz", 120))))
    tick_period = 1.0 / float(control_hz)

    hud_cfg = config.get("hud", {})
    window_title = str(hud_cfg.get("window_title", "Forza Hand Controller"))
    show_landmarks = bool(hud_cfg.get("show_landmarks", True)) and not args.no_landmarks

    if not args.headless:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, tracker.frame_width, tracker.frame_height)
        _apply_window_icon(window_title, base_dir / "assets" / "forza_logo.ico")

    tracker.start_async()
    print(
        f"[Profile] Active game: {processor.game_label} "
        f"(Change Camera -> {processor.camera_button}) | {control_hz}Hz fluid interpolation."
    )

    cam_fps = 0.0
    last_pkt_seq = 0
    last_cam_ts: float | None = None
    last_tick_ts = time.perf_counter()
    last_headless_log = 0.0
    target_state = ControlState()

    try:
        while True:
            now = time.perf_counter()
            tick_dt = now - last_tick_ts
            last_tick_ts = now

            pkt = tracker.get_latest()
            if pkt is None:
                if not args.headless and (cv2.waitKey(5) & 0xFF) in (ord("q"), ord("Q")):
                    break
                if args.headless and _headless_quit_requested():
                    break
                time.sleep(0.004)
                continue

            new_frame = pkt.seq != last_pkt_seq
            if new_frame:
                last_pkt_seq = pkt.seq
                if last_cam_ts is not None:
                    cam_dt = pkt.timestamp - last_cam_ts
                    if cam_dt > 1e-4:
                        inst_fps = 1.0 / cam_dt
                        cam_fps = inst_fps if cam_fps == 0.0 else (0.15 * inst_fps + 0.85 * cam_fps)
                last_cam_ts = pkt.timestamp
                target_state = processor.update(pkt.hands, now=now)
                live_buttons = target_state.buttons
            else:
                live_buttons = processor.refresh_buttons(now=now)

            smoothed_state = interpolator.step(
                target_state,
                dt=tick_dt,
                buttons_override=live_buttons,
            )
            controller.apply(smoothed_state)

            if args.headless:
                if (now - last_headless_log) >= 1.0 or (new_frame and target_state.status_event):
                    active_btns = [k for k, v in smoothed_state.buttons.items() if v]
                    btn_str = ",".join(active_btns) if active_btns else "-"
                    print(
                        f"[{smoothed_state.mode.value}] CAM_FPS={cam_fps:4.1f} "
                        f"Steer={int(round(smoothed_state.steering * 100)):+4d}% "
                        f"RT={int(round(smoothed_state.throttle * 100)):3d}% "
                        f"LT={int(round(smoothed_state.brake * 100)):3d}% Btn={btn_str}"
                        + (f" | {target_state.status_event}" if (new_frame and target_state.status_event) else "")
                    )
                    last_headless_log = now
                if _headless_quit_requested():
                    break
            elif new_frame:
                hud_frame = pkt.frame.copy()
                draw_hud(
                    hud_frame,
                    state=smoothed_state,
                    hands=pkt.hands,
                    fps=cam_fps,
                    config=config,
                    debug_mode=args.debug,
                    show_landmarks=show_landmarks,
                )
                cv2.imshow(window_title, hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break

            elapsed = time.perf_counter() - now
            remaining = tick_period - elapsed
            if remaining > 0.001:
                time.sleep(remaining - 0.0005)
    except KeyboardInterrupt:
        print("\n[Main] Interrupt received — resetting controller and exiting...")
    finally:
        controller.close()
        tracker.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print("[Main] Clean exit complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
