"""Lightweight high-contrast OpenCV HUD for the FH4 hand-tracking controller."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import cv2
import numpy as np

from gestures import ControlMode, ControlState, resolve_game_profile
from tracker import HandLabel, TrackedHand

# Standard 21-point MediaPipe hand skeleton connections
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# BGR colors for high contrast on webcam feeds
COLOR_BG = (18, 20, 26)
COLOR_BORDER = (75, 85, 102)
COLOR_WHITE = (242, 245, 250)
COLOR_MUTED = (155, 165, 180)
COLOR_DRIVING = (72, 218, 125)
COLOR_MENU = (45, 185, 255)
COLOR_FAILSAFE = (70, 75, 245)
COLOR_THROTTLE = (72, 225, 110)
COLOR_BRAKE = (72, 85, 245)
COLOR_LEFT_HAND = (245, 205, 55)
COLOR_RIGHT_HAND = (55, 175, 255)
COLOR_ACTIVE_BTN = (80, 235, 140)

BUTTON_PILLS: tuple[tuple[str, str], ...] = (
    ("A", "A"),
    ("B", "B"),
    ("X", "X"),
    ("Y", "Y"),
    ("LB", "LB"),
    ("RB", "RB"),
    ("BACK", "VIEW"),
    ("START", "MENU"),
    ("DPAD_UP", "UP"),
    ("DPAD_DOWN", "DN"),
    ("DPAD_LEFT", "LT"),
    ("DPAD_RIGHT", "RT"),
)


def _draw_landmarks(frame: np.ndarray, hands: dict[HandLabel, TrackedHand]) -> None:
    h, w = frame.shape[:2]
    for label, hand in hands.items():
        color = COLOR_LEFT_HAND if label == "Left" else COLOR_RIGHT_HAND
        pts = hand.to_pixel_xy(w, h)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for idx, pt in enumerate(pts):
            radius = 5 if idx in (0, 4, 8, 12, 16, 20) else 3
            fill = COLOR_WHITE if idx in (4, 8, 12) else color
            cv2.circle(frame, tuple(pt), radius, fill, -1, cv2.LINE_AA)
        wrist = pts[0]
        cv2.putText(
            frame,
            label.upper(),
            (int(wrist[0]) - 22, min(h - 8, int(wrist[1]) + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_top_bar(
    frame: np.ndarray,
    state: ControlState,
    fps: float,
    debug_mode: bool,
    game_label: str = "FH4",
) -> None:
    _, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), COLOR_BG, -1)
    cv2.line(frame, (0, 42), (w, 42), COLOR_BORDER, 1)

    mode_color = COLOR_DRIVING if state.mode == ControlMode.DRIVING else COLOR_MENU
    cv2.rectangle(frame, (10, 8), (146, 34), mode_color, -1)
    cv2.putText(
        frame,
        state.mode.value,
        (22, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (15, 18, 22),
        2,
        cv2.LINE_AA,
    )

    if state.failsafe_active:
        badge_text = "FAILSAFE"
        badge_color = COLOR_FAILSAFE
    elif debug_mode:
        badge_text = "DEBUG"
        badge_color = COLOR_MENU
    else:
        badge_text = "X360 LIVE"
        badge_color = COLOR_DRIVING

    cv2.putText(
        frame,
        badge_text,
        (158, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        badge_color,
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(frame, (272, 10), (336, 32), (38, 44, 56), -1)
    cv2.rectangle(frame, (272, 10), (336, 32), COLOR_BORDER, 1)
    cv2.putText(
        frame,
        game_label,
        (284, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )

    l_col = COLOR_LEFT_HAND if state.left_present else COLOR_BORDER
    r_col = COLOR_RIGHT_HAND if state.right_present else COLOR_BORDER
    cv2.putText(frame, "L", (w - 175, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, l_col, 2, cv2.LINE_AA)
    cv2.putText(frame, "R", (w - 150, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, r_col, 2, cv2.LINE_AA)

    fps_text = f"FPS: {fps:4.1f}"
    cv2.putText(
        frame,
        fps_text,
        (w - 115, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )


def _draw_steering_gauge(
    frame: np.ndarray,
    state: ControlState,
    angle_max_deg: float,
    deadzone_deg: float,
) -> None:
    h, w = frame.shape[:2]
    cx = w // 2
    cy = h - 54
    radius = 62

    # Semi-circular steering wheel arc (-165 to -15 deg in screen coordinates)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 195, 345, COLOR_BORDER, 4, cv2.LINE_AA)

    # Deadzone highlight around top-center (270 deg)
    dz_span = (deadzone_deg / max(1.0, angle_max_deg)) * 75.0
    cv2.ellipse(
        frame,
        (cx, cy),
        (radius, radius),
        0,
        270.0 - dz_span,
        270.0 + dz_span,
        COLOR_WHITE,
        5,
        cv2.LINE_AA,
    )

    # Smoothed steering needle
    steer_deg = state.steering * 75.0
    needle_rad = math.radians(270.0 + steer_deg)
    nx = int(cx + (radius - 6) * math.cos(needle_rad))
    ny = int(cy + (radius - 6) * math.sin(needle_rad))
    cv2.line(frame, (cx, cy), (nx, ny), COLOR_DRIVING, 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 5, COLOR_WHITE, -1, cv2.LINE_AA)

    steer_pct = int(round(state.steering * 100.0))
    label = f"STEER {steer_pct:+4d}% ({state.raw_steering_angle_deg:+5.1f}d)"
    cv2.rectangle(frame, (cx - 104, cy + 6), (cx + 104, cy + 28), COLOR_BG, -1)
    cv2.rectangle(frame, (cx - 104, cy + 6), (cx + 104, cy + 28), COLOR_BORDER, 1)
    cv2.putText(
        frame,
        label,
        (cx - 96, cy + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )


def _draw_driving_telemetry_strip(frame: np.ndarray, state: ControlState) -> None:
    """Draw high-contrast live Steering %, RT %, and LT % readout below top bar."""
    _, w = frame.shape[:2]
    steer_pct = int(round(state.steering * 100.0))
    rt_pct = int(round(state.throttle * 100.0))
    lt_pct = int(round(state.brake * 100.0))
    text = f"STEER: {steer_pct:+4d}%   |   RT: {rt_pct:3d}%   |   LT: {lt_pct:3d}%"
    x1, y1, x2, y2 = max(8, w // 2 - 195), 48, min(w - 8, w // 2 + 195), 72
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BG, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BORDER, 1)
    cv2.putText(
        frame,
        text,
        (x1 + 12, y2 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )


def _draw_trigger_bars(frame: np.ndarray, state: ControlState) -> None:
    h, w = frame.shape[:2]
    bar_w = 26
    bar_h = 150
    top_y = max(82, (h - bar_h) // 2)
    bottom_y = top_y + bar_h

    # Left Trigger (Brake)
    lx = 16
    lt_pct = int(round(state.brake * 100.0))
    cv2.rectangle(frame, (lx, top_y), (lx + bar_w, bottom_y), COLOR_BG, -1)
    cv2.rectangle(frame, (lx, top_y), (lx + bar_w, bottom_y), COLOR_BORDER, 1)
    lt_fill = int(round(state.brake * bar_h))
    if lt_fill > 0:
        cv2.rectangle(frame, (lx + 2, bottom_y - lt_fill), (lx + bar_w - 2, bottom_y - 2), COLOR_BRAKE, -1)
    cv2.putText(frame, "LT", (lx + 3, top_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (lx - 10, bottom_y + 4), (lx + bar_w + 16, bottom_y + 24), COLOR_BG, -1)
    cv2.putText(
        frame,
        f"{lt_pct:3d}%",
        (lx - 6, bottom_y + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_WHITE if lt_pct > 0 else COLOR_MUTED,
        1,
        cv2.LINE_AA,
    )

    # Right Trigger (Gas)
    rx = w - 16 - bar_w
    rt_pct = int(round(state.throttle * 100.0))
    cv2.rectangle(frame, (rx, top_y), (rx + bar_w, bottom_y), COLOR_BG, -1)
    cv2.rectangle(frame, (rx, top_y), (rx + bar_w, bottom_y), COLOR_BORDER, 1)
    rt_fill = int(round(state.throttle * bar_h))
    if rt_fill > 0:
        cv2.rectangle(frame, (rx + 2, bottom_y - rt_fill), (rx + bar_w - 2, bottom_y - 2), COLOR_THROTTLE, -1)
    cv2.putText(frame, "RT", (rx + 3, top_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (rx - 14, bottom_y + 4), (rx + bar_w + 12, bottom_y + 24), COLOR_BG, -1)
    cv2.putText(
        frame,
        f"{rt_pct:3d}%",
        (rx - 10, bottom_y + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_WHITE if rt_pct > 0 else COLOR_MUTED,
        1,
        cv2.LINE_AA,
    )


def _draw_menu_overlay(
    frame: np.ndarray,
    state: ControlState,
    box_cx: float,
    box_cy: float,
    box_hw: float,
    box_hh: float,
) -> None:
    h, w = frame.shape[:2]
    x1 = int((box_cx - box_hw) * w)
    y1 = int((box_cy - box_hh) * h)
    x2 = int((box_cx + box_hw) * w)
    y2 = int((box_cy + box_hh) * h)

    box_color = COLOR_MENU if state.active_dpad_zone is None else COLOR_ACTIVE_BTN
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        state.active_dpad_zone.replace("DPAD_", "DPAD ") if state.active_dpad_zone else "MENU CENTER",
        (x1, max(62, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        box_color,
        1,
        cv2.LINE_AA,
    )

    if state.menu_cursor is not None:
        cx_px = int(state.menu_cursor[0] * w)
        cy_px = int(state.menu_cursor[1] * h)
        cv2.circle(frame, (cx_px, cy_px), 8, COLOR_MENU, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx_px, cy_px), 10, COLOR_WHITE, 1, cv2.LINE_AA)


def _draw_progress_banner(frame: np.ndarray, title: str, progress: float, color: tuple[int, int, int]) -> None:
    if progress <= 0.0:
        return
    _, w = frame.shape[:2]
    bx1, by1, bx2, by2 = w // 2 - 140, 52, w // 2 + 140, 76
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_BG, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
    fill_w = int((bx2 - bx1 - 4) * min(1.0, max(0.0, progress)))
    if fill_w > 0:
        cv2.rectangle(frame, (bx1 + 2, by1 + 2), (bx1 + 2 + fill_w, by2 - 2), color, -1)
    cv2.putText(
        frame,
        title,
        (bx1 + 8, by2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )


def _draw_button_strip(frame: np.ndarray, state: ControlState) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), COLOR_BG, -1)
    cv2.line(frame, (0, h - 30), (w, h - 30), COLOR_BORDER, 1)

    x = 8
    y_top = h - 24
    y_bot = h - 6
    for key, label in BUTTON_PILLS:
        active = bool(state.buttons.get(key, False))
        pill_w = max(28, 12 + len(label) * 9)
        if x + pill_w > w - 65:
            break
        bg = COLOR_ACTIVE_BTN if active else (34, 38, 48)
        fg = (15, 18, 22) if active else COLOR_MUTED
        cv2.rectangle(frame, (x, y_top), (x + pill_w, y_bot), bg, -1)
        cv2.putText(frame, label, (x + 6, y_bot - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, fg, 1, cv2.LINE_AA)
        x += pill_w + 5

    cv2.putText(frame, "[Q] Quit", (w - 64, y_bot - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MUTED, 1, cv2.LINE_AA)


@lru_cache(maxsize=4)
def _load_logo_rgba(resolved_path: str, max_w: int, max_h: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load, resize, and pre-split BGR and float32 alpha mask for fast HUD compositing."""
    path = Path(resolved_path)
    if not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]
    scale = min(float(max_w) / max(1, w), float(max_h) / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if resized.ndim == 2:
        bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR).astype(np.float32)
        alpha = (resized.astype(np.float32) / 255.0)[..., None]
    elif resized.shape[2] == 4:
        bgr = resized[:, :, :3].astype(np.float32)
        alpha = (resized[:, :, 3].astype(np.float32) / 255.0)[..., None]
    else:
        bgr = resized.astype(np.float32)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        alpha = np.clip((gray - 12.0) / 225.0, 0.0, 1.0)[..., None]
    return bgr, alpha


def _draw_forza_logo(frame: np.ndarray, config: dict) -> None:
    """Draw a compact Forza emblem badge in the bottom-right corner above the button strip."""
    hud_cfg = config.get("hud", {})
    if not bool(hud_cfg.get("show_logo", True)):
        return

    raw_logo = Path(str(hud_cfg.get("logo_path", "assets/forza_logo.png")))
    if not raw_logo.is_absolute():
        raw_logo = Path(__file__).resolve().parent / raw_logo
    loaded = _load_logo_rgba(str(raw_logo), max_w=78, max_h=48)
    if loaded is None:
        return

    logo_bgr, logo_alpha = loaded
    lh, lw = logo_bgr.shape[:2]
    h, w = frame.shape[:2]

    pad_x, pad_y = 8, 5
    box_w = lw + 2 * pad_x
    box_h = lh + 2 * pad_y
    x2 = w - 10
    y2 = h - 36
    x1 = x2 - box_w
    y1 = y2 - box_h
    if x1 < 0 or y1 < 0:
        return

    roi = frame[y1:y2, x1:x2]
    bg_tint = np.full_like(roi, COLOR_BG, dtype=np.uint8)
    cv2.addWeighted(bg_tint, 0.78, roi, 0.22, 0.0, dst=roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BORDER, 1)

    lx1 = x1 + pad_x
    ly1 = y1 + pad_y
    target_roi = frame[ly1 : ly1 + lh, lx1 : lx1 + lw].astype(np.float32)
    blended = logo_bgr * logo_alpha + target_roi * (1.0 - logo_alpha)
    frame[ly1 : ly1 + lh, lx1 : lx1 + lw] = np.clip(blended, 0, 255).astype(np.uint8)


def draw_hud(
    frame: np.ndarray,
    state: ControlState,
    hands: dict[HandLabel, TrackedHand],
    fps: float,
    config: dict,
    debug_mode: bool = False,
    show_landmarks: bool = True,
) -> np.ndarray:
    """Render the complete high-contrast telemetry overlay onto ``frame``."""
    if show_landmarks and hands:
        _draw_landmarks(frame, hands)

    _, profile = resolve_game_profile(config)
    game_label = str(profile.get("label", "FH4"))
    _draw_top_bar(frame, state=state, fps=fps, debug_mode=debug_mode, game_label=game_label)

    if state.mode == ControlMode.DRIVING:
        steer_cfg = config.get("steering", {})
        _draw_driving_telemetry_strip(frame, state)
        _draw_trigger_bars(frame, state)
        _draw_steering_gauge(
            frame,
            state,
            angle_max_deg=float(steer_cfg.get("angle_max_deg", 45.0)),
            deadzone_deg=float(steer_cfg.get("center_deadzone_deg", 5.0)),
        )
    else:
        menu_cfg = config.get("menu", {})
        _draw_menu_overlay(
            frame,
            state,
            box_cx=float(menu_cfg.get("box_center_x", 0.50)),
            box_cy=float(menu_cfg.get("box_center_y", 0.50)),
            box_hw=float(menu_cfg.get("box_half_width", 0.10)),
            box_hh=float(menu_cfg.get("box_half_height", 0.12)),
        )

    if state.wrist_cross_progress > 0.0:
        _draw_progress_banner(
            frame,
            f"HOLD WRISTS -> TOGGLE MODE ({int(state.wrist_cross_progress * 100)}%)",
            state.wrist_cross_progress,
            COLOR_MENU,
        )
    elif state.camera_hold_progress > 0.0:
        _draw_progress_banner(
            frame,
            f"HOLD PEACE -> CAMERA ({int(state.camera_hold_progress * 100)}%)",
            state.camera_hold_progress,
            COLOR_DRIVING,
        )
    elif state.status_event:
        cv2.putText(
            frame,
            state.status_event,
            (12, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_ACTIVE_BTN,
            2,
            cv2.LINE_AA,
        )

    _draw_forza_logo(frame, config)
    _draw_button_strip(frame, state)
    return frame
