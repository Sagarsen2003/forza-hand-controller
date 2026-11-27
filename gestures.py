"""Gesture recognition state machine for Forza Horizon 4 (DRIVING and MENU modes).

Converts normalized 3D hand landmarks labeled by handedness into normalized
analog steering/trigger values, discrete button states, and mode transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any

import numpy as np

from tracker import HandLabel, TrackedHand

# MediaPipe 21-landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGERTIP_INDICES = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

ALL_BUTTONS = (
    "A",
    "B",
    "X",
    "Y",
    "LB",
    "RB",
    "START",
    "BACK",
    "DPAD_UP",
    "DPAD_DOWN",
    "DPAD_LEFT",
    "DPAD_RIGHT",
)


class ControlMode(str, Enum):
    """Active control mode for gesture translation."""

    DRIVING = "DRIVING"
    MENU = "MENU"


@dataclass(slots=True)
class ControlState:
    """Normalized Xbox 360 control state and HUD telemetry for one frame."""

    mode: ControlMode = ControlMode.DRIVING
    steering: float = 0.0
    raw_steering_angle_deg: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    buttons: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in ALL_BUTTONS}
    )
    hands_present: int = 0
    left_present: bool = False
    right_present: bool = False
    failsafe_active: bool = False
    wrist_cross_progress: float = 0.0
    camera_hold_progress: float = 0.0
    menu_cursor: tuple[float, float] | None = None
    active_dpad_zone: str | None = None
    status_event: str | None = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


DEFAULT_GAME_PROFILES: dict[str, dict[str, Any]] = {
    "fh4": {
        "label": "FH4",
        "camera_button": "BACK",
        "steering_spring_hz": 4.8,
        "trigger_spring_hz": 6.5,
    },
    "fh5": {
        "label": "FH5",
        "camera_button": "RB",
        "steering_spring_hz": 4.8,
        "trigger_spring_hz": 6.5,
    },
    "fh6": {
        "label": "FH6",
        "camera_button": "RB",
        "steering_spring_hz": 4.8,
        "trigger_spring_hz": 6.5,
    },
    "motorsport": {
        "label": "FM",
        "camera_button": "RB",
        "steering_spring_hz": 5.4,
        "trigger_spring_hz": 7.0,
    },
}


def resolve_game_profile(
    config: dict[str, Any], game_override: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Return ``(profile_key, profile_dict)`` for the selected Forza title."""
    profiles = config.get("game_profiles", {})
    key = (game_override or config.get("active_game") or "fh4").strip().lower()
    if key in ("fm", "forza_motorsport"):
        key = "motorsport"
    base = dict(DEFAULT_GAME_PROFILES.get(key, DEFAULT_GAME_PROFILES["fh4"]))
    if isinstance(profiles, dict) and isinstance(profiles.get(key), dict):
        base.update(profiles[key])
    elif key not in DEFAULT_GAME_PROFILES:
        raise ValueError(
            f"Unsupported game profile '{key}'. Supported: fh4, fh5, fh6, motorsport."
        )
    cam_btn = str(base.get("camera_button", "BACK")).upper()
    if cam_btn not in ALL_BUTTONS:
        cam_btn = "BACK"
    base["camera_button"] = cam_btn
    return key, base


def _dt_ema_alpha(base_alpha_60hz: float, dt: float) -> float:
    """Scale a 60Hz EMA alpha for frame-rate-independent smoothing."""
    alpha = _clamp(base_alpha_60hz, 0.0, 1.0)
    if alpha <= 0.0 or alpha >= 1.0 or dt <= 0.0:
        return alpha
    # Reset lag softly across large frame gaps (>0.25s) while remaining consistent across 30-120fps.
    clamped_dt = min(dt, 0.25)
    return float(1.0 - (1.0 - alpha) ** (clamped_dt * 60.0))


class OneEuroArrayFilter:
    """Vectorized One-Euro low-pass filter for adaptive jitter removal with low lag."""

    def __init__(
        self,
        min_cutoff: float = 1.8,
        beta: float = 6.0,
        d_cutoff: float = 1.0,
        reset_gap_sec: float = 0.25,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset_gap_sec = float(reset_gap_sec)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(dt: float, cutoff: float | np.ndarray) -> float | np.ndarray:
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def filter(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        """Filter an array of landmark coordinates at ``timestamp`` seconds."""
        x = np.asarray(value, dtype=np.float32)
        if (
            self._x_prev is None
            or self._dx_prev is None
            or self._t_prev is None
            or (timestamp - self._t_prev) > self.reset_gap_sec
            or timestamp <= self._t_prev
        ):
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = float(timestamp)
            return x.copy()

        dt = max(1e-4, float(timestamp - self._t_prev))
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx = (x - self._x_prev) / dt
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._alpha(dt, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = float(timestamp)
        return x_hat.copy()

    def reset(self) -> None:
        """Clear filter history."""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None


class CriticallyDampedSmoother:
    """C1-continuous critically-damped spring smoother with zero overshoot."""

    def __init__(self, natural_freq_hz: float = 4.8) -> None:
        self.omega = max(0.5, 2.0 * math.pi * float(natural_freq_hz))
        self.value = 0.0
        self.velocity = 0.0

    def reset(self, value: float = 0.0) -> None:
        """Snap value immediately and zero spring velocity."""
        self.value = float(value)
        self.velocity = 0.0

    def step(
        self,
        target: float,
        dt: float,
        snap_zero: bool = False,
        snap_one: bool = False,
    ) -> float:
        """Advance spring analytically by ``dt`` seconds toward ``target``."""
        if dt <= 0.0:
            return self.value
        clamped_dt = min(dt, 0.1)
        err = self.value - float(target)
        temp = (self.velocity + self.omega * err) * clamped_dt
        decay = math.exp(-self.omega * clamped_dt)
        self.velocity = (self.velocity - self.omega * temp) * decay
        self.value = float(target) + (err + temp) * decay

        if snap_zero and abs(target) < 1e-6 and abs(self.value) < 0.004 and abs(self.velocity) < 0.04:
            self.value = 0.0
            self.velocity = 0.0
        if snap_one and target >= 0.999 and self.value >= 0.992:
            self.value = 1.0
            self.velocity = 0.0
        return self.value


class ControlInterpolator:
    """Sub-frame 120Hz interpolator turning ~30Hz webcam updates into fluid motion."""

    def __init__(self, config: dict[str, Any], game: str | None = None) -> None:
        smooth = config.get("smoothing", {})
        _, profile = resolve_game_profile(config, game_override=game)
        self.enabled = bool(smooth.get("enabled", True))
        steer_hz = float(profile.get("steering_spring_hz", smooth.get("steering_spring_hz", 4.8)))
        trig_hz = float(profile.get("trigger_spring_hz", smooth.get("trigger_spring_hz", 6.5)))
        self._steer = CriticallyDampedSmoother(steer_hz)
        self._throttle = CriticallyDampedSmoother(trig_hz)
        self._brake = CriticallyDampedSmoother(trig_hz)

    def reset(self) -> None:
        """Reset all interpolated analog channels to neutral."""
        self._steer.reset(0.0)
        self._throttle.reset(0.0)
        self._brake.reset(0.0)

    def step(
        self,
        target: ControlState,
        dt: float,
        buttons_override: dict[str, bool] | None = None,
    ) -> ControlState:
        """Advance smooth interpolation toward ``target`` over ``dt`` seconds."""
        buttons = dict(buttons_override if buttons_override is not None else target.buttons)
        if not self.enabled or target.failsafe_active or target.mode == ControlMode.MENU:
            self.reset()
            return ControlState(
                mode=target.mode,
                steering=target.steering,
                raw_steering_angle_deg=target.raw_steering_angle_deg,
                throttle=target.throttle,
                brake=target.brake,
                buttons=buttons,
                hands_present=target.hands_present,
                left_present=target.left_present,
                right_present=target.right_present,
                failsafe_active=target.failsafe_active,
                wrist_cross_progress=target.wrist_cross_progress,
                camera_hold_progress=target.camera_hold_progress,
                menu_cursor=target.menu_cursor,
                active_dpad_zone=target.active_dpad_zone,
                status_event=target.status_event,
            )

        steer = _clamp(self._steer.step(target.steering, dt, snap_zero=True), -1.0, 1.0)
        throttle = _clamp(
            self._throttle.step(target.throttle, dt, snap_zero=True, snap_one=True),
            0.0,
            1.0,
        )
        if target.brake == 0.0 and buttons.get("X", False):
            # Keep Shift-Down guard instant so LT never lags above 0% during downshift.
            self._brake.reset(0.0)
            brake = 0.0
        else:
            brake = _clamp(
                self._brake.step(target.brake, dt, snap_zero=True, snap_one=True),
                0.0,
                1.0,
            )

        return ControlState(
            mode=target.mode,
            steering=steer,
            raw_steering_angle_deg=target.raw_steering_angle_deg,
            throttle=throttle,
            brake=brake,
            buttons=buttons,
            hands_present=target.hands_present,
            left_present=target.left_present,
            right_present=target.right_present,
            failsafe_active=target.failsafe_active,
            wrist_cross_progress=target.wrist_cross_progress,
            camera_hold_progress=target.camera_hold_progress,
            menu_cursor=target.menu_cursor,
            active_dpad_zone=target.active_dpad_zone,
            status_event=target.status_event,
        )


class GestureProcessor:
    """Stateful two-hand gesture interpreter for DRIVING and MENU modes."""

    def __init__(self, config: dict[str, Any], game: str | None = None) -> None:
        cam = config.get("camera", {})
        smooth = config.get("smoothing", {})
        steer = config.get("steering", {})
        trig = config.get("triggers", {})
        pinch = config.get("pinch_thresholds", {})
        gest = config.get("gestures", {})
        menu = config.get("menu", {})
        self.game_key, profile = resolve_game_profile(config, game_override=game)
        self.game_label = str(profile.get("label", self.game_key.upper()))
        self.camera_button = str(profile.get("camera_button", "BACK"))

        self.frame_width = int(cam.get("frame_width", 640))
        self.frame_height = int(cam.get("frame_height", 480))
        self.aspect_ratio = float(self.frame_width) / max(1.0, float(self.frame_height))
        self.mirror = bool(cam.get("mirror", True))

        # Adaptive landmark smoothing (One-Euro Filter)
        self.smoothing_enabled = bool(smooth.get("enabled", True))
        self._landmark_filters: dict[HandLabel, OneEuroArrayFilter] = {
            "Left": OneEuroArrayFilter(
                min_cutoff=float(smooth.get("landmark_min_cutoff", 1.8)),
                beta=float(smooth.get("landmark_beta", 6.0)),
                d_cutoff=float(smooth.get("landmark_d_cutoff", 1.0)),
                reset_gap_sec=float(smooth.get("landmark_reset_gap_sec", 0.25)),
            ),
            "Right": OneEuroArrayFilter(
                min_cutoff=float(smooth.get("landmark_min_cutoff", 1.8)),
                beta=float(smooth.get("landmark_beta", 6.0)),
                d_cutoff=float(smooth.get("landmark_d_cutoff", 1.0)),
                reset_gap_sec=float(smooth.get("landmark_reset_gap_sec", 0.25)),
            ),
        }

        # Steering config
        self.steering_angle_max_deg = float(steer.get("angle_max_deg", 45.0))
        self.steering_deadzone_deg = float(steer.get("center_deadzone_deg", 5.0))
        self.steering_curve_exponent = float(steer.get("curve_exponent", 1.20))
        self.steering_ema_alpha = float(steer.get("ema_alpha", 0.45))
        self.steering_single_hand_grace_frames = int(steer.get("single_hand_grace_frames", 4))

        # Trigger config
        self.fist_open_ratio = float(trig.get("fist_open_ratio", 1.65))
        self.fist_closed_ratio = float(trig.get("fist_closed_ratio", 1.20))
        self.trigger_deadzone = float(trig.get("trigger_deadzone", 0.06))
        self.trigger_curve_exponent = float(trig.get("curve_exponent", 0.85))
        self.trigger_ema_alpha = float(trig.get("ema_alpha", 0.65))

        # Pinch config
        self.pinch_thumb_index = float(pinch.get("thumb_index", 0.35))
        self.pinch_thumb_middle = float(pinch.get("thumb_middle", 0.35))
        self.pinch_release_hysteresis = float(pinch.get("release_hysteresis", 0.08))

        # Gesture timings & thresholds
        self.mode_toggle_hold_sec = float(gest.get("mode_toggle_hold_sec", 1.5))
        self.mode_toggle_grace_sec = float(gest.get("mode_toggle_grace_sec", 0.15))
        self.wrist_cross_palm_ratio = float(gest.get("wrist_cross_palm_ratio", 1.15))
        self.shift_cooldown_sec = float(gest.get("shift_cooldown_sec", 0.30))
        self.camera_hold_sec = float(gest.get("camera_hold_sec", 0.80))
        self.camera_grace_sec = float(gest.get("camera_grace_sec", 0.12))
        self.rewind_spread_ratio = float(gest.get("rewind_spread_ratio", 0.37))
        self.rewind_thumb_spread_ratio = float(gest.get("rewind_thumb_spread_ratio", 0.78))
        self.rewind_release_grace_sec = float(gest.get("rewind_release_grace_sec", 0.10))
        self.button_pulse_sec = float(gest.get("button_pulse_sec", 0.09))
        self.failsafe_grace_frames = int(gest.get("failsafe_grace_frames", 3))

        # Menu navigation config
        self.menu_box_cx = float(menu.get("box_center_x", 0.50))
        self.menu_box_cy = float(menu.get("box_center_y", 0.50))
        self.menu_box_hw = float(menu.get("box_half_width", 0.10))
        self.menu_box_hh = float(menu.get("box_half_height", 0.12))
        self.menu_box_hysteresis = float(menu.get("box_hysteresis", 0.015))
        self.menu_dpad_repeat_sec = float(menu.get("dpad_repeat_delay_sec", 0.32))
        self.menu_tab_cooldown_sec = float(menu.get("tab_cooldown_sec", 0.35))

        # Internal state
        self.mode: ControlMode = ControlMode.DRIVING
        self._last_update_ts: float | None = None
        self._smoothed_steering: float = 0.0
        self._last_raw_steering_angle: float = 0.0
        self._missing_steer_hand_frames: int = 0
        self._smoothed_throttle: float = 0.0
        self._smoothed_brake: float = 0.0
        self._no_hands_frames: int = 0

        # Mode toggle tracking
        self._cross_start_ts: float | None = None
        self._cross_last_seen_ts: float = 0.0
        self._mode_toggle_latched: bool = False

        # Timed button pulse expirations & held baseline
        self._pulses: dict[str, float] = {}
        self._last_held_buttons: dict[str, bool] = {name: False for name in ALL_BUTTONS}

        # Driving state trackers
        self._handbrake_latched: bool = False
        self._last_shift_up_ts: float = -10.0
        self._last_shift_down_ts: float = -10.0
        self._shift_up_latched: bool = False
        self._shift_down_latched: bool = False
        self._prev_right_index_ratio: float | None = None
        self._prev_left_index_ratio: float | None = None
        self._camera_start_ts: float | None = None
        self._camera_last_seen_ts: float = 0.0
        self._camera_latched: bool = False
        self._rewind_active: bool = False
        self._rewind_last_seen_ts: float = -10.0

        # Menu state trackers
        self._menu_select_latched: bool = False
        self._menu_back_latched: bool = False
        self._last_dpad_dir: str | None = None
        self._last_dpad_fire_ts: float = -10.0
        self._lb_latched: bool = False
        self._rb_latched: bool = False
        self._last_lb_ts: float = -10.0
        self._last_rb_ts: float = -10.0

    def reset_transient_state(self) -> None:
        """Reset smoothers, pulses, and gesture latches to neutral."""
        self._smoothed_steering = 0.0
        self._last_raw_steering_angle = 0.0
        self._missing_steer_hand_frames = 0
        self._smoothed_throttle = 0.0
        self._smoothed_brake = 0.0
        self._pulses.clear()
        for name in self._last_held_buttons:
            self._last_held_buttons[name] = False
        self._handbrake_latched = False
        self._shift_up_latched = False
        self._shift_down_latched = False
        self._prev_right_index_ratio = None
        self._prev_left_index_ratio = None
        self._camera_start_ts = None
        self._camera_last_seen_ts = 0.0
        self._camera_latched = False
        self._rewind_active = False
        self._rewind_last_seen_ts = -10.0
        self._menu_select_latched = False
        self._menu_back_latched = False
        self._last_dpad_dir = None
        self._lb_latched = False
        self._rb_latched = False
        for filt in self._landmark_filters.values():
            filt.reset()

    def _smooth_hand(self, hand: TrackedHand, now: float) -> TrackedHand:
        if not self.smoothing_enabled:
            return hand
        filt = self._landmark_filters.get(hand.handedness)
        if filt is None:
            return hand
        return TrackedHand(
            handedness=hand.handedness,
            score=hand.score,
            landmarks=filt.filter(hand.landmarks, now),
        )

    def _to_iso_xy(self, hand: TrackedHand) -> np.ndarray:
        """Scale normalized x by frame aspect ratio so angles & distances are isotropic."""
        pts = hand.landmarks[:, :2].copy()
        pts[:, 0] *= self.aspect_ratio
        return pts

    def _palm_scale(self, iso_pts: np.ndarray) -> float:
        return float(np.linalg.norm(iso_pts[MIDDLE_MCP] - iso_pts[WRIST]))

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _finger_extended(
        self, iso_pts: np.ndarray, mcp: int, pip: int, tip: int, palm_scale: float
    ) -> bool:
        wrist = iso_pts[WRIST]
        d_tip = self._dist(iso_pts[tip], wrist)
        d_pip = self._dist(iso_pts[pip], wrist)
        d_mcp = self._dist(iso_pts[mcp], wrist)
        u = iso_pts[mcp] - iso_pts[pip]
        v = iso_pts[tip] - iso_pts[pip]
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        cos_pip = float(np.dot(u, v) / (nu * nv)) if (nu > 1e-5 and nv > 1e-5) else 0.0
        return (
            d_tip > d_pip * 1.08
            and d_tip > d_mcp * 1.22
            and d_tip > palm_scale * 1.25
            and cos_pip < -0.45
        )

    def _finger_curled(
        self, iso_pts: np.ndarray, mcp: int, pip: int, tip: int, palm_scale: float
    ) -> bool:
        wrist = iso_pts[WRIST]
        d_tip = self._dist(iso_pts[tip], wrist)
        d_pip = self._dist(iso_pts[pip], wrist)
        d_mcp = self._dist(iso_pts[mcp], wrist)
        return d_tip <= d_pip * 1.06 or d_tip <= d_mcp * 1.15 or d_tip <= palm_scale * 1.15

    def _thumb_extended(self, iso_pts: np.ndarray, palm_scale: float) -> bool:
        d_tip_pinky = self._dist(iso_pts[THUMB_TIP], iso_pts[PINKY_MCP])
        d_tip_index = self._dist(iso_pts[THUMB_TIP], iso_pts[INDEX_MCP])
        return d_tip_pinky > palm_scale * 1.10 and d_tip_index > palm_scale * 0.50

    def _mean_tip_wrist_ratio(self, iso_pts: np.ndarray, palm_scale: float) -> float:
        """Return average fingertip-to-wrist distance normalized by palm scale (0->9)."""
        if palm_scale < 1e-4:
            return self.fist_open_ratio
        wrist = iso_pts[WRIST]
        return sum(self._dist(iso_pts[idx], wrist) for idx in FINGERTIP_INDICES) / (
            len(FINGERTIP_INDICES) * palm_scale
        )

    def _fist_closure_ratio(self, iso_pts: np.ndarray, palm_scale: float) -> float:
        """Map fingertip-to-wrist distance from 0.0 (open palm) to 1.0 (closed fist)."""
        if palm_scale < 1e-4:
            return 0.0
        mean_ratio = self._mean_tip_wrist_ratio(iso_pts, palm_scale)
        span = max(1e-4, self.fist_open_ratio - self.fist_closed_ratio)
        raw = _clamp((self.fist_open_ratio - mean_ratio) / span, 0.0, 1.0)
        if raw <= self.trigger_deadzone:
            return 0.0
        scaled = _clamp(
            (raw - self.trigger_deadzone) / max(1e-4, 1.0 - self.trigger_deadzone),
            0.0,
            1.0,
        )
        return _clamp(scaled ** self.trigger_curve_exponent, 0.0, 1.0)

    def _is_five_fingers_spread(self, iso_pts: np.ndarray, palm_scale: float) -> bool:
        """Return True when thumb + all 4 fingers are extended and spread wide."""
        if palm_scale < 1e-4:
            return False
        all_five_extended = (
            self._thumb_extended(iso_pts, palm_scale)
            and self._finger_extended(iso_pts, INDEX_MCP, INDEX_PIP, INDEX_TIP, palm_scale)
            and self._finger_extended(iso_pts, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, palm_scale)
            and self._finger_extended(iso_pts, RING_MCP, RING_PIP, RING_TIP, palm_scale)
            and self._finger_extended(iso_pts, PINKY_MCP, PINKY_PIP, PINKY_TIP, palm_scale)
        )
        if not all_five_extended:
            return False
        g_ti = self._dist(iso_pts[THUMB_TIP], iso_pts[INDEX_TIP]) / palm_scale
        g_im = self._dist(iso_pts[INDEX_TIP], iso_pts[MIDDLE_TIP]) / palm_scale
        g_mr = self._dist(iso_pts[MIDDLE_TIP], iso_pts[RING_TIP]) / palm_scale
        g_rp = self._dist(iso_pts[RING_TIP], iso_pts[PINKY_TIP]) / palm_scale
        return (
            g_ti >= self.rewind_thumb_spread_ratio
            and g_im >= self.rewind_spread_ratio
            and g_mr >= self.rewind_spread_ratio
            and g_rp >= self.rewind_spread_ratio
        )

    def _pinch_ratio(
        self, iso_pts: np.ndarray, tip_a: int, tip_b: int, palm_scale: float
    ) -> float:
        if palm_scale < 1e-4:
            return 999.0
        return self._dist(iso_pts[tip_a], iso_pts[tip_b]) / palm_scale

    def _trigger_pulse(self, button: str, now: float) -> None:
        self._pulses[button] = now + self.button_pulse_sec

    def _apply_active_pulses(self, buttons: dict[str, bool], now: float) -> None:
        expired = [btn for btn, until in self._pulses.items() if now >= until]
        for btn in expired:
            del self._pulses[btn]
        for btn in self._pulses:
            if btn in buttons:
                buttons[btn] = True

    def refresh_buttons(self, now: float | None = None) -> dict[str, bool]:
        """Compute current buttons between camera frames so timed pulses release on schedule."""
        ts = time.perf_counter() if now is None else float(now)
        buttons = dict(self._last_held_buttons)
        self._apply_active_pulses(buttons, ts)
        return buttons

    def _update_mode_toggle(
        self,
        left: TrackedHand | None,
        right: TrackedHand | None,
        left_iso: np.ndarray | None,
        right_iso: np.ndarray | None,
        left_palm: float,
        right_palm: float,
        now: float,
    ) -> tuple[bool, float, str | None]:
        """Check crossed-wrists hold duration to toggle DRIVING <-> MENU."""
        is_crossing = False
        if left is not None and right is not None and left_iso is not None and right_iso is not None:
            avg_palm = 0.5 * (left_palm + right_palm)
            if avg_palm > 1e-4:
                wrist_dist = self._dist(left_iso[WRIST], right_iso[WRIST])
                is_crossing = (wrist_dist / avg_palm) <= self.wrist_cross_palm_ratio

        if is_crossing:
            self._cross_last_seen_ts = now
            if self._cross_start_ts is None:
                self._cross_start_ts = now
        elif self._cross_start_ts is not None and (now - self._cross_last_seen_ts) > self.mode_toggle_grace_sec:
            self._cross_start_ts = None
            self._mode_toggle_latched = False

        if self._cross_start_ts is None or self._mode_toggle_latched:
            return is_crossing, 0.0, None

        elapsed = now - self._cross_start_ts
        progress = _clamp(elapsed / max(0.1, self.mode_toggle_hold_sec), 0.0, 1.0)
        if elapsed >= self.mode_toggle_hold_sec:
            self.mode = (
                ControlMode.MENU if self.mode == ControlMode.DRIVING else ControlMode.DRIVING
            )
            self.reset_transient_state()
            self._trigger_pulse("START", now)
            self._mode_toggle_latched = True
            return True, 1.0, f"MODE -> {self.mode.value} (START)"

        return is_crossing, progress, None

    def _compute_steering(
        self,
        left_iso: np.ndarray | None,
        right_iso: np.ndarray | None,
        suppress: bool,
        dt: float,
    ) -> tuple[float, float]:
        """Compute dt-normalized EMA-smoothed steering in [-1.0, 1.0] and raw angle."""
        alpha = _dt_ema_alpha(self.steering_ema_alpha, dt)
        if suppress:
            self._missing_steer_hand_frames = 0
            self._last_raw_steering_angle = 0.0
            self._smoothed_steering *= (1.0 - alpha)
            if abs(self._smoothed_steering) < 1e-3:
                self._smoothed_steering = 0.0
            return self._smoothed_steering, 0.0

        if left_iso is None or right_iso is None:
            # Hold brief steering grace if one hand momentarily flickers at frame edge.
            self._missing_steer_hand_frames += 1
            if self._missing_steer_hand_frames <= self.steering_single_hand_grace_frames:
                return self._smoothed_steering, self._last_raw_steering_angle
            self._last_raw_steering_angle = 0.0
            self._smoothed_steering *= (1.0 - alpha)
            if abs(self._smoothed_steering) < 1e-3:
                self._smoothed_steering = 0.0
            return self._smoothed_steering, 0.0

        self._missing_steer_hand_frames = 0
        lw = left_iso[WRIST]
        rw = right_iso[WRIST]
        dx_raw = float(rw[0] - lw[0]) if self.mirror else float(lw[0] - rw[0])
        dx = max(dx_raw, 1e-3)
        dy = float(rw[1] - lw[1])
        angle_deg = math.degrees(math.atan2(dy, dx))
        self._last_raw_steering_angle = angle_deg

        abs_angle = abs(angle_deg)
        if abs_angle <= self.steering_deadzone_deg:
            curved = 0.0
        else:
            effective_span = max(1e-3, self.steering_angle_max_deg - self.steering_deadzone_deg)
            linear_mag = _clamp((abs_angle - self.steering_deadzone_deg) / effective_span, 0.0, 1.0)
            curved = math.copysign(linear_mag ** self.steering_curve_exponent, angle_deg)

        self._smoothed_steering = _clamp(
            alpha * curved + (1.0 - alpha) * self._smoothed_steering,
            -1.0,
            1.0,
        )
        return self._smoothed_steering, angle_deg

    def _process_driving(
        self,
        left_iso: np.ndarray | None,
        right_iso: np.ndarray | None,
        left_palm: float,
        right_palm: float,
        is_crossing: bool,
        state: ControlState,
        now: float,
        dt: float,
    ) -> None:
        # 1. Steering (Left Stick X)
        steering, raw_angle = self._compute_steering(left_iso, right_iso, suppress=is_crossing, dt=dt)
        state.steering = steering
        state.raw_steering_angle_deg = raw_angle

        t_alpha = _dt_ema_alpha(self.trigger_ema_alpha, dt)
        if is_crossing:
            # Keep triggers neutral while crossing wrists to toggle mode.
            self._smoothed_throttle *= (1.0 - t_alpha)
            self._smoothed_brake *= (1.0 - t_alpha)
            state.throttle = self._smoothed_throttle
            state.brake = self._smoothed_brake
            return

        # 2. Throttle (RT - Right fist closure: 0% open palm -> 100% closed fist)
        raw_throttle = (
            self._fist_closure_ratio(right_iso, right_palm)
            if (right_iso is not None and right_palm > 1e-4)
            else 0.0
        )
        self._smoothed_throttle = _clamp(
            t_alpha * raw_throttle + (1.0 - t_alpha) * self._smoothed_throttle, 0.0, 1.0
        )
        if self._smoothed_throttle < 0.01:
            state.throttle = 0.0
        elif self._smoothed_throttle >= 0.985:
            state.throttle = 1.0
        else:
            state.throttle = self._smoothed_throttle

        # 3. Right Hand gestures (Handbrake A, Shift Up B, Change Camera BACK)
        if right_iso is not None and right_palm > 1e-4:
            r_idx_ext = self._finger_extended(right_iso, INDEX_MCP, INDEX_PIP, INDEX_TIP, right_palm)
            r_mid_ext = self._finger_extended(right_iso, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, right_palm)

            r_idx_curl = self._finger_curled(right_iso, INDEX_MCP, INDEX_PIP, INDEX_TIP, right_palm)
            r_mid_curl = self._finger_curled(right_iso, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, right_palm)
            r_ring_curl = self._finger_curled(right_iso, RING_MCP, RING_PIP, RING_TIP, right_palm)
            r_pnk_curl = self._finger_curled(right_iso, PINKY_MCP, PINKY_PIP, PINKY_TIP, right_palm)

            # Handbrake (A): Right Thumb tip + Right Middle tip pinch.
            # Prevent closed-fist throttle (RT) and Shift-Up poses from falsely triggering A.
            pinch_tm = self._pinch_ratio(right_iso, THUMB_TIP, MIDDLE_TIP, right_palm)
            shift_up_pose = r_idx_ext and r_mid_curl and r_ring_curl
            not_closed_fist = (not (r_idx_curl and r_ring_curl and r_pnk_curl)) and raw_throttle < 0.70
            pinch_allowed = not_closed_fist and not shift_up_pose
            if not self._handbrake_latched:
                if pinch_tm <= self.pinch_thumb_middle and pinch_allowed:
                    self._handbrake_latched = True
            else:
                if pinch_tm > (self.pinch_thumb_middle + self.pinch_release_hysteresis) or not pinch_allowed:
                    self._handbrake_latched = False
            state.buttons["A"] = self._handbrake_latched

            # Shift Up (B): Quick upward flick / extension of Right Index while middle/ring curled.
            r_idx_ratio = self._dist(right_iso[INDEX_TIP], right_iso[WRIST]) / right_palm
            flick_up = (
                self._prev_right_index_ratio is not None
                and (r_idx_ratio - self._prev_right_index_ratio) > 0.22
            )
            self._prev_right_index_ratio = r_idx_ratio

            if r_mid_curl and r_ring_curl and (r_idx_ext or flick_up):
                if not self._shift_up_latched and (now - self._last_shift_up_ts) >= self.shift_cooldown_sec:
                    self._trigger_pulse("B", now)
                    self._last_shift_up_ts = now
                    self._shift_up_latched = True
                    state.status_event = "SHIFT UP (B)"
            elif r_idx_curl or not (r_mid_curl and r_ring_curl):
                self._shift_up_latched = False

            # Change Camera (View/Back): Right Peace Sign (index + middle extended, ring + pinky curled) held 0.8s.
            is_peace = r_idx_ext and r_mid_ext and r_ring_curl and r_pnk_curl
            if is_peace:
                self._camera_last_seen_ts = now
                if self._camera_start_ts is None:
                    self._camera_start_ts = now
                if not self._camera_latched:
                    elapsed = now - self._camera_start_ts
                    state.camera_hold_progress = _clamp(
                        elapsed / max(0.1, self.camera_hold_sec), 0.0, 1.0
                    )
                    if elapsed >= self.camera_hold_sec:
                        self._trigger_pulse(self.camera_button, now)
                        self._camera_latched = True
                        state.camera_hold_progress = 1.0
                        state.status_event = f"CAMERA ({self.camera_button})"
            elif self._camera_start_ts is not None and (now - self._camera_last_seen_ts) <= self.camera_grace_sec:
                if not self._camera_latched:
                    elapsed = now - self._camera_start_ts
                    state.camera_hold_progress = _clamp(
                        elapsed / max(0.1, self.camera_hold_sec), 0.0, 1.0
                    )
            else:
                self._camera_start_ts = None
                self._camera_latched = False
        else:
            self._handbrake_latched = False
            self._shift_up_latched = False
            self._prev_right_index_ratio = None
            self._camera_start_ts = None
            self._camera_latched = False

        # 4. Left Hand gestures (Fist Brake LT, Shift Down X) + Two-hand Rewind (Y)
        if left_iso is not None and left_palm > 1e-4:
            l_idx_ext = self._finger_extended(left_iso, INDEX_MCP, INDEX_PIP, INDEX_TIP, left_palm)

            l_idx_curl = self._finger_curled(left_iso, INDEX_MCP, INDEX_PIP, INDEX_TIP, left_palm)
            l_mid_curl = self._finger_curled(left_iso, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, left_palm)
            l_ring_curl = self._finger_curled(left_iso, RING_MCP, RING_PIP, RING_TIP, left_palm)

            # Shift Down (X): Quick upward flick / extension of Left Index while middle/ring curled.
            l_idx_ratio = self._dist(left_iso[INDEX_TIP], left_iso[WRIST]) / left_palm
            flick_down_hand = (
                self._prev_left_index_ratio is not None
                and (l_idx_ratio - self._prev_left_index_ratio) > 0.22
            )
            self._prev_left_index_ratio = l_idx_ratio

            shift_down_pose = (l_mid_curl and l_ring_curl) and (
                l_idx_ext or flick_down_hand or self._shift_down_latched
            )
            if l_mid_curl and l_ring_curl and (l_idx_ext or flick_down_hand):
                if not self._shift_down_latched and (now - self._last_shift_down_ts) >= self.shift_cooldown_sec:
                    self._trigger_pulse("X", now)
                    self._last_shift_down_ts = now
                    self._shift_down_latched = True
                    state.status_event = "SHIFT DOWN (X)"
            elif l_idx_curl or not (l_mid_curl and l_ring_curl):
                self._shift_down_latched = False

            # Brake (LT): Left-hand fist closure (0% open palm -> 100% closed fist).
            # Guard: avoid spiking LT when curling middle/ring only to downshift (X) off-brake.
            if shift_down_pose:
                if self._smoothed_brake < 0.25:
                    raw_brake = 0.0
                    self._smoothed_brake = 0.0
                else:
                    raw_brake = self._smoothed_brake
            else:
                raw_brake = self._fist_closure_ratio(left_iso, left_palm)
                self._smoothed_brake = _clamp(
                    t_alpha * raw_brake + (1.0 - t_alpha) * self._smoothed_brake, 0.0, 1.0
                )

            if self._smoothed_brake < 0.01:
                state.brake = 0.0
            elif self._smoothed_brake >= 0.985:
                state.brake = 1.0
            else:
                state.brake = self._smoothed_brake

            # Rewind (Y): Trigger ONLY when BOTH hands are open with all 5 fingers spread wide.
            both_spread = (
                right_iso is not None
                and right_palm > 1e-4
                and self._is_five_fingers_spread(left_iso, left_palm)
                and self._is_five_fingers_spread(right_iso, right_palm)
            )
            if both_spread:
                self._rewind_active = True
                self._rewind_last_seen_ts = now
            elif self._rewind_active and (now - self._rewind_last_seen_ts) > self.rewind_release_grace_sec:
                self._rewind_active = False
            state.buttons["Y"] = self._rewind_active
        else:
            self._smoothed_brake = 0.0
            state.brake = 0.0
            self._shift_down_latched = False
            self._prev_left_index_ratio = None
            self._rewind_active = False

    def _process_menu(
        self,
        right: TrackedHand | None,
        left_iso: np.ndarray | None,
        right_iso: np.ndarray | None,
        left_palm: float,
        right_palm: float,
        is_crossing: bool,
        state: ControlState,
        now: float,
    ) -> None:
        # Keep analog driving channels neutral in MENU mode.
        self._smoothed_steering = 0.0
        self._smoothed_throttle = 0.0
        self._smoothed_brake = 0.0
        state.steering = 0.0
        state.raw_steering_angle_deg = 0.0
        state.throttle = 0.0
        state.brake = 0.0

        if is_crossing:
            return

        # Right hand: Select (A), Back (B), and D-Pad navigation
        if right is not None and right_iso is not None and right_palm > 1e-4:
            pinch_ti = self._pinch_ratio(right_iso, THUMB_TIP, INDEX_TIP, right_palm)
            pinch_tm = self._pinch_ratio(right_iso, THUMB_TIP, MIDDLE_TIP, right_palm)

            # Disambiguate if both thumb-index and thumb-middle are close.
            want_select = (
                pinch_ti <= (
                    self.pinch_thumb_index + self.pinch_release_hysteresis
                    if self._menu_select_latched
                    else self.pinch_thumb_index
                )
            )
            want_back = (
                pinch_tm <= (
                    self.pinch_thumb_middle + self.pinch_release_hysteresis
                    if self._menu_back_latched
                    else self.pinch_thumb_middle
                )
            )
            if want_select and want_back:
                if pinch_ti <= pinch_tm:
                    want_back = False
                else:
                    want_select = False

            self._menu_select_latched = want_select
            self._menu_back_latched = want_back
            state.buttons["A"] = want_select
            state.buttons["B"] = want_back

            # Virtual center box D-Pad navigation via Right Index fingertip (normalized frame coords)
            tip_x = float(right.landmarks[INDEX_TIP, 0])
            tip_y = float(right.landmarks[INDEX_TIP, 1])
            state.menu_cursor = (tip_x, tip_y)

            # Suppress D-Pad while pinching A/B so finger curl does not jump menu selection.
            if not (want_select or want_back):
                hyst = self.menu_box_hysteresis if self._last_dpad_dir is not None else 0.0
                hw = max(1e-4, self.menu_box_hw - hyst)
                hh = max(1e-4, self.menu_box_hh - hyst)
                dx_left = (self.menu_box_cx - hw) - tip_x
                dx_right = tip_x - (self.menu_box_cx + hw)
                dy_up = (self.menu_box_cy - hh) - tip_y
                dy_down = tip_y - (self.menu_box_cy + hh)

                candidates: list[tuple[float, str]] = []
                if dx_left > 0.0:
                    candidates.append((dx_left / hw, "DPAD_LEFT"))
                if dx_right > 0.0:
                    candidates.append((dx_right / hw, "DPAD_RIGHT"))
                if dy_up > 0.0:
                    candidates.append((dy_up / hh, "DPAD_UP"))
                if dy_down > 0.0:
                    candidates.append((dy_down / hh, "DPAD_DOWN"))

                if candidates:
                    _, dpad_dir = max(candidates, key=lambda item: item[0])
                    state.active_dpad_zone = dpad_dir
                    if dpad_dir != self._last_dpad_dir or (now - self._last_dpad_fire_ts) >= self.menu_dpad_repeat_sec:
                        self._trigger_pulse(dpad_dir, now)
                        self._last_dpad_dir = dpad_dir
                        self._last_dpad_fire_ts = now
                else:
                    self._last_dpad_dir = None
            else:
                self._last_dpad_dir = None
        else:
            self._menu_select_latched = False
            self._menu_back_latched = False
            self._last_dpad_dir = None

        # Left hand: Tab Left (LB = Left index extended) / Tab Right (RB = Left pinky extended)
        if left_iso is not None and left_palm > 1e-4:
            l_idx_ext = self._finger_extended(left_iso, INDEX_MCP, INDEX_PIP, INDEX_TIP, left_palm)
            l_mid_ext = self._finger_extended(left_iso, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, left_palm)
            l_ring_ext = self._finger_extended(left_iso, RING_MCP, RING_PIP, RING_TIP, left_palm)
            l_pnk_ext = self._finger_extended(left_iso, PINKY_MCP, PINKY_PIP, PINKY_TIP, left_palm)

            l_idx_curl = self._finger_curled(left_iso, INDEX_MCP, INDEX_PIP, INDEX_TIP, left_palm)
            l_mid_curl = self._finger_curled(left_iso, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, left_palm)
            l_ring_curl = self._finger_curled(left_iso, RING_MCP, RING_PIP, RING_TIP, left_palm)
            l_pnk_curl = self._finger_curled(left_iso, PINKY_MCP, PINKY_PIP, PINKY_TIP, left_palm)

            want_lb = l_idx_ext and l_mid_curl and l_ring_curl and l_pnk_curl
            want_rb = l_pnk_ext and l_idx_curl and l_mid_curl and l_ring_curl

            if want_lb:
                if not self._lb_latched and (now - self._last_lb_ts) >= self.menu_tab_cooldown_sec:
                    self._trigger_pulse("LB", now)
                    self._last_lb_ts = now
                    self._lb_latched = True
                    state.status_event = "TAB LEFT (LB)"
            elif not l_idx_ext or l_mid_ext or l_ring_ext:
                self._lb_latched = False

            if want_rb:
                if not self._rb_latched and (now - self._last_rb_ts) >= self.menu_tab_cooldown_sec:
                    self._trigger_pulse("RB", now)
                    self._last_rb_ts = now
                    self._rb_latched = True
                    state.status_event = "TAB RIGHT (RB)"
            elif not l_pnk_ext or l_mid_ext or l_ring_ext:
                self._rb_latched = False
        else:
            self._lb_latched = False
            self._rb_latched = False

    def update(
        self,
        hands: dict[HandLabel, TrackedHand],
        now: float | None = None,
    ) -> ControlState:
        """Evaluate detected hands and return the current ControlState."""
        ts = time.perf_counter() if now is None else float(now)
        dt = (
            1.0 / 60.0
            if self._last_update_ts is None or ts <= self._last_update_ts
            else float(ts - self._last_update_ts)
        )
        self._last_update_ts = ts

        raw_left = hands.get("Left")
        raw_right = hands.get("Right")
        left = self._smooth_hand(raw_left, ts) if raw_left is not None else None
        right = self._smooth_hand(raw_right, ts) if raw_right is not None else None

        state = ControlState(
            mode=self.mode,
            hands_present=len(hands),
            left_present=left is not None,
            right_present=right is not None,
        )

        # Failsafe if hands leave the frame
        if not hands:
            self._no_hands_frames += 1
            if self._no_hands_frames >= self.failsafe_grace_frames:
                self.reset_transient_state()
                self._cross_start_ts = None
                self._mode_toggle_latched = False
                state.failsafe_active = True
                state.status_event = "FAILSAFE: NEUTRAL"
                return state
        else:
            self._no_hands_frames = 0

        left_iso = self._to_iso_xy(left) if left is not None else None
        right_iso = self._to_iso_xy(right) if right is not None else None
        left_palm = self._palm_scale(left_iso) if left_iso is not None else 0.0
        right_palm = self._palm_scale(right_iso) if right_iso is not None else 0.0

        is_crossing, cross_progress, toggle_event = self._update_mode_toggle(
            left=left,
            right=right,
            left_iso=left_iso,
            right_iso=right_iso,
            left_palm=left_palm,
            right_palm=right_palm,
            now=ts,
        )
        state.mode = self.mode
        state.wrist_cross_progress = cross_progress
        if toggle_event is not None:
            state.status_event = toggle_event

        if self.mode == ControlMode.DRIVING:
            self._process_driving(
                left_iso=left_iso,
                right_iso=right_iso,
                left_palm=left_palm,
                right_palm=right_palm,
                is_crossing=is_crossing,
                state=state,
                now=ts,
                dt=dt,
            )
        else:
            self._process_menu(
                right=right,
                left_iso=left_iso,
                right_iso=right_iso,
                left_palm=left_palm,
                right_palm=right_palm,
                is_crossing=is_crossing,
                state=state,
                now=ts,
            )

        self._last_held_buttons = dict(state.buttons)
        self._apply_active_pulses(state.buttons, ts)
        return state
