"""Xbox 360 virtual controller wrapper built around ``vgamepad.VX360Gamepad``.

Translates normalized ``ControlState`` values into native Windows XInput commands,
handles failsafe neutral resets, supports ``--debug`` dry runs, and surfaces clear
diagnostic errors when the ViGEmBus kernel driver is missing.
"""

from __future__ import annotations

import logging
from typing import Any

from gestures import ALL_BUTTONS, ControlState

logger = logging.getLogger(__name__)


class ViGEmDriverError(RuntimeError):
    """Raised when the ViGEmBus virtual gamepad driver is missing or unavailable."""


class XboxController:
    """Wrapper around ``vgamepad.VX360Gamepad`` with failsafe & debug support."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = bool(debug)
        self._gamepad: Any = None
        self._button_map: dict[str, Any] = {}
        self._pressed_buttons: dict[str, bool] = {name: False for name in ALL_BUTTONS}
        self.last_state: ControlState | None = None

        if not self.debug:
            self._initialize_gamepad()

    def _initialize_gamepad(self) -> None:
        try:
            import vgamepad as vg

            self._button_map = {
                "A": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
                "B": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                "X": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
                "Y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
                "LB": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
                "RB": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
                "START": vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
                "BACK": vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
                "DPAD_UP": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
                "DPAD_DOWN": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
                "DPAD_LEFT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
                "DPAD_RIGHT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
            }

            self._gamepad = vg.VX360Gamepad()
            self._gamepad.reset()
            self._gamepad.update()
        except ImportError as exc:
            msg = (
                "Python package 'vgamepad' is not installed in the active environment. "
                "Install dependencies with: pip install -r requirements.txt"
            )
            logger.error(msg)
            raise ViGEmDriverError(msg) from exc
        except Exception as exc:
            msg = (
                "WARNING: ViGEmBus virtual controller driver was not found or failed to initialize.\n"
                f"Underlying error: {exc}\n\n"
                "To fix this on Windows:\n"
                "  1. Download and install the ViGEmBus driver (ViGEmBus_1.22.0_x64_x86_arm64.exe) "
                "from https://github.com/nefarius/ViGEmBus/releases\n"
                "  2. Reboot Windows if prompted by the installer.\n"
                "  3. Or run with '--debug' to test webcam tracking and gestures without ViGEmBus."
            )
            logger.warning("ViGEmBus driver missing or unavailable: %s", exc)
            raise ViGEmDriverError(msg) from exc

    @property
    def is_connected(self) -> bool:
        """Return True when the native ViGEm virtual Xbox 360 device is active."""
        return self._gamepad is not None

    def apply(self, state: ControlState) -> None:
        """Send a ``ControlState`` snapshot to the virtual Xbox 360 controller."""
        self.last_state = state
        if self._gamepad is None:
            return

        if state.failsafe_active:
            self.reset_to_neutral()
            return

        steering = max(-1.0, min(1.0, float(state.steering)))
        throttle = max(0.0, min(1.0, float(state.throttle)))
        brake = max(0.0, min(1.0, float(state.brake)))

        self._gamepad.left_joystick_float(x_value_float=steering, y_value_float=0.0)
        self._gamepad.right_trigger_float(value_float=throttle)
        self._gamepad.left_trigger_float(value_float=brake)

        for name, xusb_btn in self._button_map.items():
            want_pressed = bool(state.buttons.get(name, False))
            was_pressed = self._pressed_buttons.get(name, False)
            if want_pressed and not was_pressed:
                self._gamepad.press_button(button=xusb_btn)
            elif not want_pressed and was_pressed:
                self._gamepad.release_button(button=xusb_btn)
            self._pressed_buttons[name] = want_pressed

        self._gamepad.update()

    def reset_to_neutral(self) -> None:
        """Reset all axes, triggers, and buttons to safe neutral defaults."""
        for name in self._pressed_buttons:
            self._pressed_buttons[name] = False
        if self._gamepad is not None:
            self._gamepad.reset()
            self._gamepad.update()

    def close(self) -> None:
        """Neutralize controller outputs and release the virtual device handle."""
        try:
            self.reset_to_neutral()
        finally:
            self._gamepad = None

    def __enter__(self) -> XboxController:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
