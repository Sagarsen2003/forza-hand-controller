# Forza Series — Webcam Hand-Tracking Xbox 360 Controller

A modular, low-latency Python system that uses **OpenCV**, **MediaPipe**, and **vgamepad (ViGEmBus)** to translate two-hand webcam gestures into a native virtual **Xbox 360 controller** for **Forza Horizon 4**, **Forza Horizon 5**, **Forza Horizon 6**, and **Forza Motorsport** on Windows (Steam & Xbox App).

---

## Project Structure

```text
forza_hand_controller/
├── .venv/                  # Isolated Python virtual environment
├── models/
│   └── hand_landmarker.task# Offline MediaPipe HandLandmarker model (auto-downloaded if absent)
├── config.json             # Tunable camera, steering curve, EMA, pinch & timing parameters
├── tracker.py              # Low-latency OpenCV capture + MediaPipe 3D landmarks (Left/Right)
├── gestures.py             # DRIVING & MENU state machine, steering curve, EMA, cooldowns
├── controller.py           # vgamepad.VX360Gamepad wrapper with neutral failsafe & ViGEm checks
├── hud.py                  # High-contrast OpenCV telemetry overlay (mode, gauge, RT/LT, FPS)
├── main.py                 # Entry point (--debug and --headless flags, clean exit)
├── requirements.txt        # Python dependencies
└── README.md               # Setup, Steam Input guide, calibration, and FH4 deadzone tuning
```

---

## 1. Installing the ViGEmBus Driver on Windows

`vgamepad` requires the Windows kernel-mode **ViGEmBus** driver to emulate a hardware Xbox 360 controller.

1. Download the latest stable **ViGEmBus** installer from the official GitHub releases page:
   `https://github.com/nefarius/ViGEmBus/releases`
   *(Download `ViGEmBus_1.22.0_x64_x86_arm64.exe` or latest signed release.)*
2. Run the installer as Administrator and complete the setup wizard.
3. **Reboot Windows** if prompted so the kernel driver loads cleanly.
4. *(Optional)* Verify in Device Manager under **System devices** that **Nefarius Virtual Gamepad Emulation Bus** is present and enabled.

> **Note:** If ViGEmBus is not installed, `main.py` prints a clear diagnostic warning. You can still test webcam tracking, gestures, and the HUD at any time using `--debug`.

---

## 2. Quickstart & Launch Modes

From the `forza_hand_controller` directory, set up the virtual environment once:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
*(MediaPipe's `hand_landmarker.task` downloads automatically to `models/` on first launch if absent.)*

### A. Debug Mode (Test Camera & Gestures Without ViGEmBus)
```powershell
.\.venv\Scripts\python.exe main.py --debug
```
Displays the live OpenCV HUD with steering gauge, throttle/brake bars, active button triggers, and FPS without creating a virtual Xbox 360 device. Press **`q`** to quit cleanly.

### B. Calibrated Live Mode (With HUD Preview)
```powershell
.\.venv\Scripts\python.exe main.py
```

### C. Headless Mode for Steam / Xbox App (Recommended for Playing)
Once calibrated, launch with `--headless` and your `--game` profile **before** opening the game:
```powershell
# Forza Horizon 4 (default)
.\.venv\Scripts\python.exe main.py --headless

# Forza Horizon 5
.\.venv\Scripts\python.exe main.py --game fh5 --headless

# Forza Horizon 6
.\.venv\Scripts\python.exe main.py --game fh6 --headless

# Forza Motorsport
.\.venv\Scripts\python.exe main.py --game motorsport --headless
```
- **Why launch first?** Starting `main.py` before opening Forza ensures `vgamepad.VX360Gamepad()` enumerates in Windows and Steam prior to game initialization.
- **Why `--headless`?** Disabling the OpenCV preview window eliminates window-focus stealing and avoids conflicts with the Steam Overlay. Press **`q`** in the terminal or **`Ctrl+C`** to neutralize inputs and exit cleanly.

### Useful CLI Flags
- `--game {fh4,fh5,fh6,motorsport}`: Select game profile (`fh4` maps Change Camera to `BACK`; `fh5`, `fh6`, and `motorsport` map Change Camera to `RB`, and `motorsport` adds crisper track spring response).
- `--config path/to/config.json`: Load a custom configuration file.
- `--camera 1`: Override `camera.index` (useful if you have multiple webcams).
- `--debug`: Disable `vgamepad` while testing tracking, gestures, and HUD.
- `--headless`: Disable the OpenCV GUI window once calibrated.
- `--no-landmarks`: Hide the 21-point hand skeleton overlay in preview mode.

---

## 3. Verifying Controller Inputs with Windows `joy.cpl`

Before launching Forza Horizon 4, confirm Windows sees every axis and button:

1. Start the controller (`.\.venv\Scripts\python.exe main.py`).
2. Press **`Win + R`**, type **`joy.cpl`**, and press **Enter**.
3. Select **Controller (Xbox 360 For Windows)** and click **Properties** → **Test**.
4. Verify:
   - Tilting both hands left/right moves the **X Axis** crosshairs smoothly (`-45°` to `+45°`).
   - Closing your Right fist moves the **Z Axis / Right Trigger** bar cleanly; closing your Left fist moves the **Left Trigger** bar.
   - Pinching, shifting, camera-holding, rewinding, and crossing wrists light up buttons **`1`–`8`** and the **Point of View Hat (D-Pad)** in `MENU` mode.

---

## 4. Steam Version Compatibility (Disabling Steam Input)

Steam Input can intercept, remap, or override virtual ViGEm Xbox 360 controllers in **Forza Horizon 4, 5, 6, and Forza Motorsport** on Steam.

Follow these steps in the **Steam Desktop Client**:

1. Open **Steam** and go to your **Library**.
2. Right-click your Forza title (**Forza Horizon 4**, **5**, **6**, or **Forza Motorsport**) and select **Properties...**
3. In the left sidebar, click **Controller**.
4. Change **"Override for \<Game\>"** from *Use default settings* to **Disable Steam Input**.
5. Close Properties.
6. Verify the launch order every session:
   1. Run `.\.venv\Scripts\python.exe main.py --game <profile> --headless` (wait for `[Controller] Virtual Xbox 360 controller active`).
   2. Launch the game from Steam (or Xbox App).

---

## 5. In-Game Deadzone & Steering Setup (FH4 / FH5 / FH6 / Motorsport)

Because `gestures.py` already applies a `±5°` center deadzone, gentle steering curve (`1.2` exponent), and adaptive smoothing, stacking the game's default controller deadzones on top would create sluggish center response.

In-game: **Settings → Controls → Controller → Advanced** (or **Advanced Input** in Forza Motorsport):

| Setting | Recommended Value | Why |
| :--- | :---: | :--- |
| **Steering Linearity** | `50` | Keeps Forza's curve linear so `gestures.py` handles the precision curve cleanly. |
| **Steering Axis Deadzone Inside** | `0` – `3` | Prevents double-deadzone; `config.json` already provides a crisp `±5.0°` deadzone. |
| **Steering Axis Deadzone Outside** | `100` | Allows full steering lock at `±45°` hand tilt. |
| **Acceleration Axis Deadzone Inside** | `0` – `5` | `config.json` already filters open-palm noise (`trigger_deadzone: 0.06`). |
| **Acceleration Axis Deadzone Outside** | `95` – `100` | Ensures `100%` throttle is easy to reach with a comfortable fist. |
| **Deceleration Axis Deadzone Inside** | `0` – `5` | Prevents brake drag while coasting. |
| **Deceleration Axis Deadzone Outside** | `95` – `100` | Ensures full braking lock cleanly. |
| **Vibration** | `Off` | Virtual webcam hands don't receive rumble feedback. |
| **Difficulty → Shifting** | `Manual` (optional) | Lets you use the Shift Up (`B`) and Shift Down (`X`) index flick gestures! |

---

## 6. Gesture Reference

### Mode Toggle (Active in Both Modes)
| Gesture | Action |
| :--- | :--- |
| **Cross both wrists** closely and hold for **1.5s** | Toggles `DRIVING ↔ MENU` and triggers Xbox **Start / Menu** |

### DRIVING Mode
| Hand / Gesture | Xbox 360 Input | Forza Action |
| :--- | :--- | :--- |
| Tilt both wrists (`-45°` to `+45°`) | **Left Stick X** | Steering (with `±5°` deadzone, curve & EMA) |
| Right-hand fist closure (`0.0` → `1.0`) | **Right Trigger (RT)** | Progressive Throttle / Gas |
| Left-hand fist closure (`0.0` open palm → `1.0` closed fist) | **Left Trigger (LT)** | Progressive Brake / Reverse |
| Pinch Right Thumb + Right Middle tip | **`A`** | Handbrake |
| Quick Right Index flick / extend *(middle & ring curled)* | **`B`** *(300ms cooldown)* | Shift Up |
| Quick Left Index flick / extend *(middle & ring curled)* | **`X`** *(300ms cooldown)* | Shift Down |
| **Both hands** open + all 5 fingers spread wide | **`Y`** | Rewind *(never fires on left-hand brake)* |
| Right Peace Sign (index + middle up, ring + pinky curled) held **0.8s** | **`BACK`** (`fh4`) / **`RB`** (`fh5`, `fh6`, `motorsport`) | Change Camera |

### MENU Mode
| Hand / Gesture | Xbox 360 Input | Forza Menu Action |
| :--- | :--- | :--- |
| Move Right Index tip outside the virtual center box | **D-Pad Up / Down / Left / Right** | Menu Navigation (with repeat delay) |
| Pinch Right Thumb + Right Index tip | **`A`** | Select / Confirm |
| Pinch Right Thumb + Right Middle tip | **`B`** | Back / Cancel |
| Extend Left Index finger *(others curled)* | **`LB`** | Previous Tab |
| Extend Left Pinky finger *(others curled)* | **`RB`** | Next Tab |

> **Failsafe:** If both hands leave the webcam frame, all axes, triggers, and buttons immediately reset to neutral (`0.0` / released).

---

## 7. Calibrating `config.json`

Edit `config.json` to match your webcam, seating distance, and driving feel:

- **`camera.index` (`0`)**: Change to `1` or `2` if the wrong camera opens.
- **`camera.mirror` (`true`)**: Keep `true` for selfie-view webcams. If your physical left/right hands appear swapped in the HUD, toggle `camera.swap_handedness` to `true`.
- **`smoothing.landmark_min_cutoff` (`1.4`) / `landmark_beta` (`7.5`)**: Controls adaptive One-Euro landmark filtering across all controls. Lower `landmark_min_cutoff` (`1.1`) for steadier straight-line driving and menu hovering; raise `landmark_beta` (`10.0`) if fast snaps or index flicks feel slightly damped.
- **`smoothing.control_loop_hz` (`120`) / `steering_spring_hz` (`4.8`) / `trigger_spring_hz` (`6.5`)**: Runs `vgamepad` XInput updates at `120 Hz` using critically-damped `C¹` spring interpolation between camera frames. Lower `steering_spring_hz` (`3.8`) for silkier gradual curves or raise (`6.0`) for crisper turn-in.
- **`steering.angle_max_deg` (`45.0`)**: Angle in degrees that produces `100%` steering lock. Lower (`32.0`) for sharper turns with smaller hand tilts, or keep `45.0` for full wheel rotation.
- **`steering.center_deadzone_deg` (`5.0`)**: Increase to `7.0` if your car wobbles on straightaways.
- **`steering.curve_exponent` (`1.20`)**: `1.0` = linear (maximum mid-corner turn strength); `1.3`–`1.8` = softer center for high-speed stability.
- **`steering.ema_alpha` (`0.45`)**: Lower (`0.25`) for smoother/heavier steering; higher (`0.60`) for snappier turn-in.
- **`triggers.fist_open_ratio` (`1.65`) / `fist_closed_ratio` (`1.20`)**: Watch the `LT`/`RT` bars in `--debug` mode. If an open palm leaks above `0%`, lower `fist_open_ratio` (e.g., `1.55`) or raise `trigger_deadzone` (`0.08`). If a comfortable closed fist (`RT` gas or `LT` brake) drops below `100%`, raise `fist_closed_ratio` (e.g., `1.28`).
- **`triggers.ema_alpha` (`0.65`) / `curve_exponent` (`0.85`)**: Raise `ema_alpha` (`0.80`) for even faster pedal snap; lower `curve_exponent` (`0.70`) for stronger throttle/brake on partial squeezes.
- **`pinch_thresholds.thumb_index` / `thumb_middle` (`0.35`)**: Raise (`0.42`) if pinches miss, or lower (`0.28`) if pinches trigger accidentally.
- **`gestures.wrist_cross_palm_ratio` (`1.15`)**: Raise (`1.35`) if crossing wrists struggles to start the `1.5s` mode-toggle timer.
- **`menu.box_half_width` (`0.10`) / `box_half_height` (`0.12`)**: Adjust the deadzone size of the virtual D-Pad center box in `MENU` mode.
