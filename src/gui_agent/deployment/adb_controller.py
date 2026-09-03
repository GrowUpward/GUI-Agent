import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import socket
import urllib.error
import urllib.request
from datetime import datetime


def get_png_size(image_path):
    with open(image_path, "rb") as f:
        header = f.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
    return None, None


def write_debug_html(image_path, prediction):
    coordinate = prediction.get("coordinate")
    if not coordinate or len(coordinate) != 2:
        return None

    width, height = get_png_size(image_path)
    if not width or not height:
        return None

    x, y = map(int, coordinate)
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    html_path = os.path.splitext(image_path)[0] + "_prediction.html"
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>GUI-Agent Prediction</title>
<style>
body {{ margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }}
.wrap {{ position: relative; display: inline-block; }}
img {{ display: block; width: {width}px; height: {height}px; }}
.point {{
  position: absolute;
  left: {x}px;
  top: {y}px;
  width: 22px;
  height: 22px;
  margin-left: -11px;
  margin-top: -11px;
  border: 3px solid #ff2d2d;
  border-radius: 999px;
  box-sizing: border-box;
  background: rgba(255, 45, 45, 0.25);
}}
.label {{
  position: absolute;
  left: {x + 14}px;
  top: {y + 14}px;
  background: rgba(0, 0, 0, 0.8);
  color: #fff;
  padding: 4px 6px;
  font-size: 14px;
}}
pre {{ max-width: {width}px; white-space: pre-wrap; padding: 12px; }}
</style>
</head>
<body>
<div class="wrap">
  <img src="data:image/png;base64,{image_base64}">
  <div class="point"></div>
  <div class="label">({x}, {y})</div>
</div>
<pre>{json.dumps(prediction, ensure_ascii=False, indent=2)}</pre>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def next_screenshot_path(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(args.screenshot_dir, f"screen_{timestamp}.png")


def resolve_adb(adb):
    if os.path.isabs(adb) or os.path.sep in adb:
        if os.path.exists(adb) and os.access(adb, os.X_OK):
            return adb
        raise FileNotFoundError(f"ADB executable is not usable: {adb}")

    resolved = shutil.which(adb)
    if resolved:
        return resolved

    raise FileNotFoundError(
        "Cannot find 'adb'. Install Android platform-tools and add it to PATH, "
        "or pass the executable path with --adb /path/to/adb. If the emulator "
        "runs on your PC, run this controller on that PC and connect it to the "
        "remote model server with --server_url http://SERVER_IP:8000 --send_base64."
    )


def run_adb(args, adb_args, capture_output=False):
    cmd = [args.adb]
    if args.serial:
        cmd.extend(["-s", args.serial])
    cmd.extend(adb_args)
    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )


def run_adb_result(args, adb_args, capture_output=False):
    cmd = [args.adb]
    if args.serial:
        cmd.extend(["-s", args.serial])
    cmd.extend(adb_args)
    return subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )


def ensure_device_connected(args):
    result = run_adb_result(args, ["devices"], capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        raise RuntimeError(f"failed to run adb devices: {stderr}")

    stdout = result.stdout.decode("utf-8", errors="replace")
    devices = []
    for line in stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            devices.append((parts[0], parts[1]))

    online_devices = [serial for serial, state in devices if state == "device"]
    if args.serial:
        matched = [state for serial, state in devices if serial == args.serial]
        if matched == ["device"]:
            return
        raise RuntimeError(
            f"ADB device '{args.serial}' is not connected as 'device'. adb devices output:\n{stdout}"
        )

    if len(online_devices) == 1:
        args.serial = online_devices[0]
        return

    if not online_devices:
        raise RuntimeError(
            "No online ADB device found. Start MuMu, enable/allow ADB connection, then run "
            "`adb connect 127.0.0.1:PORT` and verify `adb devices` shows a device.\n"
            f"adb devices output:\n{stdout}"
        )

    raise RuntimeError(
        "Multiple online ADB devices found. Pass one with --serial.\n"
        f"adb devices output:\n{stdout}"
    )


def capture_screenshot(args, image_path):
    result = run_adb_result(args, ["exec-out", "screencap", "-p"], capture_output=True)
    if result.returncode == 0 and result.stdout:
        with open(image_path, "wb") as f:
            f.write(result.stdout)
        return

    remote_path = "/sdcard/gui_agent_current_screen.png"
    run_adb(args, ["shell", "screencap", "-p", remote_path])
    run_adb(args, ["pull", remote_path, image_path], capture_output=True)
    run_adb_result(args, ["shell", "rm", remote_path], capture_output=True)


def request_prediction(args, image_path, instruction):
    if args.send_base64:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {"instruction": instruction, "image_base64": image_base64}
    else:
        payload = {"instruction": instruction, "image_path": os.path.abspath(image_path)}

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.server_url.rstrip("/") + "/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model server HTTP {exc.code}: {detail}") from exc
    except socket.timeout as exc:
        raise TimeoutError(
            f"Timed out waiting for {args.server_url.rstrip()}/predict after {args.timeout}s. "
            "Check whether /health is reachable from this PC and whether the server printed "
            "a '[predict] start' log. If it did, increase --timeout or reduce screenshot size."
        ) from exc


def execute_action(args, prediction):
    action = prediction.get("action")
    coordinate = prediction.get("coordinate")

    if action == "click":
        if not coordinate or len(coordinate) != 2:
            raise ValueError(f"click action missing coordinate: {prediction}")
        x, y = map(int, coordinate)
        run_adb(args, ["shell", "input", "tap", str(x), str(y)])
        return f"tap {x} {y}"

    if action == "scroll":
        width, height = prediction.get("image_size", [1000, 2000])
        direction = prediction.get("direction", "up")
        points = {
            "up": (width // 2, 3 * height // 4, width // 2, height // 4),
            "down": (width // 2, height // 4, width // 2, 3 * height // 4),
            "left": (3 * width // 4, height // 2, width // 4, height // 2),
            "right": (width // 4, height // 2, 3 * width // 4, height // 2),
        }
        x1, y1, x2, y2 = points.get(direction, points["up"])
        run_adb(
            args,
            [
                "shell",
                "input",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(args.scroll_duration_ms),
            ],
        )
        return f"swipe {x1} {y1} {x2} {y2} {args.scroll_duration_ms}"

    if action == "input_text":
        text = str(prediction.get("text", ""))
        if not text:
            raise ValueError(f"input_text action missing text: {prediction}")
        adb_text = text.replace(" ", "%s")
        run_adb(args, ["shell", "input", "text", adb_text])
        return f"input text ({len(text)} chars)"

    keycodes = {"press_back": "4", "press_home": "3", "press_enter": "66"}
    if action in keycodes:
        run_adb(args, ["shell", "input", "keyevent", keycodes[action]])
        return f"keyevent {keycodes[action]}"

    if action == "stop":
        return "stop"

    raise ValueError(f"unsupported or empty action: {action}. Full prediction: {prediction}")


def ask_confirmation(prediction):
    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    answer = input("Execute this action? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def interactive_loop(args):
    os.makedirs(args.screenshot_dir, exist_ok=True)
    print("Type an instruction, or type ':q' to quit.")
    while True:
        instruction = input("\nInstruction> ").strip()
        if instruction in {":q", "q", "quit", "exit"}:
            return
        if not instruction:
            continue

        image_path = next_screenshot_path(args)
        capture_screenshot(args, image_path)
        print(f"Saved screenshot sent to model: {image_path}")
        prediction = request_prediction(args, image_path, instruction)
        debug_html = write_debug_html(image_path, prediction)
        if debug_html:
            print(f"Prediction overlay: {debug_html}")

        if args.confirm and not ask_confirmation(prediction):
            print("Skipped.")
            continue

        executed = execute_action(args, prediction)
        print(f"Executed: {executed}")
        time.sleep(args.wait_after_action)


def single_step(args):
    os.makedirs(args.screenshot_dir, exist_ok=True)
    image_path = next_screenshot_path(args)
    capture_screenshot(args, image_path)
    print(f"Saved screenshot sent to model: {image_path}")
    prediction = request_prediction(args, image_path, args.instruction)
    debug_html = write_debug_html(image_path, prediction)
    if debug_html:
        print(f"Prediction overlay: {debug_html}")
    if args.confirm and not ask_confirmation(prediction):
        print("Skipped.")
        return
    executed = execute_action(args, prediction)
    print(json.dumps({"prediction": prediction, "executed": executed}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Capture Android screenshots, ask GUI-Agent, and execute ADB actions.")
    parser.add_argument("--server_url", default="http://127.0.0.1:8000")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", default=None, help="ADB device serial. Use when multiple devices are connected.")
    parser.add_argument("--instruction", default=None, help="Run one instruction and exit. Omit for interactive mode.")
    parser.add_argument("--screenshot_dir", default="/tmp/gui_agent_mobile")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--send_base64", action="store_true", help="Send screenshot bytes to server instead of local image_path.")
    parser.set_defaults(confirm=True)
    parser.add_argument("--confirm", dest="confirm", action="store_true", help="Ask before executing each predicted action (default).")
    parser.add_argument("--no-confirm", dest="confirm", action="store_false", help="Execute without confirmation. Use only in a safe test environment.")
    parser.add_argument("--wait_after_action", type=float, default=1.0)
    parser.add_argument("--scroll_duration_ms", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    try:
        cli_args = parse_args()
        cli_args.adb = resolve_adb(cli_args.adb)
        ensure_device_connected(cli_args)
        if cli_args.instruction:
            single_step(cli_args)
        else:
            interactive_loop(cli_args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
