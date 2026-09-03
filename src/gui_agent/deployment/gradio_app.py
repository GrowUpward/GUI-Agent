import argparse
import json
import os
import socket
import urllib.request
from types import SimpleNamespace

import gradio as gr
from PIL import Image, ImageDraw

from gui_agent.deployment.adb_controller import (
    capture_screenshot,
    ensure_device_connected,
    execute_action,
    next_screenshot_path,
    request_prediction,
    resolve_adb,
)


def make_args(adb, serial, server_url, screenshot_dir, timeout, send_base64):
    return SimpleNamespace(
        adb=resolve_adb(adb),
        serial=serial.strip() or None,
        server_url=server_url.strip(),
        screenshot_dir=screenshot_dir.strip(),
        timeout=int(timeout),
        send_base64=send_base64,
        scroll_x1=500,
        scroll_y1=1600,
        scroll_x2=500,
        scroll_y2=600,
        scroll_duration_ms=300,
    )


def draw_prediction(image_path, prediction):
    coordinate = prediction.get("coordinate")
    if not coordinate or len(coordinate) != 2:
        return image_path

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y = map(int, coordinate)
    radius = 14
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=5)
    draw.line((x - 28, y, x + 28, y), fill="red", width=3)
    draw.line((x, y - 28, x, y + 28), fill="red", width=3)
    draw.rectangle((x + 18, y + 18, x + 150, y + 50), fill=(0, 0, 0))
    draw.text((x + 24, y + 24), f"({x}, {y})", fill="white")

    overlay_path = os.path.splitext(image_path)[0] + "_overlay.png"
    image.save(overlay_path)
    return overlay_path


def check_connections(adb, serial, server_url, screenshot_dir, timeout, send_base64):
    args = make_args(adb, serial, server_url, screenshot_dir, timeout, send_base64)
    ensure_device_connected(args)

    request = urllib.request.Request(args.server_url.rstrip("/") + "/health", method="GET")
    with urllib.request.urlopen(request, timeout=int(timeout)) as response:
        health = response.read().decode("utf-8")

    return f"ADB device OK: {args.serial}\nModel server OK: {health}"


def predict_once(instruction, adb, serial, server_url, screenshot_dir, timeout, send_base64, auto_execute):
    if not instruction.strip():
        raise gr.Error("Instruction is empty.")

    args = make_args(adb, serial, server_url, screenshot_dir, timeout, send_base64)
    ensure_device_connected(args)
    os.makedirs(args.screenshot_dir, exist_ok=True)

    image_path = next_screenshot_path(args)
    capture_screenshot(args, image_path)
    prediction = request_prediction(args, image_path, instruction.strip())
    overlay_path = draw_prediction(image_path, prediction)

    state = {
        "prediction": prediction,
        "adb": args.adb,
        "serial": args.serial,
        "server_url": args.server_url,
        "screenshot_dir": args.screenshot_dir,
        "timeout": args.timeout,
        "send_base64": args.send_base64,
    }
    executed = None
    if auto_execute:
        executed = execute_action(args, prediction)

    status = (
        f"Saved screenshot: {image_path}\n"
        f"Prediction overlay: {overlay_path}\n"
        f"Action: {prediction.get('action')} Coordinate: {prediction.get('coordinate')}"
    )
    if executed:
        status += f"\nExecuted: {executed}"
    return image_path, overlay_path, json.dumps(prediction, ensure_ascii=False, indent=2), status, state


def execute_last(state):
    if not state or not state.get("prediction"):
        raise gr.Error("No prediction to execute. Run Predict first.")

    args = SimpleNamespace(
        adb=state["adb"],
        serial=state["serial"],
        server_url=state["server_url"],
        screenshot_dir=state["screenshot_dir"],
        timeout=state["timeout"],
        send_base64=state["send_base64"],
        scroll_x1=500,
        scroll_y1=1600,
        scroll_x2=500,
        scroll_y2=600,
        scroll_duration_ms=300,
    )
    executed = execute_action(args, state["prediction"])
    return f"Executed: {executed}"


def build_ui(defaults):
    with gr.Blocks(title="GUI-Agent Mobile Debugger") as demo:
        gr.Markdown("# GUI-Agent Mobile Debugger")
        state = gr.State({})

        with gr.Row():
            server_url = gr.Textbox(label="Model Server URL", value=defaults.server_url)
            timeout = gr.Number(label="Timeout Seconds", value=defaults.timeout, precision=0)

        with gr.Row():
            adb = gr.Textbox(label="ADB Path", value=defaults.adb)
            serial = gr.Textbox(label="ADB Serial", value=defaults.serial)

        with gr.Row():
            screenshot_dir = gr.Textbox(label="Screenshot Directory", value=defaults.screenshot_dir)
            send_base64 = gr.Checkbox(label="Send screenshot as base64", value=True)

        instruction = gr.Textbox(label="Instruction", value="open settings", lines=2)
        auto_execute = gr.Checkbox(
            label="Auto execute after prediction",
            value=False,
            info="Enable only after prediction coordinates are stable.",
        )

        with gr.Row():
            check_btn = gr.Button("Check Connections")
            predict_btn = gr.Button("Capture + Predict", variant="primary")
            execute_btn = gr.Button("Execute Last Prediction", variant="stop")

        status = gr.Textbox(label="Status", lines=5)

        with gr.Row():
            screenshot = gr.Image(label="Screenshot Sent To Model", type="filepath")
            overlay = gr.Image(label="Prediction Overlay", type="filepath")

        prediction_json = gr.Code(label="Raw Prediction JSON", language="json")

        check_btn.click(
            check_connections,
            inputs=[adb, serial, server_url, screenshot_dir, timeout, send_base64],
            outputs=status,
        )
        predict_btn.click(
            predict_once,
            inputs=[instruction, adb, serial, server_url, screenshot_dir, timeout, send_base64, auto_execute],
            outputs=[screenshot, overlay, prediction_json, status, state],
        )
        execute_btn.click(execute_last, inputs=state, outputs=status)

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio WebUI for GUI-Agent mobile debugging.")
    parser.add_argument("--server_url", default="http://127.0.0.1:8000")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", default="")
    parser.add_argument("--screenshot_dir", default="mobile_debug")
    parser.add_argument("--timeout", type=int, default=500)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    try:
        demo = build_ui(cli_args)
        demo.launch(
            server_name=cli_args.host,
            server_port=cli_args.port,
            share=cli_args.share,
            allowed_paths=[os.path.abspath(cli_args.screenshot_dir)],
        )
    except (FileNotFoundError, socket.timeout) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
