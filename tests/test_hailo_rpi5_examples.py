import pytest
import subprocess
import os
import sys
import time
import signal
import glob
import logging

# Adjust the sys.path to include the parent directory of the test folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basic_pipelines.get_usb_camera import get_usb_video_devices

try:
    from picamera2 import Picamera2
    rpi_camera_available = True
except ImportError:
    rpi_camera_available = False

TEST_RUN_TIME = 10

def get_device_architecture():
    """Get the device architecture from hailortcli."""
    try:
        result = subprocess.run(['hailortcli', 'fw-control', 'identify'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if "Device Architecture" in line:
                return line.split(':')[1].strip()
    except Exception:
        return "unknown"

def get_pipelines_list():
    """Get a list of available pipeline scripts."""
    return ["detection.py", "pose_estimation.py", "instance_segmentation.py"]

def get_detection_compatible_hefs(architecture):
    """Get a list of compatible HEF files based on the device architecture."""
    H8_HEFS = [
        "yolov5m_wo_spp.hef",
        "yolov6n.hef",
        "yolov8s.hef",
        "yolov8m.hef",
    ]

    H8L_HEFS = [
        "yolov8s_h8l.hef",
        "yolov6n.hef",
        "yolox_s_leaky_h8l_mz.hef"
    ]
    hef_list = H8L_HEFS
    if architecture == 'HAILO8':
        # check both HAILO8 and HAILO8L
        hef_list = hef_list + H8_HEFS

    return [os.path.join("resources", hef) for hef in hef_list]

def test_all_pipelines():
    """
    Combined test function for basic pipeline scripts with different HEFs and input sources.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    available_cameras = get_usb_video_devices()
    pipeline_list = get_pipelines_list()
    if rpi_camera_available:
        available_cameras.append("rpi")

    for pipeline in pipeline_list:
        # Test with video input
        log_file_path = os.path.join(log_dir, f"test_{pipeline}_video_test.log")

        with open(log_file_path, "w") as log_file:
            cmd = ['python', f'basic_pipelines/{pipeline}']

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info(f"Running {pipeline} with video input")
            try:
                time.sleep(TEST_RUN_TIME)
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                pytest.fail(f"{pipeline} (video input) could not be terminated within 5 seconds after running for {TEST_RUN_TIME} seconds")

            stdout, stderr = process.communicate()
            log_file.write(f"{pipeline} (video input) stdout:\n{stdout.decode()}\n")
            log_file.write(f"{pipeline} (video input) stderr:\n{stderr.decode()}\n")

            assert "Traceback" not in stderr.decode(), f"{pipeline} (video input) encountered an exception: {stderr.decode()}"
            assert "Error" not in stderr.decode(), f"{pipeline} (video input) encountered an error: {stderr.decode()}"
            assert "frame" in stdout.decode().lower(), f"{pipeline} (video input) did not process any frames"
            assert "detection" in stdout.decode().lower(), f"{pipeline} (video input) did not make any detections"

        # Test with available cameras
        for device in available_cameras:
            # if device is /dev/video* device name should be video*
            if device.startswith("/dev/video"):
                device_name = device.split("/")[-1]
            else:
                device_name = device
            log_file_path = os.path.join(log_dir, f"test_{pipeline}_{device_name}_camera_test.log")
            logging.info(f"Running {pipeline} with {device} camera")
            with open(log_file_path, "w") as log_file:
                cmd = ['python', f'basic_pipelines/{pipeline}', '--input', device]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    time.sleep(TEST_RUN_TIME)
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    pytest.fail(f"{pipeline} ({device} camera) could not be terminated within 5 seconds after running for {TEST_RUN_TIME} seconds")
                stdout, stderr = process.communicate()
                log_file.write(f"{pipeline} ({device} camera) stdout:\n{stdout.decode()}\n")
                log_file.write(f"{pipeline} ({device} camera) stderr:\n{stderr.decode()}\n")
                assert "Traceback" not in stderr.decode(), f"{pipeline} ({device} camera) encountered an exception: {stderr.decode()}"
                assert "Error" not in stderr.decode(), f"{pipeline} ({device} camera) encountered an error: {stderr.decode()}"

    # Check if expected cameras are available
    if len(available_cameras) == 0:
        pytest.fail(f"No available cameras found for testing")
    if len(available_cameras) < 2 and rpi_camera_available:
        pytest.fail(f"Only one camera found for testing, both USB or RPi camera is required")

def test_detection_hefs():
    """
    Combined test function for basic pipeline scripts with different HEFs and input sources.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    architecture = get_device_architecture()
    compatible_hefs = get_detection_compatible_hefs(architecture)
    for hef in compatible_hefs:
        hef_name = os.path.basename(hef)

        # Test with video input
        log_file_path = os.path.join(log_dir, f"detection_{hef_name}_video_test.log")
        logging.info(f"Running detection with {hef_name} (video input)")
        with open(log_file_path, "w") as log_file:
            process = subprocess.Popen(['python', 'basic_pipelines/detection.py', '--input', 'resources/detection0.mp4', '--hef-path', hef],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                time.sleep(TEST_RUN_TIME)
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                pytest.fail(f"Detection with {hef_name} (video input) could not be terminated within 5 seconds after running for {TEST_RUN_TIME} seconds")

            stdout, stderr = process.communicate()
            log_file.write(f"Detection with {hef_name} (video input) stdout:\n{stdout.decode()}\n")
            log_file.write(f"Detection with {hef_name} (video input) stderr:\n{stderr.decode()}\n")

            assert "Traceback" not in stderr.decode(), f"Detection with {hef_name} (video input) encountered an exception: {stderr.decode()}"
            assert "Error" not in stderr.decode(), f"Detection with {hef_name} (video input) encountered an error: {stderr.decode()}"
            assert "frame" in stdout.decode().lower(), f"Detection with {hef_name} (video input) did not process any frames"
            assert "detection" in stdout.decode().lower(), f"Detection with {hef_name} (video input) did not make any detections"

def test_detection_retraining():
    """
    Test the detection pipeline with a retrained model.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    retrained_hef = "resources/yolov8s-hailo8l-barcode.hef"
    labels_json = "resources/barcode-labels.json"
    video_path = "resources/barcode.mp4"
    log_file_path = os.path.join(log_dir, "detection_retrained_video_test.log")
    logging.info("Running detection with retrained model (video input)")
    with open(log_file_path, "w") as log_file:
        cmd = ['python', 'basic_pipelines/detection.py', '--labels-json', labels_json, '--hef-path', retrained_hef, '--input', video_path]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            time.sleep(TEST_RUN_TIME)
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            pytest.fail(f"Detection with retrained model (video input) could not be terminated within 5 seconds after running for {TEST_RUN_TIME} seconds")

        stdout, stderr = process.communicate()
        log_file.write(f"Detection with retrained model (video input) stdout:\n{stdout.decode()}\n")
        log_file.write(f"Detection with retrained model (video input) stderr:\n{stderr.decode()}\n")

        assert "Traceback" not in stderr.decode(), f"Detection with retrained model (video input) encountered an exception: {stderr.decode()}"
        assert "Error" not in stderr.decode(), f"Detection with retrained model (video input) encountered an error: {stderr.decode()}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])