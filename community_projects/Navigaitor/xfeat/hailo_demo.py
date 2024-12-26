"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Real-time homography estimation demo. Note that scene has to be planar or just rotate the camera for the estimation to work properly.
"""

import cv2
import numpy as np
import torch

from time import time, sleep
import time
import argparse, sys
import threading
import os
from datetime import datetime
from modules.xfeat import XFeat

import sys
sys.path.append("/home/pi/navigaitor/community_projects/Navigaitor/server/external")
import McLumk_Wheel_Sports as mclumk
 

def argparser():
    parser = argparse.ArgumentParser(description="Configurations for the real-time matching demo.")
    parser.add_argument('--width', type=int, default=640, help='Width of the video capture stream. only 640x480 or 320x224 resolution')
    parser.add_argument('--height', type=int, default=480, help='Height of the video capture stream. only 640x480 or 320x224 resolution')
    parser.add_argument('--max_kpts', type=int, default=3_000, help='Maximum number of keypoints.')
    parser.add_argument('--method', type=str, choices=['ORB', 'SIFT', 'XFeat'], default='XFeat', help='Local feature detection method to use.')
    parser.add_argument('--cam', type=int, default=0, help='Webcam device number.')
    parser.add_argument('--video', type=str, default="", help='video path.')
    parser.add_argument('--inference_type', type=str, default='hailo', help='"hailo" or torch or onnx')
    parser.add_argument('--record', action="store_true")
    parser.add_argument('--retreat', action="store_true")
    return parser.parse_args()


class FrameGrabber(threading.Thread):
    def __init__(self, cap, width, height):
        super().__init__()
        self.cap = cap
        _, self.frame = self.cap.read()
        self.running = False
        self.width = width
        self.height = height
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.fourcc = cv2.VideoWriter_fourcc(*"X264")
        self.hls_directory = "./test"
        self.gst_hls_pipeline = (
            f"appsrc ! "
            f"videoconvert ! "
            f"x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
            f"mpegtsmux ! "
            f"hlssink location={self.hls_directory}/segment_%05d.ts playlist-location={self.hls_directory}/playlist.m3u8 target-duration=5 max-files=5"
        )

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream ended?).")
            if (frame is None) or (self.frame is None):
                print("BADDDDD")
            cv2.VideoWriter(self.gst_hls_pipeline, self.fourcc, self.fps, (self.width, self.height))
            self.frame = frame
            sleep(0.05)

    def stop(self):
        self.running = False
        self.cap.release()

    def get_last_frame(self):
        self.frame = np.resize(self.frame, (self.height, self.width, 3))
        return self.frame

class ImageRecorder(threading.Thread):
    def __init__(self, frame_grabber, storage_dir):
        """
        Initialize the ImageRecorder class.

        Args:
            frame_grabber (FrameGrabber): Instance of an existing FrameGrabber.
            storage_dir (str): Directory to store recorded images.
        """
        super().__init__()
        self.frame_grabber = frame_grabber
        self.storage_dir = storage_dir
        self.running = False
        self.mode = "playback"  # Modes: 'record' or 'playback'
        self.output_queue = []
        self.current_image_index = 0

        # Ensure the storage directory exists
        os.makedirs(storage_dir, exist_ok=True)

    def run(self):
        self.running = True
        while self.running:
            if self.mode == "record":
                self.record_images()
            elif self.mode == "playback":
                time.sleep(0.1)  # Sleep to avoid busy looping in playback mode

    def stop(self):
        """
        Stop the thread and release resources.
        """
        self.running = False

    def switch_to_record(self):
        """
        Switch to recording mode.
        """
        self.mode = "record"

    def switch_to_playback(self):
        """
        Switch to playback mode and reset the playback index.
        """
        self.mode = "playback"
        self.current_image_index = 0

    def record_images(self):
        """
        Continuously capture and save images every 0.5 seconds in sequential order using the frame grabber.
        """
        while self.mode == "record" and self.running:
            frame = self.frame_grabber.get_last_frame()

            if frame is not None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
                filename = os.path.join(self.storage_dir, f"image_{timestamp}.png")
                cv2.imwrite(filename, frame)
                print(f"Image saved: {filename}")
                time.sleep(0.3)
            else:
                print("No frame available from frame grabber.")

    def get_next_image(self):
        """
        Get the next image in playback mode.

        Returns:
            frame (numpy array): The next image frame, or None if no more images are available.
        """
        if self.mode == "playback":
            image_files = sorted(os.listdir(self.storage_dir))
            print(len(image_files),"images found")
            if self.current_image_index < len(image_files):
                image_file = image_files[self.current_image_index]
                image_path = os.path.join(self.storage_dir, image_file)
                frame = cv2.imread(image_path)
                if frame is not None:
                    self.current_image_index += 1
                    print(f"Sent image: {image_path}")
                    return frame
                else:
                    print(f"Failed to load image: {image_path}")
            else:
                print("No more images to display.")
        return None

    def get_previous_image(self):
        """
        Get the previous image in playback mode.

        Returns:
            frame (numpy array): The previous image frame, or None if no more images are available.
        """
        if self.mode == "playback":
            image_files = sorted(os.listdir(self.storage_dir))
            if self.current_image_index > 0:
                self.current_image_index -= 1
                image_file = image_files[self.current_image_index]
                image_path = os.path.join(self.storage_dir, image_file)
                frame = cv2.imread(image_path)
                if frame is not None:
                    print(f"Sent image: {image_file}")
                    return frame
                else:
                    print(f"Failed to load image: {image_path}")
            else:
                print("Already at the first image.")
        return None

    def clean_images(self):
        """
        Remove all images from the storage directory.
        """
        image_files = os.listdir(self.storage_dir)
        for image_file in image_files:
            file_path = os.path.join(self.storage_dir, image_file)
            try:
                os.remove(file_path)
                print(f"Deleted image: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd
    def detectAndCompute(self, x, mask=None):
        return self.mtd.detectAndCompute(torch.tensor(x).permute(2,0,1).float()[None])[0]

class Method:
    def __init__(self, descriptor, matcher):
        self.descriptor = descriptor
        self.matcher = matcher

def init_method(method, max_kpts, width, height, device):
    if method == "ORB":
        return Method(descriptor=cv2.ORB_create(max_kpts, fastThreshold=10), matcher=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True))
    elif method == "SIFT":
        return Method(descriptor=cv2.SIFT_create(max_kpts, contrastThreshold=-1, edgeThreshold=1000), matcher=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True))
    elif method == "XFeat":
        return Method(descriptor=CVWrapper(XFeat(top_k = max_kpts, width=width, height=height, device=device)), matcher=XFeat(width=width, height=height, device=device))
    else:
        raise RuntimeError("Invalid Method.")


class MatchingDemo:
    def __init__(self, args):
        self.args = args
        if args.video != "":
            self.cap = cv2.VideoCapture(args.video)
        else:
            self.cap = cv2.VideoCapture(args.cam)
        self.width = args.width
        self.height = args.height
        self.device = args.inference_type
        self.ref_frame = None
        self.ref_precomp = [[],[]]
        self.corners = [[50, 50], [self.width-50, 50], [self.width-50, self.height-50], [50, self.height-50]]
        self.current_frame = None
        self.H = None
        self.setup_camera()

        #Init frame grabber thread
        self.frame_grabber = FrameGrabber(self.cap, self.width, self.height)
        self.frame_grabber.start()

        #recorder
        self.recorder = ImageRecorder(frame_grabber=self.frame_grabber, storage_dir="recorded_images")
        self.recorder.start()

        #Homography params
        self.min_inliers = 50
        self.ransac_thr = 4.0

        self.win = False

        #FPS check
        self.FPS = 0
        self.time_list = []
        self.max_cnt = 30 #avg FPS over this number of frames

        #Set local feature method here -- we expect cv2 or Kornia convention
        self.method = init_method(args.method, max_kpts=args.max_kpts, width= self.width, height=self.height, device=self.device)
        
        # Setting up font for captions
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9
        self.line_type = cv2.LINE_AA
        self.line_color = (0,255,0)
        self.line_thickness = 3

        self.window_name = "Real-time matching - Press 's' to set the reference frame."

        # Removes toolbar and status bar
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_GUI_NORMAL)
        # Set the window size
        cv2.resizeWindow(self.window_name, self.width*2, self.height*2)
        #Set Mouse Callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def setup_camera(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, 200)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def draw_quad(self, frame, point_list):
        if len(self.corners) > 1:
            for i in range(len(self.corners) - 1):
                cv2.line(frame, tuple(point_list[i]), tuple(point_list[i + 1]), self.line_color, self.line_thickness, lineType = self.line_type)
            if len(self.corners) == 4:  # Close the quadrilateral if 4 corners are defined
                cv2.line(frame, tuple(point_list[3]), tuple(point_list[0]), self.line_color, self.line_thickness, lineType = self.line_type)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) >= 4:
                self.corners = []  # Reset corners if already 4 points were clicked
            self.corners.append((x, y))

    def putText(self, canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
        # Draw the border
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=borderColor, thickness=thickness+2, lineType=lineType)
        # Draw the text
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=textColor, thickness=thickness, lineType=lineType)

    def warp_points(self, points, H, x_offset = 0):
        points_np = np.array(points, dtype='float32').reshape(-1,1,2)

        warped_points_np = cv2.perspectiveTransform(points_np, H).reshape(-1, 2)
        warped_points_np[:, 0] += x_offset
        warped_points = warped_points_np.astype(int).tolist()
        
        return warped_points

    def create_top_frame(self):
        top_frame_canvas = np.zeros((self.height, self.width*2, 3), dtype=np.uint8)
        top_frame = np.hstack((self.ref_frame, self.current_frame))
        color = (3, 186, 252)
        cv2.rectangle(top_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)  # Orange color line as a separator
        top_frame_canvas[0:self.height, 0:self.width*2] = top_frame
        
        # Adding captions on the top frame canvas
        self.putText(canvas=top_frame_canvas, text="Reference Frame:", org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        self.putText(canvas=top_frame_canvas, text="Target Frame:", org=(self.width+10, 30), fontFace=self.font, 
                    fontScale=self.font_scale,  textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
        self.draw_quad(top_frame_canvas, self.corners)
        
        return top_frame_canvas

    def get_area_mid(self, points):
        X = 0
        Y = 1
        top_left = 0
        top_right = 1
        bottom_right = 2
        bottom_left = 3

        height = points[bottom_right][Y] - points[top_right][Y]
        width = points[top_right][X] - points[top_left][X]
        area = height * width

        midx = points[top_left][X] + (width / 2)
        midy = points[top_left][X] + (height / 2)

        return area, midx, midy

    def print_directions(self, points, ref_points):
        area, midx, midy = self.get_area_mid(points)
        ref_area, ref_midx, ref_midy = self.get_area_mid(ref_points)
        midx -= self.width

        area_threshold = 0.05
        midx_threshold = 0.15
        speed_default = 5

        if ((1 - midx_threshold) < abs(midx / ref_midx) < (1 + midx_threshold)):
            if ((1 - area_threshold) < abs(area / ref_area) < (1 + area_threshold)):
                # Robot is in the right spot, next image
                self.ref_frame = self.recorder.get_next_image()
                if self.ref_frame is None:
                    print("Reached destination")
                    self.win = True
                    return
                self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None)
            elif area < ref_area:
                mclumk.move_forward(speed_default)
                print("Forward")
            else:
                mclumk.move_backward(speed_default)
                print("Backward")
        elif midx < ref_midx:
            mclumk.rotate_left(speed_default)
            print("Left")
        else:
            mclumk.rotate_right(speed_default)
            print("Right")

    def process(self):
        # Create a blank canvas for the top frame
        top_frame_canvas = self.create_top_frame()

        # Match features and draw matches on the bottom frame
        i = 0
        bottom_frame = self.match_and_draw(self.ref_frame, self.current_frame)
        # Draw warped corners
        if self.H is not None and len(self.corners) > 1:
            self.print_directions(self.warp_points(self.corners, self.H, self.width), self.corners)
            self.draw_quad(top_frame_canvas, self.warp_points(self.corners, self.H, self.width))

        # Stack top and bottom frames vertically on the final canvas
        canvas = np.vstack((top_frame_canvas, bottom_frame))

        cv2.imshow(self.window_name, canvas)

    def match_and_draw(self, ref_frame, current_frame):
        bad_frame = False
        bad_threshold = 60
        matches, good_matches = [], []
        kp1, kp2 = [], []
        points1, points2 = [], []

        # Detect and compute features
        if self.args.method in ['SIFT', 'ORB']:
            kp1, des1 = self.ref_precomp
            kp2, des2 = self.method.descriptor.detectAndCompute(current_frame, None)
        else:
            # start = time()
            current = self.method.descriptor.detectAndCompute(current_frame)
            # end = time()
            # print(end-start)
            kpts1, descs1 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']
            kpts2, descs2 = current['keypoints'], current['descriptors']
            if len(kpts1) == 0 or len(kpts2) == 0:
                return np.hstack([ref_frame, current_frame])
            idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.82)
            points1 = kpts1[idx0].cpu().numpy()
            points2 = kpts2[idx1].cpu().numpy()

        # if len(kp1) > 10 and len(kp2) > 10 and self.args.method in ['SIFT', 'ORB']:
        #     # Match descriptors
        #     matches = self.method.matcher.match(des1, des2)

        #     if len(matches) > 10:
        #         points1 = np.zeros((len(matches), 2), dtype=np.float32)
        #         points2 = np.zeros((len(matches), 2), dtype=np.float32)

        #         for i, match in enumerate(matches):
        #             points1[i, :] = kp1[match.queryIdx].pt
        #             points2[i, :] = kp2[match.trainIdx].pt

        if len(points1) > bad_threshold and len(points2) > bad_threshold:
            # Find homography
            self.H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700, confidence=0.995)
            inliers = inliers.flatten() > 0
            
            if inliers.sum() < self.min_inliers:
                self.H = None

            if self.args.method in ["SIFT", "ORB"]:
                good_matches = [m for i,m in enumerate(matches) if inliers[i]]
            else:
                kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
                kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
                good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]
#                print("kp1",kp1)
#                print("kp2",kp2)
#                print(good_matches) 
            # Draw matches
            matched_frame = cv2.drawMatches(ref_frame, kp1, current_frame, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)
            
        else:
            matched_frame = np.hstack([ref_frame, current_frame])
            bad_frame = True

        color = (240, 89, 169)

        # Add a colored rectangle to separate from the top frame
        cv2.rectangle(matched_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)

        # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="%s Matches: %d"%(self.args.method, len(good_matches)), org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
                # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="FPS (registration): {:.1f}".format(self.FPS), org=(self.width+10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        if bad_frame:
            return None
        return matched_frame
    
    """main API functions: start_playback, start_recording, stop recording"""
    def start_playback(self):
        self.recorder.switch_to_playback()
        self.ref_frame = self.recorder.get_next_image()
        self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None)

        while not self.win:
            self.current_frame = self.frame_grabber.get_last_frame()
            if self.current_frame is None:
                print("frame is none, bye")
                break

            self.process()
            
        self.cleanup()

    def start_recording(self):
        self.recorder.switch_to_record()
        
    def stop_recording(self):
        self.recorder.switch_to_playback()

        
    def main_loop(self):
        self.start_recording()
        # self.current_frame = self.frame_grabber.get_last_frame()
        # # self.ref_frame = self.current_frame.copy()
        # # self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features

        # self.ref_frame = self.recorder.get_next_image()
        # self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None)

        # #record for 5 seconds
        sleep(10)
        self.stop_recording()
        print("STOPPED RECORDING, MOVING TO PLAYBACK")
        sleep(5)
        self.start_playback()

        # while True:
        #     if self.current_frame is None:
        #         break

        #     t0 = time.time()
        #     self.process()

        #     key = cv2.waitKey(1)
        #     if key == ord('q'):
        #         break
        #     # elif key == ord('s'):
        #     #     self.ref_frame = self.current_frame.copy()  # Update reference frame
        #     #     self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features

        #     self.current_frame = self.frame_grabber.get_last_frame()

        #     #Measure avg. FPS
        #     self.time_list.append(time.time()-t0)
        #     if len(self.time_list) > self.max_cnt:
        #         self.time_list.pop(0)
        #     self.FPS = 1.0 / np.array(self.time_list).mean()
        
        self.cleanup()

    def cleanup(self):
        self.recorder.stop()
        self.frame_grabber.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        mclumk.stop_robot()

if __name__ == "__main__":

    mclumk.stop_robot()
    args = argparser()
    demo = MatchingDemo(args)
    if args.record:
        demo.start_recording()
    elif args.retreat:
        demo.start_playback()
