#%%
#
# WebCam.py
# ========================================================
# interface to cv2 for using a webcam or for reading from
# a video file
#
# Version 1.0, before 30.03.2023:
#   written by David...
#
# Version 1.1, 30.03.2023:
#   thrown out test image
#   added test code
#   added code to "capture" from video file
#
# Version 1.2, 20.06.2023:
#   added code to capture wirelessly from "GoPro" camera
#   added test code for "GoPro"
#
# Version 1.3, 23.06.2023
#   test display in pyplot or cv2
#
# Version 1.4, 28.06.2023
#   solved Windows DirectShow problem
#


from PIL import Image
import os
import cv2
import torch
import torchvision as tv

# for GoPro
import time
import socket

try:
    print("Trying to import GoPro modules...")
    from goprocam import GoProCamera, constants

    gopro_exists = True
except:
    print("...not found, continuing!")
    gopro_exists = False
import platform


class WebCam:

    # test_pattern: torch.Tensor
    # test_pattern_gray: torch.Tensor

    source: int
    framesize: tuple[int, int]
    fps: float
    cap_frame_width: int
    cap_frame_height: int
    cap_fps: float
    webcam_is_ready: bool
    cap_frames_available: int

    default_dtype = torch.float32

    def __init__(
        self,
        source: str | int = 1,
        framesize: tuple[int, int] = (720, 1280),  # (1920, 1080),  # (640, 480),
        fps: float = 30.0,
    ):
        super().__init__()
        assert fps > 0

        self.source = source
        self.framesize = framesize
        self.cap = None
        self.fps = fps
        self.webcam_is_ready = False

    def open_cam(self) -> bool:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.webcam_is_ready = False

        # handle GoPro...
        if self.source == "GoProWireless":

            if not gopro_exists:
                print("No GoPro driver/support!")
                self.webcam_is_ready = False
                return False

            print("GoPro: Starting access")
            gpCam = GoProCamera.GoPro()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print("GoPro: Socket created!!!")

            self.t = time.time()
            gpCam.livestream("start")
            gpCam.video_settings(res="1080p", fps="30")
            gpCam.gpControlSet(
                constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720
            )

            self.cap = cv2.VideoCapture("udp://10.5.5.9:8554", cv2.CAP_FFMPEG)
            print("GoPro: Video capture started!!!")

        else:
            self.sock = None
            self.t = -1
            if platform.system().lower() == "windows":
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.source)
            print("Normal capture started!!!")

        assert self.cap is not None

        if self.cap.isOpened() is not True:
            self.webcam_is_ready = False
            return False

        if type(self.source) != str:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.framesize[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.framesize[1])
            self.cap_frames_available = None
        else:
            # ...for files and GoPro...
            self.cap_frames_available = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        self.cap_frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        print(
            (
                f"Capturing or reading with: {self.cap_frame_width:.0f} x "
                f"{self.cap_frame_height:.0f} @ "
                f"{self.cap_fps:.1f}."
            )
        )
        self.webcam_is_ready = True
        return True

    def close_cam(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def get_frame(self) -> torch.Tensor | None:
        if self.cap is None:
            return None
        else:
            if self.sock:
                dt_min = 0.015
                success, frame = self.cap.read()
                t_next = time.time()
                t_prev = t_next
                while t_next - t_prev < dt_min:
                    t_prev = t_next
                    success, frame = self.cap.read()
                    t_next = time.time()

                if self.t >= 0:
                    if time.time() - self.t > 2.5:
                        print("GoPro-Stream must be kept awake!...")
                        self.sock.sendto(
                            "_GPHD_:0:0:2:0.000000\n".encode(), ("10.5.5.9", 8554)
                        )
                        self.t = time.time()
            else:
                success, frame = self.cap.read()

        if success is False:
            self.webcam_is_ready = False
            return None

        output = (
            torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            .movedim(-1, 0)
            .type(dtype=self.default_dtype)
            / 255.0
        )
        return output


# for testing the code if module is executed from command line
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import cv2

    TEST_FILEREAD = False
    TEST_WEBCAM = True
    TEST_GOPRO = False

    display = "cv2"
    n_capture = 200
    delay_capture = 0.001

    print("Testing the WebCam interface")

    if TEST_FILEREAD:

        file_name = "level1.mp4"

        # open
        print("Opening video file")
        w = WebCam(file_name)
        if not w.open_cam():
            raise OSError(f"Opening file with name {file_name} failed!")

        # print information
        print(
            f"Frame size {w.cap_frame_width} x {w.cap_frame_height} at {w.cap_fps} fps."
        )

        # capture three frames and show them
        for i in range(min([n_capture, w.cap_frames_available])):  # TODO: available?
            frame = w.get_frame()
            if frame == None:
                raise OSError(f"Can not get frame from file with name {file_name}!")
            print(f"frame {i} has shape {frame.shape}")

            frame_numpy = (frame.movedim(0, -1) * 255).type(dtype=torch.uint8).numpy()

            if display == "pyplot":
                plt.imshow(frame_numpy)
                plt.show()
            if display == "cv2":
                cv2.imshow("File", frame_numpy[:, :, (2, 1, 0)])
                cv2.waitKey(1)
            time.sleep(delay_capture)

        # close
        print("Closing file")
        w.close_cam()

    if TEST_WEBCAM:

        camera_index = 0

        # open
        print("Opening camera")
        w = WebCam(camera_index)
        if not w.open_cam():
            raise OSError(f"Opening web cam with index {camera_index} failed!")

        # print information
        print(
            f"Frame size {w.cap_frame_width} x {w.cap_frame_height} at {w.cap_fps} fps."
        )

        # capture three frames and show them
        for i in range(n_capture):
            frame = w.get_frame()
            if frame == None:
                raise OSError(
                    f"Can not get frame from camera with index {camera_index}!"
                )
            print(f"frame {i} has shape {frame.shape}")

            frame_numpy = (frame.movedim(0, -1) * 255).type(dtype=torch.uint8).numpy()
            if display == "pyplot":
                plt.imshow(frame_numpy)
                plt.show()
            if display == "cv2":
                cv2.imshow("WebCam", frame_numpy[:, :, (2, 1, 0)])
                cv2.waitKey(1)
            time.sleep(delay_capture)

        # close
        print("Closing camera")
        w.close_cam()

    if TEST_GOPRO:

        camera_name = "GoProWireless"

        # open
        print("Opening GoPro")
        w = WebCam(camera_name)
        if not w.open_cam():
            raise OSError(f"Opening GoPro with index {camera_index} failed!")

        # print information
        print(
            f"Frame size {w.cap_frame_width} x {w.cap_frame_height} at {w.cap_fps} fps."
        )
        w.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        # capture three frames and show them
        # print("Empty Buffer...")
        # for i in range(500):
        #     print(i)
        #     frame = w.get_frame()
        # print("Buffer Emptied...")

        for i in range(n_capture):
            frame = w.get_frame()
            if frame == None:
                raise OSError(
                    f"Can not get frame from camera with index {camera_index}!"
                )
            print(f"frame {i} has shape {frame.shape}")

            frame_numpy = (frame.movedim(0, -1) * 255).type(dtype=torch.uint8).numpy()
            if display == "pyplot":
                plt.imshow(frame_numpy)
                plt.show()
            if display == "cv2":
                cv2.imshow("GoPro", frame_numpy[:, :, (2, 1, 0)])
                cv2.waitKey(1)
            time.sleep(delay_capture)

        # close
        print("Closing Cam/File/GoPro")
        w.close_cam()

    if display == "cv2":
        cv2.destroyAllWindows()

# %%
