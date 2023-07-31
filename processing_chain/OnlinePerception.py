#%%

import torch
import numpy as np
from processing_chain.BuildImage import BuildImage
import matplotlib.pyplot as plt
import time
import cv2
from gui.GUIEvents import GUIEvents
from gui.GUICombiData import GUICombiData
import tkinter as tk


# TODO required?
def embed_image(frame_torch, out_height, out_width, torch_device, init_value=0):

    out_shape = torch.tensor(frame_torch.shape)

    frame_width = frame_torch.shape[-1]
    frame_height = frame_torch.shape[-2]

    frame_width_idx0 = max([0, (frame_width - out_width) // 2])
    frame_height_idx0 = max([0, (frame_height - out_height) // 2])

    select_width = min([frame_width, out_width])
    select_height = min([frame_height, out_height])

    out_shape[-1] = out_width
    out_shape[-2] = out_height

    out_torch = init_value * torch.ones(tuple(out_shape), device=torch_device)

    out_width_idx0 = max([0, (out_width - frame_width) // 2])
    out_height_idx0 = max([0, (out_height - frame_height) // 2])

    out_torch[
        ...,
        out_height_idx0 : (out_height_idx0 + select_height),
        out_width_idx0 : (out_width_idx0 + select_width),
    ] = frame_torch[
        ...,
        frame_height_idx0 : (frame_height_idx0 + select_height),
        frame_width_idx0 : (frame_width_idx0 + select_width),
    ]

    return out_torch


class OnlinePerception:

    # SELFies...
    #
    # torch_device, default_dtype (fixed)
    # canvas_size, features, phosphene, position_found (parameters)
    # percept
    #
    # root, events, confdata, use_gui
    #

    def __init__(self, target, use_gui=False):

        # CPU or GPU?
        self.default_dtype = torch.float32
        torch.set_default_dtype(self.default_dtype)
        if torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
        else:
            self.torch_device = torch.device("cpu")

        self.use_gui = use_gui
        if self.use_gui:
            self.root = tk.Tk()
            self.confdata: GUICombiData = GUICombiData()
            self.events = GUIEvents(tk_root=self.root, confdata=self.confdata)

        # default dictionary parameters
        n_xy_canvas = 400
        n_xy_features = 41
        n_features = 32
        n_positions = 1

        # populate internal parameters
        self.canvas_size = [1, 8, n_xy_canvas, n_xy_canvas]
        self.features = torch.rand(
            (n_features, 1, n_xy_features, n_xy_features), device=self.torch_device
        )
        self.phosphene = torch.ones(
            (1, 1, n_xy_features, n_xy_features), device=self.torch_device
        )
        self.position_found = torch.zeros((1, n_positions, 3), device=self.torch_device)
        self.position_found[0, :, 0] = torch.randint(n_features, (n_positions,))
        self.position_found[0, :, 1:] = torch.randint(n_xy_canvas, (n_positions, 2))
        self.percept = torch.zeros(
            (1, 1, n_xy_canvas, n_xy_canvas), device=self.torch_device
        )

        self.p_FEATperSECperPOS = 3.0
        self.tau_SEC = 0.3  # percept time constant
        self.t = time.time()

        self.target = target  # display target
        self.display = target

        self.selection = 0

        return

    def update(self, data_in: dict):

        if data_in:  # not NONE?
            print(f"Parameter update requested for keys {data_in.keys()}!")
            if "position_found" in data_in.keys():
                self.position_found = data_in["position_found"]
            if "canvas_size" in data_in.keys():
                self.canvas_size = data_in["canvas_size"]
                self.percept = embed_image(
                    self.percept,
                    self.canvas_size[-2],
                    self.canvas_size[-1],
                    torch_device=self.torch_device,
                    init_value=0,
                )
            if "features" in data_in.keys():
                print(self.features.shape, self.features.max(), self.features.min())
                self.features = data_in["features"]
                self.features /= self.features.max()
                print(self.features.shape, self.features.max(), self.features.min())
            if "phosphene" in data_in.keys():
                self.phosphene = data_in["phosphene"]
                self.phosphene /= self.phosphene.max()

        # parameters of (optional) GUI changed?
        data_out = None
        if self.use_gui:
            self.root.update()
            if not self.confdata.gui_running:
                data_out = {"exit": 42}
                # graceful exit
            else:
                if self.confdata.check_for_change() is True:

                    data_out = {
                        "value": self.confdata.yolo_class.value,
                        "sigma_kernel_DVA": self.confdata.contour_extraction.sigma_kernel_DVA,
                        "n_orientations": self.confdata.contour_extraction.n_orientations,
                        "size_DVA": self.confdata.alphabet.size_DVA,
                        # "tau_SEC": self.confdata.alphabet.tau_SEC,
                        # "p_FEATperSECperPOS": self.confdata.alphabet.p_FEATperSECperPOS,
                        "sigma_width": self.confdata.alphabet.phosphene_sigma_width,
                        "n_dir": self.confdata.alphabet.clocks_n_dir,
                        "pointer_width": self.confdata.alphabet.clocks_pointer_width,
                        "pointer_length": self.confdata.alphabet.clocks_pointer_length,
                        "number_of_patches": self.confdata.sparsifier.number_of_patches,
                        "use_exp_deadzone": self.confdata.sparsifier.use_exp_deadzone,
                        "use_cutout_deadzone": self.confdata.sparsifier.use_cutout_deadzone,
                        "size_exp_deadzone_DVA": self.confdata.sparsifier.size_exp_deadzone_DVA,
                        "size_cutout_deadzone_DVA": self.confdata.sparsifier.size_cutout_deadzone_DVA,
                        "enable_cam": self.confdata.output_mode.enable_cam,
                        "enable_yolo": self.confdata.output_mode.enable_yolo,
                        "enable_contour": self.confdata.output_mode.enable_contour,
                    }
                    print(data_out)

                    self.p_FEATperSECperPOS = self.confdata.alphabet.p_FEATperSECperPOS
                    self.tau_SEC = self.confdata.alphabet.tau_SEC

                    # print(self.confdata.alphabet.selection)
                    self.selection = self.confdata.alphabet.selection
                    # print(f"Selektion gemacht {self.selection}")

                    # print(self.confdata.output_mode.enable_percept)
                    if self.confdata.output_mode.enable_percept:
                        self.display = self.target
                    else:
                        self.display = None
                        if self.target == "cv2":
                            cv2.destroyWindow("Percept")

                    self.confdata.reset_change_detector()

        # keep track of time, yields dt
        t_new = time.time()
        dt = t_new - self.t
        self.t = t_new

        # exponential decay
        self.percept *= torch.exp(-torch.tensor(dt / self.tau_SEC))

        # new stimulation
        p_dt = self.p_FEATperSECperPOS * dt
        n_positions = self.position_found.shape[-2]
        position_select = torch.rand((n_positions,), device=self.torch_device) < p_dt
        n_select = position_select.sum()
        if n_select > 0:
            # print(f"Selektion ausgewertet {self.selection}")
            if self.selection:
                dictionary = self.features
            else:
                dictionary = self.phosphene
            percept_addon = BuildImage(
                canvas_size=self.canvas_size,
                dictionary=dictionary,
                position_found=self.position_found[:, position_select, :],
                default_dtype=self.default_dtype,
                torch_device=self.torch_device,
            )
            self.percept += percept_addon

        # prepare for display
        display = self.percept[0, 0].cpu().numpy()
        if self.display == "cv2":

            # display, and update
            cv2.namedWindow("Percept", cv2.WINDOW_NORMAL)
            cv2.imshow("Percept", display)
            q = cv2.waitKey(1)

        if self.display == "pyplot":

            # display, RUNS SLOWLY, just for testing
            plt.imshow(display, cmap="gray", vmin=0, vmax=1)
            plt.show()
            time.sleep(0.5)

        return data_out

    def __del__(self):
        print("Now I'm deleting me!")
        try:
            cv2.destroyAllWindows()
        except:
            pass
        if self.use_gui:
            print("...and the GUI!")
            try:
                if self.confdata.gui_running is True:
                    self.root.destroy()
            except:
                pass


if __name__ == "__main__":

    use_gui = True

    data_in = None
    op = OnlinePerception("cv2", use_gui=use_gui)

    t_max = 40.0
    dt_update = 10.0
    t_update = dt_update

    t = time.time()
    t0 = t
    while t - t0 < t_max:

        data_out = op.update(data_in)
        data_in = None
        if data_out:
            print("Output given!")
            if "exit" in data_out.keys():
                break

        t = time.time()
        if t - t0 > t_update:

            # new canvas size
            n_xy_canvas = int(400 + torch.randint(200, (1,)))
            canvas_size = [1, 8, n_xy_canvas, n_xy_canvas]

            # new features/phosphenes
            n_features = int(16 + torch.randint(16, (1,)))
            n_xy_features = int(31 + 2 * torch.randint(10, (1,)))
            features = torch.rand(
                (n_features, 1, n_xy_features, n_xy_features), device=op.torch_device
            )
            phosphene = torch.ones(
                (1, 1, n_xy_features, n_xy_features), device=op.torch_device
            )

            # new positions
            n_positions = int(1 + torch.randint(3, (1,)))
            position_found = torch.zeros((1, n_positions, 3), device=op.torch_device)
            position_found[0, :, 0] = torch.randint(n_features, (n_positions,))
            position_found[0, :, 1] = torch.randint(n_xy_canvas, (n_positions,))
            position_found[0, :, 2] = torch.randint(n_xy_canvas, (n_positions,))
            t_update += dt_update
            data_in = {
                "position_found": position_found,
                "canvas_size": canvas_size,
                "features": features,
                "phosphene": phosphene,
            }

    print("done!")
    del op


# %%
