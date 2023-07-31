from gui.GUIMasterData import GUIMasterData
from gui.GUIMasterGUI import GUIMasterGUI
import tkinter as tk
from tkinter import ttk

import torch
import torchvision as tv


class GUIContourExtractionData(GUIMasterData):
    sigma_kernel_DVA: float = 0.06
    # sigma_kernel: float
    # lambda_kernel: float
    n_orientations: int = 8

    # padding_x: int
    # padding_y: int
    # padding_fill: int

    def __init__(self) -> None:
        super().__init__()

        # self.calculate_setting()
        # self.padding_fill: int = int(
        #     tv.transforms.functional.rgb_to_grayscale(
        #         torch.full((3, 1, 1), 0)
        #     ).squeeze()
        # )

    # def calculate_setting(self) -> None:
    #     self.sigma_kernel = 1.0 * self.scale_kernel
    #     self.lambda_kernel = 2.0 * self.scale_kernel
    #     self.padding_x: int = int(3.0 * self.scale_kernel)
    #     self.padding_y: int = int(3.0 * self.scale_kernel)


class GUIContourExtractionGUI(GUIMasterGUI):
    def __init__(
        self,
        tk_root: tk.Tk | tk.ttk.Labelframe | tk.ttk.Frame,
        name: str = "Contour extraction",
        row_id: int = 0,
        column_id: int = 0,
        data_class=None,
    ):
        super().__init__(
            tk_root,
            name=name,
            row_id=row_id,
            column_id=column_id,
            data_class=data_class,
        )
        width_element: int = 10
        width_label: int = 20
        width_button_extra: int = 5

        # orientations ->
        self.label_orientations = ttk.Label(
            self.frame, text="Number orientations", width=width_label
        )
        self.label_orientations.grid(row=0, column=0, sticky="w")

        self.spinbox_n_orientations = ttk.Spinbox(
            self.frame,
            values=list("{:d}".format(x) for x in range(1, 33)),
            width=width_element,
        )
        self.spinbox_n_orientations.grid(row=0, column=1, sticky="w")
        self.spinbox_n_orientations.set(self.data.n_orientations)
        # <- orientations

        # scale_kernel ->
        self.label_scale_kernel = ttk.Label(
            self.frame, text="Scale sigma [DVA]", width=width_label
        )
        self.label_scale_kernel.grid(row=1, column=0, sticky="w")

        self.string_var_scale_kernel = tk.StringVar(
            value=f"{self.data.sigma_kernel_DVA}"
        )

        self.entry_scale_kernel = ttk.Entry(
            self.frame, textvariable=self.string_var_scale_kernel, width=width_element
        )
        self.entry_scale_kernel.grid(row=1, column=1, sticky="w")
        # <- scale_kernel

        # button_configure ->
        self.button_configure = ttk.Button(
            self.frame,
            text="Configure",
            command=self.button_configure_pressed,
            width=(width_label + width_element + width_button_extra),
        )
        self.button_configure.grid(row=2, column=0, sticky="w", columnspan=2, pady=5)
        # <- button_configure

    def button_configure_pressed(self) -> None:
        self.data.n_orientations = int(self.spinbox_n_orientations.get())
        self.data.sigma_kernel_DVA = float(self.entry_scale_kernel.get())
        # self.data.calculate_setting()
        self.data.data_changed = True
