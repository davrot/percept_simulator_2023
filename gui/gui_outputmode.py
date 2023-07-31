from gui.GUIMasterData import GUIMasterData
from gui.GUIMasterGUI import GUIMasterGUI
import tkinter as tk
from tkinter import ttk


class GUIOutputModeData(GUIMasterData):

    enable_cam: bool = True
    enable_yolo: bool = True
    enable_contour: bool = True
    enable_percept: bool = True


class GUIOutputModeGUI(GUIMasterGUI):
    def __init__(
        self,
        tk_root: tk.Tk | tk.ttk.Labelframe | tk.ttk.Frame,
        name: str = "Output Filter",
        row_id: int = 0,
        column_id: int = 3,
        data_class=None,
    ) -> None:
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

        # option0 ->
        self.option0_var = tk.IntVar()
        self.option0_var.set(int(self.data.enable_cam))
        self.option0 = ttk.Checkbutton(
            self.frame,
            text="CAM",
            onvalue=1,
            offvalue=0,
            variable=self.option0_var,
            command=self.selection_changed,
            width=(width_label + width_element + width_button_extra),
        )
        self.option0.pack(anchor=tk.W)
        # <- option0

        # option1 ->
        self.option1_var = tk.IntVar()
        self.option1_var.set(int(self.data.enable_yolo))
        self.option1 = ttk.Checkbutton(
            self.frame,
            text="YOLO",
            onvalue=1,
            offvalue=0,
            variable=self.option1_var,
            command=self.selection_changed,
            width=(width_label + width_element + width_button_extra),
        )
        self.option1.pack(anchor=tk.W)
        # <- option1

        # option2 ->bool(self.option0_var.get())
        self.option2_var = tk.IntVar()
        self.option2_var.set(int(self.data.enable_contour))
        self.option2 = ttk.Checkbutton(
            self.frame,
            text="Contour",
            onvalue=1,
            offvalue=0,
            variable=self.option2_var,
            command=self.selection_changed,
            width=(width_label + width_element + width_button_extra),
        )
        self.option2.pack(anchor=tk.W)
        # <- option2

        # option3 ->
        self.option3_var = tk.IntVar()
        self.option3_var.set(int(self.data.enable_percept))
        self.option3 = ttk.Checkbutton(
            self.frame,
            text="Percept",
            onvalue=1,
            offvalue=0,
            variable=self.option3_var,
            command=self.selection_changed,
            width=(width_label + width_element + width_button_extra),
        )
        self.option3.pack(anchor=tk.W)
        # <- option3

    def selection_changed(self):
        self.data.enable_cam = bool(self.option0_var.get())
        self.data.enable_yolo = bool(self.option1_var.get())
        self.data.enable_contour = bool(self.option2_var.get())
        self.data.enable_percept = bool(self.option3_var.get())

        self.data.data_changed = True
