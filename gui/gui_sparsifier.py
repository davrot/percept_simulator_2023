from gui.GUIMasterData import GUIMasterData
from gui.GUIMasterGUI import GUIMasterGUI
import tkinter as tk
from tkinter import ttk


class GUISparsifierData(GUIMasterData):

    number_of_patches: int = 10

    use_exp_deadzone: bool = True
    size_exp_deadzone_DVA: float = 1.0

    use_cutout_deadzone: bool = True
    size_cutout_deadzone_DVA: float = 1.0


class GUISparsifierGUI(GUIMasterGUI):
    def __init__(
        self,
        tk_root: tk.Tk | tk.ttk.Labelframe | tk.ttk.Frame,
        name: str = "Sparsifier Options",
        row_id: int = 0,
        column_id: int = 4,
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

        # number_of_patches ->
        self.label_number_of_patches = ttk.Label(
            self.frame, text="Number Patches", width=width_label
        )
        self.label_number_of_patches.grid(row=0, column=0, sticky="w")

        self.spinbox_number_of_patches = ttk.Spinbox(
            self.frame,
            values=list("{:d}".format(x) for x in range(1, 301)),
            width=width_element,
        )
        self.spinbox_number_of_patches.grid(row=0, column=1, sticky="w")
        self.spinbox_number_of_patches.set(self.data.number_of_patches)
        # <- number_of_patches

        self.label_forbidden_zone = ttk.Label(
            self.frame, text="Forbidden Zone:", width=width_label
        )
        self.label_forbidden_zone.grid(row=1, column=0, sticky="w", pady=[15, 0])
        # use_cutout_deadzone ->

        self.label_use_cutout_deadzone = ttk.Label(
            self.frame, text="Hard Circle", width=width_label
        )
        self.label_use_cutout_deadzone.grid(row=2, column=0, sticky="w")

        self.use_cutout_deadzone_var = tk.IntVar()
        self.checkbox_use_cutout_deadzone = ttk.Checkbutton(
            self.frame,
            text="Enable",
            onvalue=1,
            offvalue=0,
            variable=self.use_cutout_deadzone_var,
            width=width_element,
        )
        self.checkbox_use_cutout_deadzone.grid(
            row=2,
            column=1,
            sticky="w",
        )
        self.use_cutout_deadzone_var.set(int(self.data.use_cutout_deadzone))
        # <- use_cutout_deadzone

        # size_cutout_deadzone_DVA ->
        self.label_size_cutout_deadzone_DVA = ttk.Label(
            self.frame, text="Size Hard Circle", width=width_label
        )
        self.label_size_cutout_deadzone_DVA.grid(row=3, column=0, sticky="w")

        self.entry_size_cutout_deadzone_DVA_var = tk.StringVar()
        self.entry_size_cutout_deadzone_DVA_var.set(
            str(self.data.size_cutout_deadzone_DVA)
        )
        self.entry_size_cutout_deadzone_DVA = ttk.Entry(
            self.frame,
            textvariable=self.entry_size_cutout_deadzone_DVA_var,
            width=width_element,
        )
        self.entry_size_cutout_deadzone_DVA.grid(row=3, column=1, sticky="w")
        # <- size_cutout_deadzone_DVA

        # use_exp_deadzone ->

        self.label_use_exp_deadzone = ttk.Label(
            self.frame, text="Exponential Decay", width=width_label
        )
        self.label_use_exp_deadzone.grid(row=4, column=0, sticky="w")

        self.checkbox_use_exp_deadzone_var = tk.IntVar()
        self.checkbox_use_exp_deadzone = ttk.Checkbutton(
            self.frame,
            text="Enable",
            onvalue=1,
            offvalue=0,
            variable=self.checkbox_use_exp_deadzone_var,
            width=width_element,
        )
        self.checkbox_use_exp_deadzone.grid(row=4, column=1, sticky="w")
        self.checkbox_use_exp_deadzone_var.set(int(self.data.use_exp_deadzone))
        # <- use_exp_deadzone

        # size_exp_deadzone_DVA ->
        self.label_size_exp_deadzone_DVA = ttk.Label(
            self.frame, text="Size Exponential Decay", width=width_label
        )
        self.label_size_exp_deadzone_DVA.grid(row=5, column=0, sticky="w")

        self.entry_size_exp_deadzone_DVA_var = tk.StringVar()
        self.entry_size_exp_deadzone_DVA_var.set(str(self.data.size_exp_deadzone_DVA))
        self.entry_size_exp_deadzone_DVA = ttk.Entry(
            self.frame,
            textvariable=self.entry_size_exp_deadzone_DVA_var,
            width=width_element,
        )
        self.entry_size_exp_deadzone_DVA.grid(row=5, column=1, sticky="w")
        # <- size_exp_deadzone_DVA

        # button_configure ->
        self.button_configure = ttk.Button(
            self.frame,
            text="Configure",
            command=self.button_configure_pressed,
            width=(width_label + width_element + width_button_extra),
        )
        self.button_configure.grid(row=6, column=0, sticky="w", columnspan=2, pady=5)
        # <- button_configure

    def button_configure_pressed(self):
        self.data.number_of_patches = int(self.spinbox_number_of_patches.get())
        self.data.use_cutout_deadzone = bool(self.use_cutout_deadzone_var.get())
        self.data.use_exp_deadzone = bool(self.checkbox_use_exp_deadzone_var.get())
        self.data.size_cutout_deadzone_DVA = float(
            self.entry_size_cutout_deadzone_DVA_var.get()
        )
        self.data.size_exp_deadzone_DVA = float(
            self.entry_size_exp_deadzone_DVA_var.get()
        )
        self.data.data_changed = True
