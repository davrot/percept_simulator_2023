from gui.GUIMasterData import GUIMasterData
from gui.GUIMasterGUI import GUIMasterGUI
import tkinter as tk
from tkinter import ttk


class GUIAlphabetData(GUIMasterData):
    phosphene_sigma_width: float
    size_DVA: float
    clocks_n_dir: int
    clocks_pointer_width: float
    clocks_pointer_length: float
    selection: int = 0

    tau_SEC: float = 0.3
    p_FEATperSECperPOS: float = 3.0

    def __init__(self) -> None:
        super().__init__()

        self.phosphene_sigma_width = 0.18
        self.size_DVA = 1.0

        self.clocks_n_dir = 8
        self.clocks_pointer_width = 0.07
        self.clocks_pointer_length = 0.18


class GUIAlphabetGUI(GUIMasterGUI):
    def __init__(
        self,
        tk_root: tk.Tk | tk.ttk.Labelframe | tk.ttk.Frame,
        name: str = "Alphabet",
        row_id: int = 0,
        column_id: int = 2,
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

        # patch_size ->
        self.label_size_DVA = ttk.Label(
            self.frame, text="Patch size [DVA]", width=width_label
        )
        self.label_size_DVA.grid(row=0, column=0, sticky="nw")

        self.entry_string_var_size_DVA = tk.StringVar(
            value=f"{self.data.size_DVA}",
        )

        self.entry_size_DVA = ttk.Entry(
            self.frame,
            textvariable=self.entry_string_var_size_DVA,
            width=width_element,
        )
        self.entry_size_DVA.grid(row=0, column=1, sticky="nw")
        # <- patch_size

        # tau_SEC ->
        self.label_tau_SEC = ttk.Label(
            self.frame, text="Patch decay [s]", width=width_label
        )
        self.label_tau_SEC.grid(row=1, column=0, sticky="nw")

        self.entry_string_var_tau_SEC = tk.StringVar(
            value=f"{self.data.tau_SEC}",
        )

        self.entry_tau_SEC = ttk.Entry(
            self.frame,
            textvariable=self.entry_string_var_tau_SEC,
            width=width_element,
        )
        self.entry_tau_SEC.grid(row=1, column=1, sticky="nw")
        # <- tau_SEC

        # p_FEATperSECperPOS ->
        self.label_p_FEATperSECperPOS = ttk.Label(
            self.frame, text="Patch prob [1/s pos]", width=width_label
        )
        self.label_p_FEATperSECperPOS.grid(row=2, column=0, sticky="nw", pady=[0, 15])

        self.entry_string_var_p_FEATperSECperPOS = tk.StringVar(
            value=f"{self.data.p_FEATperSECperPOS}",
        )

        self.entry_p_FEATperSECperPOS = ttk.Entry(
            self.frame,
            textvariable=self.entry_string_var_p_FEATperSECperPOS,
            width=width_element,
        )
        self.entry_p_FEATperSECperPOS.grid(row=2, column=1, sticky="nw", pady=[0, 15])
        # <- p_FEATperSECperPOS

        # Phosphene vs Clock ->
        self.radio_selection_var = tk.IntVar()
        self.radio_selection_var.set(self.data.selection)

        self.option0 = ttk.Radiobutton(
            self.frame,
            text="Phosphene",
            variable=self.radio_selection_var,
            value=0,
        )
        self.option0.grid(row=3, column=0, sticky="w", pady=5)

        # sigma_width ->
        self.label_sigma_width = ttk.Label(
            self.frame, text="Sigma Width", width=width_label
        )
        self.label_sigma_width.grid(row=4, column=0, sticky="w")

        self.entry_string_var_sigma_width = tk.StringVar(
            value=f"{self.data.phosphene_sigma_width}",
        )

        self.entry_sigma_width = ttk.Entry(
            self.frame,
            textvariable=self.entry_string_var_sigma_width,
            width=width_element,
        )
        self.entry_sigma_width.grid(row=4, column=1, sticky="w")
        # <- sigma_width

        self.option1 = ttk.Radiobutton(
            self.frame,
            text="Clock",
            variable=self.radio_selection_var,
            value=1,
        )
        self.option1.grid(row=5, column=0, sticky="w", pady=[15, 0])
        # <- Phosphene vs Clock

        # clocks_n_dir ->
        self.label_clocks_n_dir = ttk.Label(
            self.frame, text="Number Orientations", width=width_label
        )
        self.label_clocks_n_dir.grid(row=6, column=0, sticky="w")

        self.spinbox_clocks_n_dir = ttk.Spinbox(
            self.frame,
            values=list("{:d}".format(x) for x in range(1, 33)),
            width=width_element,
        )
        self.spinbox_clocks_n_dir.grid(row=6, column=1, sticky="w")
        self.spinbox_clocks_n_dir.set(self.data.clocks_n_dir)
        # <- clocks_n_dir

        # clocks_pointer_width ->
        self.label_clocks_pointer_width = ttk.Label(
            self.frame, text="Pointer Width", width=width_label
        )
        self.label_clocks_pointer_width.grid(row=7, column=0, sticky="w")

        self.entry_string_var_clocks_pointer_width = tk.StringVar(
            value=f"{self.data.clocks_pointer_width}"
        )
        self.entry_clocks_pointer_width = ttk.Entry(
            self.frame,
            textvariable=self.entry_string_var_clocks_pointer_width,
            width=width_element,
        )
        self.entry_clocks_pointer_width.grid(row=7, column=1, sticky="w")
        # <- clocks_pointer_width

        # clocks_pointer_length ->
        self.label_clocks_pointer_length = ttk.Label(
            self.frame, text="Pointer Length", width=width_label
        )
        self.label_clocks_pointer_length.grid(row=8, column=0, sticky="w")

        self.entry_string_var_clocks_pointer_length = tk.StringVar(
            value=f"{self.data.clocks_pointer_length}"
        )
        self.entry_clocks_pointer_length = ttk.Entry(
            self.frame,
            textvariable=self.entry_string_var_clocks_pointer_length,
            width=width_element,
        )
        self.entry_clocks_pointer_length.grid(row=8, column=1, sticky="w")
        # <- clocks_pointer_length

        # button_configure ->
        self.button_configure = ttk.Button(
            self.frame,
            text="Configure",
            command=self.button_configure_pressed,
            width=(width_label + width_element + width_button_extra),
        )
        self.button_configure.grid(row=9, column=0, sticky="w", columnspan=2, pady=5)

        # <- button_configure

    def button_configure_pressed(self) -> None:
        self.data.phosphene_sigma_width = float(self.entry_sigma_width.get())

        self.data.size_DVA = float(self.entry_size_DVA.get())
        self.data.tau_SEC = float(self.entry_tau_SEC.get())
        self.data.p_FEATperSECperPOS = float(self.entry_p_FEATperSECperPOS.get())

        self.data.clocks_n_dir = int(self.spinbox_clocks_n_dir.get())
        self.data.clocks_pointer_width = float(self.entry_clocks_pointer_width.get())
        self.data.clocks_pointer_length = float(self.entry_clocks_pointer_length.get())

        self.data.selection = int(self.radio_selection_var.get())
        self.data.data_changed = True
