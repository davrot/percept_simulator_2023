import tkinter as tk
from tkinter import ttk

from gui.gui_logo import GUILogoGUI
from gui.gui_yolo_class import GUIYoloClassGUI
from gui.gui_contour_extraction import GUIContourExtractionGUI
from gui.gui_alphabet import GUIAlphabetGUI
from gui.gui_outputmode import GUIOutputModeGUI
from gui.gui_sparsifier import GUISparsifierGUI
from gui.GUICombiData import GUICombiData


class GUIEvents:
    tk_root: tk.Tk
    exit_button: tk.ttk.Button

    def __init__(self, tk_root: tk.Tk, confdata: GUICombiData | None = None):
        super().__init__()
        assert confdata is not None

        width_element: int = 10
        width_label: int = 20
        width_button_extra: int = 5

        self.tk_root = tk_root
        self.tk_root.title("Percept Simulator 2023")

        self.confdata: GUICombiData = confdata
        self.confdata.gui_running = True

        self.logo = GUILogoGUI(self.tk_root)

        # frame ->
        self.frame = ttk.Frame(self.tk_root)
        self.frame.pack()
        # <- frame

        self.frame_left = ttk.Frame(self.frame)
        self.frame_left.grid(row=0, column=0, sticky="nw")

        self.frame_right = ttk.Frame(self.frame)
        self.frame_right.grid(row=0, column=1, sticky="nw")

        # gui_element_list ->
        self.gui_element_list: list = []

        self.gui_element_list.append(
            GUIContourExtractionGUI(
                self.frame_left,
                name="Contour Extraction",
                row_id=0,
                column_id=0,
                data_class=self.confdata.contour_extraction,
            )
        )

        self.gui_element_list.append(
            GUIOutputModeGUI(
                self.frame_left,
                name="Display Options",
                row_id=1,
                column_id=0,
                data_class=self.confdata.output_mode,
            )
        )

        self.gui_element_list.append(
            GUISparsifierGUI(
                self.frame_left,
                name="Sparsifier Options",
                row_id=2,
                column_id=0,
                data_class=self.confdata.sparsifier,
            )
        )

        self.gui_element_list.append(
            GUIAlphabetGUI(
                self.frame_right,
                name="Alphabet",
                row_id=0,
                column_id=0,
                data_class=self.confdata.alphabet,
            )
        )

        self.gui_element_list.append(
            GUIYoloClassGUI(
                self.frame_right,
                name="Yolo",
                row_id=1,
                column_id=0,
                data_class=self.confdata.yolo_class,
            )
        )

        # <- gui_element_list

        self.exit_button = ttk.Button(
            self.frame,
            text="--> EXIT <--",
            command=self.exit_button_pressed,
            width=2 * (width_label + width_element + width_button_extra),
        )
        self.exit_button.grid(row=2, column=0, sticky="we", columnspan=2, pady=5)

        # windows close button ->
        self.tk_root.protocol("WM_DELETE_WINDOW", self.exit_button_pressed)
        # <- windows close button

    def exit_button_pressed(self) -> None:
        self.confdata.gui_running = False
        self.tk_root.destroy()
