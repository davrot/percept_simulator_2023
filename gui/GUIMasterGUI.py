import tkinter as tk
from tkinter import ttk


class GUIMasterGUI:
    my_tk_root: tk.Tk | tk.ttk.Labelframe | tk.ttk.Frame

    verbose: bool = True

    def __init__(
        self,
        tk_root: tk.Tk | tk.ttk.Labelframe | tk.ttk.Frame,
        name: str,
        row_id: int,
        column_id: int,
        data_class=None,
    ):
        assert data_class is not None
        super().__init__()

        self.my_tk_root = tk_root
        self.data = data_class

        self.frame = ttk.LabelFrame(self.my_tk_root, text=name)
        self.frame.grid(row=row_id, column=column_id, sticky="nw", pady=5)
