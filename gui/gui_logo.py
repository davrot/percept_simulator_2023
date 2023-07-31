import tkinter as tk

from PIL import Image, ImageTk
import os


class GUILogoGUI:
    logo: tk.Canvas
    logo_image: int
    my_tk_root: tk.Tk

    pic_path: str = os.path.join("gui", "logo")

    def __init__(self, tk_root: tk.Tk):
        self.my_tk_root = tk_root

        logo_filename: str = os.path.join(self.pic_path, "ISee2.png")
        pil_image = Image.open(logo_filename)
        self.pil_imagetk = ImageTk.PhotoImage(pil_image)
        canvas_width: int = pil_image.width
        canvas_height: int = pil_image.height

        self.logo = tk.Canvas(self.my_tk_root, width=canvas_width, height=canvas_height)
        self.logo.pack()

        self.logo_image = self.logo.create_image(
            0,
            0,
            anchor=tk.NW,
            image=self.pil_imagetk,
        )
