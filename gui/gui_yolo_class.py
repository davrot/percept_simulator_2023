from gui.GUIMasterData import GUIMasterData
from gui.GUIMasterGUI import GUIMasterGUI
import tkinter as tk
from tkinter import ttk


class GUIYoloClassData(GUIMasterData):
    default_value: str = "--: None"

    available_classes: list[str] = [
        "--: None",
        "00: person",
        "01: bicycle",
        "02: car",
        "03: motorcycle",
        "04: airplane",
        "05: bus",
        "06: train",
        "07: truck",
        "08: boat",
        "09: traffic light",
        "10: fire hydrant",
        "11: stop sign",
        "12: parking meter",
        "13: bench",
        "14: bird",
        "15: cat",
        "16: dog",
        "17: horse",
        "18: sheep",
        "19: cow",
        "20: elephant",
        "21: bear",
        "22: zebra",
        "23: giraffe",
        "24: backpack",
        "25: umbrella",
        "26: handbag",
        "27: tie",
        "28: suitcase",
        "29: frisbee",
        "30: skis",
        "31: snowboard",
        "32: sports ball",
        "33: kite",
        "34: baseball bat",
        "35: baseball glove",
        "36: skateboard",
        "37: surfboard",
        "38: tennis racket",
        "39: bottle",
        "40: wine glass",
        "41: cup",
        "42: fork",
        "43: knife",
        "44: spoon",
        "45: bowl",
        "46: banana",
        "47: apple",
        "48: sandwich",
        "49: orange",
        "50: broccoli",
        "51: carrot",
        "52: hot dog",
        "53: pizza",
        "54: donut",
        "55: cake",
        "56: chair",
        "57: couch",
        "58: potted plant",
        "59: bed",
        "60: dining table",
        "61: toilet",
        "62: tv",
        "63: laptop",
        "64: mouse",
        "65: remote",
        "66: keyboard",
        "67: cell phone",
        "68: microwave",
        "69: oven",
        "70: toaster",
        "71: sink",
        "72: refrigerator",
        "73: book",
        "74: clock",
        "75: vase",
        "76: scissors",
        "77: teddy bear",
        "78: hair drier",
        "79: toothbrush",
    ]

    value: list[int] | None = None

    def __init__(self) -> None:
        super().__init__()
        self.do_not_update.append(str("default_value"))
        self.do_not_update.append(str("available_classes"))


class GUIYoloClassGUI(GUIMasterGUI):
    def __init__(
        self,
        tk_root: tk.Tk | tk.ttk.Labelframe | tk.ttk.Frame,
        name: str = "Yolo",
        row_id: int = 1,
        column_id: int = 0,
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

        self.selection: tk.ttk.Combobox = ttk.Combobox(
            self.frame, width=(width_label + width_element + width_button_extra)
        )
        self.selection["state"] = "readonly"
        self.selection["values"] = self.data.available_classes
        self.selection.pack()
        self.selection.bind("<<ComboboxSelected>>", self.selection_changed)
        self.selection.set(self.data.default_value)
        self.selection_value = None

    def selection_changed(self, event) -> None:
        temp: str = self.selection.get()
        if temp[0] == "-":
            self.data.value = None
        else:
            self.data.value = [int(temp[0:2])]

        if self.verbose is True:
            print(f"Class changed to: {self.data.value}")

        self.data.data_changed = True
