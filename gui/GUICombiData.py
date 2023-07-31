from gui.gui_yolo_class import GUIYoloClassData
from gui.gui_contour_extraction import GUIContourExtractionData
from gui.gui_alphabet import GUIAlphabetData
from gui.gui_outputmode import GUIOutputModeData
from gui.gui_sparsifier import GUISparsifierData


class GUICombiData:
    def __init__(self) -> None:
        self.yolo_class = GUIYoloClassData()
        self.contour_extraction = GUIContourExtractionData()
        self.alphabet = GUIAlphabetData()
        self.output_mode = GUIOutputModeData()
        self.sparsifier = GUISparsifierData()

        self.gui_running: bool = True

    def update(self, input) -> None:
        self.yolo_class.update(input.yolo_class)
        self.contour_extraction.update(input.contour_extraction)
        self.alphabet.update(input.alphabet)
        self.output_mode.update(input.output_mode)
        self.sparsifier.update(input.sparsifier)

    def check_for_change(self) -> bool:
        if self.yolo_class.data_changed is True:
            return True
        if self.contour_extraction.data_changed is True:
            return True
        if self.alphabet.data_changed is True:
            return True
        if self.output_mode.data_changed is True:
            return True
        if self.sparsifier.data_changed is True:
            return True

        return False

    def reset_change_detector(self) -> None:
        self.yolo_class.data_changed = False
        self.contour_extraction.data_changed = False
        self.alphabet.data_changed = False
        self.output_mode.data_changed = False
        self.sparsifier.data_changed = False
