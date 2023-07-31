#%%
import cv2
import numpy as np
import time

from communication.communicate_datapacket import DataPacket


class TestClassAnimate:

    dx: int = 400
    dy: int = 400

    def __init__(
        self,
        show_vertical_bar: bool = False,
        show_horizontal_bar: bool = False,
        wait_interval: float = 5.0,
    ):
        self.show_vertical_bar = show_vertical_bar
        self.show_horizontal_bar = show_horizontal_bar
        self.count_vertical_bar = 0
        self.count_horizontal_bar = 0
        self.wait_interval = wait_interval

    def update(self, data_in: dict):

        if self.show_vertical_bar:
            self.count_vertical_bar = np.mod(self.count_vertical_bar + 1, self.dx)
            vertical_image = np.zeros((self.dy, self.dx, 3))
            vertical_image[:, self.count_vertical_bar, 0] = 1.0
            cv2.imshow("Vertical Bar", vertical_image)
            cv2.waitKey(1)

        if self.show_horizontal_bar:
            self.count_horizontal_bar = np.mod(self.count_horizontal_bar + 2, self.dx)
            horizontal_image = np.zeros((self.dy, self.dx, 3))
            horizontal_image[self.count_horizontal_bar, :, 2] = 1.0
            cv2.imshow("Horizontal Bar", horizontal_image)
            cv2.waitKey(1)

        print(f"Waiting for {self.wait_interval} seconds...")
        time.sleep(self.wait_interval)

        print(f"Input data was: {data_in}")
        print("Generating output data...")
        data_out = {
            "CountVert": self.count_vertical_bar,
            "CountHori": self.count_horizontal_bar,
        }

        return data_out

    def __del__(self):
        if self.show_horizontal_bar or self.show_vertical_bar:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    tcas = TestClassAnimate(wait_interval=0.5, show_horizontal_bar=True)
    for i in range(50):
        data_out = tcas.update(None)
        print(f"Iteration {i}, data is: {data_out}")
    del tcas

# %%
