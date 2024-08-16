# %%
#
# offline_encoding.py
# ========================================================
# encode visual scenes into sparse representations using
# different kinds of dictionaries
#
# -> derived from OnlineEncoding.py
#
# Version 1.0, 16.04.2024:
#


# Import Python modules
# ========================================================
# import csv
# import time
# import os
# import glob
import matplotlib.pyplot as plt
import torch
import torchvision as tv  # type:ignore
# from PIL import Image
import cv2
import numpy as np
import json
from jsmin import jsmin  # type:ignore


# Import our modules
# ========================================================
from processing_chain.ContourExtract import ContourExtract
from processing_chain.PatchGenerator import PatchGenerator
from processing_chain.Sparsifier import Sparsifier
# from processing_chain.DiscardElements import discard_elements_simple
from processing_chain.BuildImage import BuildImage
# from processing_chain.WebCam import WebCam
# from processing_chain.Yolo5Segmentation import Yolo5Segmentation


class OfflineEncoding:

    # INPUT PARAMETERS
    config: dict

    # DERIVED PARAMETERS
    default_dtype: torch.dtype
    torch_device: str
    display_size_max_x_pix: float
    display_size_max_y_pix: float
    # padding_fill: float
    # DEFINED PREVIOUSLY IN "apply_parameter_changes":
    padding_pix: int
    sigma_kernel_pix: float
    lambda_kernel_pix: float
    out_x: int
    out_y: int
    clocks: torch.Tensor
    phosphene: torch.Tensor
    clocks_filter: torch.Tensor

    # DELIVERED BY ENCODING
    position_found: None | torch.Tensor
    canvas_size: None | torch.Tensor

    def __init__(self, config="config.json"):

        # Define parameters
        # ========================================================
        print("OffE-Init: Loading configuration parameters...")
        with open(config, "r") as file:
            config = json.loads(jsmin(file.read()))

        # store in class
        self.config = config
        self.position_found = None
        self.canvas_size = None

        # get sub-dicts for easier access
        display = self.config["display"]
        dictionary = self.config["dictionary"]
        gabor = self.config["gabor"]

        # print(
        #     "OE-Init: Defining paths, creating dirs, setting default device and datatype"
        # )
        # self.path = {"output": "test/output/level1/", "input": "test/images_test/"}
        # Make output directories, if necessary: the place were we dump the new images to...
        # os.makedirs(self.path["output"], mode=0o777, exist_ok=True)

        # Check if GPU is available and use it, if possible
        # =================================================
        self.default_dtype = torch.float32
        torch.set_default_dtype(self.default_dtype)
        if self.config["control"]["force_torch_use_cpu"]:
            torch_device = "cpu"
        else:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {torch_device} as TORCH device...")
        self.torch_device = torch_device

        print("OffE-Init: Compute display scaling factors and padding RGB values")

        # global scaling factors for all pixel-related length scales
        self.display_size_max_x_pix = (
            display["size_max_x_dva"] * display["pix_per_dva"]
        )
        self.display_size_max_y_pix = (
            display["size_max_y_dva"] * display["pix_per_dva"]
        )

        # determine padding fill value
        tmp = tv.transforms.Grayscale(num_output_channels=1)
        tmp_value = torch.full((3, 1, 1), 254.0/255)
        self.padding_fill = int(tmp(tmp_value).squeeze())

        # PREVIOUSLY, A SEPARATE ROUTINE APPLIED PARAMETER CHANGES
        # WE DISCARD THIS HERE BUT KEEP THE CODE AS EXAMPLE
        #
        # self.apply_parameter_changes()
        # return
    #
    # def apply_parameter_changes(self):
        #
        # GET NEW PARAMETERS
        print("OffE-Init: Computing image/patch sizes from parameters")

        # BLOCK: dictionary ----------------
        # set patch size for both dictionaries, make sure it is odd number
        dictionary_size_pix = (
            1
            + (int(dictionary["size_dva"] *
                   display["pix_per_dva"]) // 2) * 2
        )

        # BLOCK: gabor ---------------------
        # convert contour-related parameters to pixel units
        self.sigma_kernel_pix = (
            gabor["sigma_kernel_dva"] *
            display["pix_per_dva"]
        )
        self.lambda_kernel_pix = (
            gabor["lambda_kernel_dva"] *
            display["pix_per_dva"]
        )

        # BLOCK: gabor & dictionary ------------------
        # Padding
        # -------
        self.padding_pix = int(
            max(3.0 * self.sigma_kernel_pix, 1.1 * dictionary_size_pix)
        )

        # define target video/representation width/height
        multiple_of = 4
        out_x = self.display_size_max_x_pix + 2 * self.padding_pix
        out_y = self.display_size_max_y_pix + 2 * self.padding_pix
        out_x += (multiple_of - (out_x % multiple_of)) % multiple_of
        out_y += (multiple_of - (out_y % multiple_of)) % multiple_of
        self.out_x = int(out_x)
        self.out_y = int(out_y)

        # generate dictionaries
        # ---------------------
        # BLOCK: dictionary --------------------------
        print("OffE-Init: Generating dictionaries...")
        patch_generator = PatchGenerator(torch_device=self.torch_device)
        self.phosphene = patch_generator.alphabet_phosphene(
            patch_size=dictionary_size_pix,
            sigma_width=dictionary["phosphene"]["sigma_width"]
            * dictionary_size_pix,
        )
        # BLOCK: dictionary & gabor --------------------------
        self.clocks_filter, self.clocks, segments = patch_generator.alphabet_clocks(
            patch_size=dictionary_size_pix,
            n_dir=dictionary["clocks"]["n_dir"],
            n_filter=gabor["n_orientations"],
            segment_width=dictionary["clocks"]["pointer_width"]
            * dictionary_size_pix,
            segment_length=dictionary["clocks"]["pointer_length"]
            * dictionary_size_pix,
        )

        return

    # TODO image supposed to be torch.Tensor(3, Y, X) within 0...1
    def encode(self, image: torch.Tensor, number_of_patches: int = 42, border_pixel_value: float = 254.0 / 255) -> dict:

        assert len(image.shape) == 3, "Input image must be RGB (3 dimensions)!"
        assert image.shape[0] == 3, "Input image format must be (3, HEIGHT, WIDTH)!"
        control = self.config["control"]

        
        # determine padding fill value
        tmp = tv.transforms.Grayscale(num_output_channels=1)
        tmp_value = torch.full((3, 1, 1), border_pixel_value)
        padding_fill = float(tmp(tmp_value).squeeze())

        # show input image, if desired...
        if control["show_image"]:
            self.__show_torch_frame(
                image,
                title="Encode: Input Image",
                target=control["show_mode"]
            )

        # some constants for addressing specific components of output arrays
        image_id_const: int = 0
        overlap_index_const: int = 1

        # Determine target size of image
        # image: [RGB, Height, Width], dtype= tensor.torch.uint8
        print("OffE-Encode: Computing downsampling factor image -> display")
        f_x: float = self.display_size_max_x_pix / image.shape[-1]
        f_y: float = self.display_size_max_y_pix / image.shape[-2]
        f_xy_min: float = min(f_x, f_y)
        downsampling_x: int = int(f_xy_min * image.shape[-1])
        downsampling_y: int = int(f_xy_min * image.shape[-2])

        # CURRENTLY we do not crop in the end...
        # Image size for removing the fft crop later
        # center_crop_x: int = downsampling_x
        # center_crop_y: int = downsampling_y

        # define contour extraction processing chain
        # ------------------------------------------
        print("OffE-Encode: Extracting contours")
        train_processing_chain = tv.transforms.Compose(
            transforms=[
                tv.transforms.Grayscale(num_output_channels=1),  # RGB to grayscale
                tv.transforms.Resize(
                    size=(downsampling_y, downsampling_x)
                ),  # downsampling
                tv.transforms.Pad(  # extra white padding around the picture
                    padding=(self.padding_pix, self.padding_pix),
                    fill=padding_fill,
                ),
                ContourExtract(  # contour extraction
                    n_orientations=self.config["gabor"]["n_orientations"],
                    sigma_kernel=self.sigma_kernel_pix,
                    lambda_kernel=self.lambda_kernel_pix,
                    torch_device=self.torch_device,
                ),
                # CURRENTLY we do not crop in the end!
                # tv.transforms.CenterCrop(  # Remove the padding
                #     size=(center_crop_x, center_crop_y)
                # ),
            ],
        )
        # ...with and without orientation channels
        contour = train_processing_chain(image.unsqueeze(0))
        contour_collapse = train_processing_chain.transforms[-1].create_collapse(
            contour
        )

        if control["show_contours"]:
            self.__show_torch_frame(
                contour_collapse,
                title="Encode: Contours Extracted",
                cmap="gray",
                target=control["show_mode"],
            )

        # generate a prior for mapping the contour to the dictionary
        # CURRENTLY we use an uniform prior...
        # ----------------------------------------------------------
        dictionary_prior = torch.ones(
            (self.clocks_filter.shape[0]),
            dtype=self.default_dtype,
            device=torch.device(self.torch_device),
        )

        # instantiate and execute sparsifier
        # ----------------------------------
        print("OffE-Encode: Performing sparsification")
        encoding = self.config["encoding"]
        display = self.config["display"]
        sparsifier = Sparsifier(
            dictionary_filter=self.clocks_filter,
            dictionary=self.clocks,
            dictionary_prior=dictionary_prior,
            number_of_patches=encoding["n_patches_compute"],
            size_exp_deadzone=encoding["size_exp_deadzone_dva"]
            * display["pix_per_dva"],
            plot_use_map=False,  # self.control["plot_deadzone"],
            deadzone_exp=encoding["use_exp_deadzone"],
            deadzone_hard_cutout=encoding["use_cutout_deadzone"],
            deadzone_hard_cutout_size=encoding["size_cutout_deadzone_dva"]
            * display["pix_per_dva"],
            padding_deadzone_size_x=self.padding_pix,
            padding_deadzone_size_y=self.padding_pix,
            torch_device=self.torch_device,
        )
        sparsifier(contour)
        assert sparsifier.position_found is not None

        # extract and normalize the overlap found
        overlap_found = sparsifier.overlap_found[
            image_id_const, :, overlap_index_const
        ]
        overlap_found = overlap_found / overlap_found.max()

        # get overlap above certain threshold, extract corresponding elements
        overlap_idcs_valid = torch.where(
            overlap_found >= encoding["overlap_threshold"]
        )[0]
        position_selection = sparsifier.position_found[
            image_id_const : image_id_const + 1, overlap_idcs_valid, :
        ]
        n_elements = len(overlap_idcs_valid)
        print(f"OffE-Encode: {n_elements} elements positioned!")

        contour_shape = contour.shape

        n_cut = min(position_selection.shape[-2], number_of_patches)

        data_out = {
            "position_found": position_selection[:, :n_cut, :],
            "canvas_size": contour_shape,
        }

        self.position_found = data_out["position_found"]
        self.canvas_size = data_out["canvas_size"]

        return data_out

    def render(self):

        assert self.position_found is not None, "Use ""encode"" before rendering!"
        assert self.canvas_size is not None, "Use ""encode"" before rendering!"

        control = self.config["control"]

        # build the full image!
        image_clocks = BuildImage(
            canvas_size=self.canvas_size,
            dictionary=self.clocks,
            position_found=self.position_found,
            default_dtype=self.default_dtype,
            torch_device=self.torch_device,
        )

        # normalize to range [0...1]
        m = image_clocks[0].max()
        if m == 0:
            m = 1
        image_clocks_normalized = image_clocks[0] / m

        # embed into frame of desired output size
        out_torch = self.__embed_image(
            image_clocks_normalized, out_height=self.out_y, out_width=self.out_x
        )

        # show, if desired...    
        if control["show_percept"]:
            self.__show_torch_frame(
                out_torch, title="Percept",
                cmap="gray", target=control["show_mode"]
            )

        return

    def __show_torch_frame(self,
        frame_torch: torch.Tensor,
        title: str = "default",
        cmap: str = "viridis",
        target: str = "pyplot",
    ):
        frame_numpy = (
            (frame_torch.movedim(0, -1) * 255).type(dtype=torch.uint8).cpu().numpy()
        )
        if target == "pyplot":
            plt.imshow(frame_numpy, cmap=cmap)
            plt.title(title)
            plt.show()
        if target == "cv2":
            if frame_numpy.ndim == 3:
                if frame_numpy.shape[-1] == 1:
                    frame_numpy = np.tile(frame_numpy, [1, 1, 3])
                    frame_numpy = (frame_numpy - frame_numpy.min()) / (
                        frame_numpy.max() - frame_numpy.min()
                    )
            # print(frame_numpy.shape, frame_numpy.max(), frame_numpy.min())
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, frame_numpy[:, :, (2, 1, 0)])
            cv2.waitKey(1)

        return

    def __embed_image(self, frame_torch, out_height, out_width, init_value=0):

        out_shape = torch.tensor(frame_torch.shape)

        frame_width = frame_torch.shape[-1]
        frame_height = frame_torch.shape[-2]

        frame_width_idx0 = max([0, (frame_width - out_width) // 2])
        frame_height_idx0 = max([0, (frame_height - out_height) // 2])

        select_width = min([frame_width, out_width])
        select_height = min([frame_height, out_height])

        out_shape[-1] = out_width
        out_shape[-2] = out_height

        out_torch = init_value * torch.ones(tuple(out_shape))

        out_width_idx0 = max([0, (out_width - frame_width) // 2])
        out_height_idx0 = max([0, (out_height - frame_height) // 2])

        out_torch[
            ...,
            out_height_idx0: (out_height_idx0 + select_height),
            out_width_idx0: (out_width_idx0 + select_width),
        ] = frame_torch[
            ...,
            frame_height_idx0: (frame_height_idx0 + select_height),
            frame_width_idx0: (frame_width_idx0 + select_width),
        ]

        return out_torch

    def __del__(self):

        print("OffE-Delete: exiting gracefully!")
        # TODO ...only do it when necessary
        cv2.destroyAllWindows()

        return


if __name__ == "__main__":

    source = 'bernd.jpg'
    img_cv2 = cv2.imread(source)
    img_torch = torch.Tensor(img_cv2[:, :, (2, 1, 0)]).movedim(-1, 0) / 255
    # show_torch_frame(img_torch, target="cv2", title=source)
    print(f"CV2 Shape: {img_cv2.shape}")
    print(f"Torch Shape: {img_torch.shape}")

    img = img_torch
    frame_width = img.shape[-1]
    frame_height = img.shape[-2]
    print(
        f"OffE-Test: Processing image {source} of {frame_width} x {frame_height}."
    )

    # TEST  tfg = tv.transforms.Grayscale(num_output_channels=1)
    # TEST  pixel_fill = torch.full((3, 1, 1), 254.0 / 255)
    # TEST  value_fill = float(tfg(pixel_fill).squeeze())
    # TEST  tfp = tv.transforms.Pad(padding=(1, 1), fill=value_fill)

    # TEST  img_gray = tfg(img[:, :3, :3])
    # TEST  img_pad = tfp(img_gray)

    oe = OfflineEncoding()
    encoding = oe.encode(img)
    stimulus = oe.render()
    if oe.config["control"]["show_mode"] == "cv2":
        cv2.waitKey(5000)
    del oe

# %%
