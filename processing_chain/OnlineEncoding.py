# %%
#
# test_OnlineEncoding.py
# ========================================================
# encode visual scenes into sparse representations using
# different kinds of dictionaries
#
# -> derived from test_PsychophysicsEncoding.py
#
# Version 1.0, 29.04.2023:
#
# Version 1.1, 21.06.2023:
#   define proper class
#


# Import Python modules
# ========================================================
# import csv
# import time
import os
import glob
import matplotlib.pyplot as plt
import torch
import torchvision as tv
from PIL import Image
import cv2
import numpy as np


# Import our modules
# ========================================================
from processing_chain.ContourExtract import ContourExtract
from processing_chain.PatchGenerator import PatchGenerator
from processing_chain.Sparsifier import Sparsifier
from processing_chain.DiscardElements import discard_elements_simple
from processing_chain.BuildImage import BuildImage
from processing_chain.WebCam import WebCam
from processing_chain.Yolo5Segmentation import Yolo5Segmentation


# TODO required?
def show_torch_frame(
    frame_torch: torch.Tensor,
    title: str = "",
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


# TODO required?
def embed_image(frame_torch, out_height, out_width, init_value=0):

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
        out_height_idx0 : (out_height_idx0 + select_height),
        out_width_idx0 : (out_width_idx0 + select_width),
    ] = frame_torch[
        ...,
        frame_height_idx0 : (frame_height_idx0 + select_height),
        frame_width_idx0 : (frame_width_idx0 + select_width),
    ]

    return out_torch


class OnlineEncoding:

    # TODO: also pre-populate self-ies here?
    #
    # DEFINED IN "__init__":
    #
    # display (fixed)
    # gabor (changeable)
    # encoding (changeable)
    # dictionary (changeable)
    # control (fixed)
    # path (fixed)
    # verbose
    # torch_device, default_dtype
    # display_size_max_x_PIX, display_size_max_y_PIX
    # padding_fill
    # cap
    # yolo
    # classes_detect
    #
    #
    # DEFINED IN "apply_parameter_changes":
    #
    # padding_PIX
    # sigma_kernel_PIX, lambda_kernel_PIX
    # out_x, out_y
    # clocks, phosphene, clocks_filter
    #

    def __init__(self, source=0, verbose=False):

        # Define parameters
        # ========================================================
        # Unit abbreviations:
        #   dva = degrees of visual angle
        #   pix = pixels
        print("OE-Init: Defining default parameters...")
        self.verbose = verbose

        # display: Defines geometry of target display
        # ========================================================
        # The encoded image will be scaled such that it optimally uses
        # the max space available. If the orignal image has a different aspect
        # ratio than the display region, it will only use one spatial
        # dimension (horizontal or vertical) to its full extent
        #
        # If one DVA corresponds to different PIX_per_DVA on the display,
        # (i.e. varying distance observers from screen), it should be set
        # larger than the largest PIX_per_DVA required, for avoiding
        # extrapolation artefacts or blur.
        #
        self.display = {
            "size_max_x_DVA": 10.0,  # maximum x size of encoded image
            "size_max_y_DVA": 10.0,  # minimum y size of encoded image
            "PIX_per_DVA": 40.0,  # scaling factor pixels to DVA
            "scale": "same_range",  # "same_luminance" or "same_range"
        }

        # gabor: Defines paras of Gabor filters for contour extraction
        # ==============================================================
        self.gabor = {
            "sigma_kernel_DVA": 0.06,
            "lambda_kernel_DVA": 0.12,
            "n_orientations": 8,
        }

        # encoding: Defines parameters of sparse encoding process
        # ========================================================
        # Roughly speaking, after contour extraction dictionary elements
        # will be placed starting from the position with the highest
        # overlap with the contour. Elements placed can be surrounded
        # by a dead or inhibitory zone to prevent placing further elements
        # too closely. The procedure will map 'n_patches_compute' elements
        # and then stop. For each element one obtains an overlap with the
        # contour image.
        #
        # After placement, the overlaps found are normalized to the max
        # overlap found, and then all elements with a larger normalized overlap
        # than 'overlap_threshold' will be selected. These remaining
        # elements will comprise a 'full' encoding of the contour.
        #
        # To generate even sparser representations, the full encoding can
        # be reduced to a certain percentage of elements in the full encoding
        # by setting the variable 'percentages'
        #
        # Example: n_patches_compute = 100 reduced by overlap_threshold = 0.1
        # to 80 elements. Requesting a percentage of 30% yields a representation
        # with 24 elements.
        #
        self.encoding = {
            "n_patches_compute": 100,  # this amount of patches will be placed
            "use_exp_deadzone": True,  # parameters of Gaussian deadzone
            "size_exp_deadzone_DVA": 1.20,  # PREVIOUSLY 1.4283
            "use_cutout_deadzone": True,  # parameters of cutout deadzone
            "size_cutout_deadzone_DVA": 0.65,  # PREVIOUSLY 0.7575
            "overlap_threshold": 0.1,  # relative overlap threshold
            "percentages": torch.tensor([100]),
        }
        self.number_of_patches = self.encoding["n_patches_compute"]

        # dictionary: Defines parameters of dictionary
        # ========================================================
        self.dictionary = {
            "size_DVA": 1.0,  # PREVIOUSLY 1.25,
            "clocks": None,  # parameters for clocks dictionary, see below
            "phosphene": None,  # paramters for phosphene dictionary, see below
        }

        self.dictionary["phosphene"]: dict[float] = {
            "sigma_width": 0.18,  # DEFAULT 0.15,  # half-width of Gaussian
        }

        self.dictionary["clocks"]: dict[int, int, float, float] = {
            "n_dir": 8,  # number of directions for clock pointer segments
            "n_open": 4,  # number of opening angles between two clock pointer segments
            "pointer_width": 0.07,  # PREVIOUSLY 0.05,  # relative width and size of tip extension of clock pointer
            "pointer_length": 0.18,  # PREVIOUSLY 0.15,  # relative length of clock pointer
        }

        # control: For controlling plotting options and flow of script
        # ========================================================
        self.control = {
            "force_torch_use_cpu": False,  # force using CPU even if GPU available
            "show_capture": True,  # shows captured image
            "show_object": True,  # shows detected object
            "show_contours": True,  # shows extracted contours
            "show_percept": True,  # shows percept
        }

        # specify classes to detect
        class_person = 0
        self.classes_detect = [class_person]

        print(
            "OE-Init: Defining paths, creating dirs, setting default device and datatype"
        )

        # path: Path infos for input and output images
        # ========================================================
        self.path = {"output": "test/output/level1/", "input": "test/images_test/"}
        # Make output directories, if necessary: the place were we dump the new images to...
        # os.makedirs(self.path["output"], mode=0o777, exist_ok=True)

        # Check if GPU is available and use it, if possible
        # =================================================
        self.default_dtype = torch.float32
        torch.set_default_dtype(self.default_dtype)
        if self.control["force_torch_use_cpu"]:
            torch_device: str = "cpu"
        else:
            torch_device: str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {torch_device} as TORCH device...")
        self.torch_device = torch_device

        print("OE-Init: Compute display scaling factors and padding RGB values")

        # global scaling factors for all pixel-related length scales
        self.display_size_max_x_PIX: float = (
            self.display["size_max_x_DVA"] * self.display["PIX_per_DVA"]
        )
        self.display_size_max_y_PIX: float = (
            self.display["size_max_y_DVA"] * self.display["PIX_per_DVA"]
        )

        # determine padding fill value
        tmp = tv.transforms.Grayscale(num_output_channels=1)
        tmp_value = torch.full((3, 1, 1), 0)
        self.padding_fill: int = int(tmp(tmp_value).squeeze())

        print(f"OE-Init: Opening camera source or video file '{source}'")

        # open source
        self.cap = WebCam(source)
        if not self.cap.open_cam():
            raise OSError(f"Opening source {source} failed!")

        # get the video frame size, frame count and frame rate
        frame_width = self.cap.cap_frame_width
        frame_height = self.cap.cap_frame_height
        fps = self.cap.cap_fps
        print(
            f"OE-Init: Processing frames of {frame_width} x {frame_height} @ {fps} fps."
        )

        # open output file if we want to save frames
        # if output_file != None:
        #     out = cv2.VideoWriter(
        #         output_file,
        #         cv2.VideoWriter_fourcc(*"MJPG"),
        #         fps,
        #         (out_x, out_y),
        #     )
        #     if out == None:
        #         raise OSError(f"Can not open file {output_file} for writing!")

        # get an instance of the Yolo segmentation network
        print("OE-Init: initialize YOLO")
        self.yolo = Yolo5Segmentation(torch_device=self.torch_device)

        self.send_dictionaries = False

        self.apply_parameter_changes()

        return

    def apply_parameter_changes(self):

        # GET NEW PARAMETERS
        print("OE-AppParChg: Computing sizes from new parameters")

        ### BLOCK: dictionary ----------------
        # set patch size for both dictionaries, make sure it is odd number
        dictionary_size_PIX: int = (
            1
            + (int(self.dictionary["size_DVA"] * self.display["PIX_per_DVA"]) // 2) * 2
        )

        ### BLOCK: gabor ---------------------
        # convert contour-related parameters to pixel units
        self.sigma_kernel_PIX: float = (
            self.gabor["sigma_kernel_DVA"] * self.display["PIX_per_DVA"]
        )
        self.lambda_kernel_PIX: float = (
            self.gabor["lambda_kernel_DVA"] * self.display["PIX_per_DVA"]
        )

        ### BLOCK: gabor & dictionary ------------------
        # Padding
        # -------
        self.padding_PIX: int = int(
            max(3.0 * self.sigma_kernel_PIX, 1.1 * dictionary_size_PIX)
        )

        # define target video/representation width/height
        multiple_of = 4
        out_x = self.display_size_max_x_PIX + 2 * self.padding_PIX
        out_y = self.display_size_max_y_PIX + 2 * self.padding_PIX
        out_x += (multiple_of - (out_x % multiple_of)) % multiple_of
        out_y += (multiple_of - (out_y % multiple_of)) % multiple_of
        self.out_x = int(out_x)
        self.out_y = int(out_y)

        # generate dictionaries
        # ---------------------
        ### BLOCK: dictionary --------------------------
        print("OE-AppParChg: Generating dictionaries...")
        patch_generator = PatchGenerator(torch_device=self.torch_device)
        self.phosphene = patch_generator.alphabet_phosphene(
            patch_size=dictionary_size_PIX,
            sigma_width=self.dictionary["phosphene"]["sigma_width"]
            * dictionary_size_PIX,
        )
        ### BLOCK: dictionary & gabor --------------------------
        self.clocks_filter, self.clocks, segments = patch_generator.alphabet_clocks(
            patch_size=dictionary_size_PIX,
            n_dir=self.dictionary["clocks"]["n_dir"],
            n_filter=self.gabor["n_orientations"],
            segment_width=self.dictionary["clocks"]["pointer_width"]
            * dictionary_size_PIX,
            segment_length=self.dictionary["clocks"]["pointer_length"]
            * dictionary_size_PIX,
        )

        self.send_dictionaries = True

        return

    # classes_detect, out_x, out_y
    def update(self, data_in):

        # handle parameter change

        if data_in:

            print("Incoming -----------> ", data_in)

            self.number_of_patches = data_in["number_of_patches"]

            self.classes_detect = data_in["value"]

            self.gabor["sigma_kernel_DVA"] = data_in["sigma_kernel_DVA"]
            self.gabor["lambda_kernel_DVA"] = data_in["sigma_kernel_DVA"] * 2
            self.gabor["n_orientations"] = data_in["n_orientations"]

            self.dictionary["size_DVA"] = data_in["size_DVA"]
            self.dictionary["phosphene"]["sigma_width"] = data_in["sigma_width"]
            self.dictionary["clocks"]["n_dir"] = data_in["n_dir"]
            self.dictionary["clocks"]["n_open"] = data_in["n_dir"] // 2
            self.dictionary["clocks"]["pointer_width"] = data_in["pointer_width"]
            self.dictionary["clocks"]["pointer_length"] = data_in["pointer_length"]

            self.encoding["use_exp_deadzone"] = data_in["use_exp_deadzone"]
            self.encoding["size_exp_deadzone_DVA"] = data_in["size_exp_deadzone_DVA"]
            self.encoding["use_cutout_deadzone"] = data_in["use_cutout_deadzone"]
            self.encoding["size_cutout_deadzone_DVA"] = data_in[
                "size_cutout_deadzone_DVA"
            ]

            self.control["show_capture"] = data_in["enable_cam"]
            self.control["show_object"] = data_in["enable_yolo"]
            self.control["show_contours"] = data_in["enable_contour"]
            # TODO Fenster zumachen
            self.apply_parameter_changes()

        # some constants for addressing specific components of output arrays
        image_id_CONST: int = 0
        overlap_index_CONST: int = 1

        # format: color_RGB, height, width <class 'torch.tensor'> float, range=0,1
        print("OE-ProcessFrame: capturing frame")
        frame = self.cap.get_frame()
        if frame == None:
            raise OSError(f"Can not capture frame {i_frame}")
        if self.verbose:
            if self.control["show_capture"]:
                show_torch_frame(frame, title="Captured", target=self.verbose)
            else:
                try:
                    cv2.destroyWindow("Captured")
                except:
                    pass

        # perform segmentation

        frame = frame.to(device=self.torch_device)
        print("OE-ProcessFrame: frame segmentation by YOLO")
        frame_segmented = self.yolo(frame.unsqueeze(0), classes=self.classes_detect)

        # This extracts the frame in x to convert the mask in a video format
        if self.yolo.found_class_id != None:

            n_found = len(self.yolo.found_class_id)
            print(
                f"OE-ProcessFrame: {n_found} occurrences of desired object found in frame!"
            )

            mask = frame_segmented[0]

            # is there something in the mask?
            if not mask.sum() == 0:

                # yes, cut only the part of the frame that has our object of interest
                frame_masked = mask * frame

                x_height = mask.sum(axis=-2)
                x_indices = torch.where(x_height > 0)
                x_max = x_indices[0].max() + 1
                x_min = x_indices[0].min()

                y_height = mask.sum(axis=-1)
                y_indices = torch.where(y_height > 0)
                y_max = y_indices[0].max() + 1
                y_min = y_indices[0].min()

                frame_cut = frame_masked[:, y_min:y_max, x_min:x_max]
            else:
                print(f"OE-ProcessFrame: Mask contains all zeros in current frame!")
                frame_cut = None
        else:
            print(f"OE-ProcessFrame: No objects found in current frame!")
            frame_cut = None

        if frame_cut == None:
            # out_torch = torch.zeros([self.out_y, self.out_x])
            position_selection = torch.zeros((1, 0, 3))
            contour_shape = [1, self.gabor["n_orientations"], 1, 1]
        else:
            if self.verbose:
                if self.control["show_object"]:
                    show_torch_frame(
                        frame_cut, title="Selected Object", target=self.verbose
                    )
                else:
                    try:
                        cv2.destroyWindow("Selected Object")
                    except:
                        pass

            # UDO: from here on, we proceed as before, just handing
            # UDO: over the frame_cut --> image
            image = frame_cut

            # Determine target size of image
            # image: [RGB, Height, Width], dtype= tensor.torch.uint8
            print("OE-ProcessFrame: Computing downsampling factor image -> display")
            f_x: float = self.display_size_max_x_PIX / image.shape[-1]
            f_y: float = self.display_size_max_y_PIX / image.shape[-2]
            f_xy_min: float = min(f_x, f_y)
            downsampling_x: int = int(f_xy_min * image.shape[-1])
            downsampling_y: int = int(f_xy_min * image.shape[-2])

            # CURRENTLY we do not crop in the end...
            # Image size for removing the fft crop later
            # center_crop_x: int = downsampling_x
            # center_crop_y: int = downsampling_y

            # define contour extraction processing chain
            # ------------------------------------------
            print("OE-ProcessFrame: Extracting contours")
            train_processing_chain = tv.transforms.Compose(
                transforms=[
                    tv.transforms.Grayscale(num_output_channels=1),  # RGB to grayscale
                    tv.transforms.Resize(
                        size=(downsampling_y, downsampling_x)
                    ),  # downsampling
                    tv.transforms.Pad(  # extra white padding around the picture
                        padding=(self.padding_PIX, self.padding_PIX),
                        fill=self.padding_fill,
                    ),
                    ContourExtract(  # contour extraction
                        n_orientations=self.gabor["n_orientations"],
                        sigma_kernel=self.sigma_kernel_PIX,
                        lambda_kernel=self.lambda_kernel_PIX,
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

            if self.verbose:
                if self.control["show_contours"]:
                    show_torch_frame(
                        contour_collapse,
                        title="Contours Extracted",
                        cmap="gray",
                        target=self.verbose,
                    )
                else:
                    try:
                        cv2.destroyWindow("Contours Extracted")
                    except:
                        pass

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
            print("OE-ProcessFrame: Performing sparsification")
            sparsifier = Sparsifier(
                dictionary_filter=self.clocks_filter,
                dictionary=self.clocks,
                dictionary_prior=dictionary_prior,
                number_of_patches=self.encoding["n_patches_compute"],
                size_exp_deadzone=self.encoding["size_exp_deadzone_DVA"]
                * self.display["PIX_per_DVA"],
                plot_use_map=False,  # self.control["plot_deadzone"],
                deadzone_exp=self.encoding["use_exp_deadzone"],
                deadzone_hard_cutout=self.encoding["use_cutout_deadzone"],
                deadzone_hard_cutout_size=self.encoding["size_cutout_deadzone_DVA"]
                * self.display["PIX_per_DVA"],
                padding_deadzone_size_x=self.padding_PIX,
                padding_deadzone_size_y=self.padding_PIX,
                torch_device=self.torch_device,
            )
            sparsifier(contour)
            assert sparsifier.position_found is not None

            # extract and normalize the overlap found
            overlap_found = sparsifier.overlap_found[
                image_id_CONST, :, overlap_index_CONST
            ]
            overlap_found = overlap_found / overlap_found.max()

            # get overlap above certain threshold, extract corresponding elements
            overlap_idcs_valid = torch.where(
                overlap_found >= self.encoding["overlap_threshold"]
            )[0]
            position_selection = sparsifier.position_found[
                image_id_CONST : image_id_CONST + 1, overlap_idcs_valid, :
            ]
            n_elements = len(overlap_idcs_valid)
            print(f"OE-ProcessFrame: {n_elements} elements positioned!")

            contour_shape = contour.shape

        n_cut = min(position_selection.shape[-2], self.number_of_patches)

        data_out = {
            "position_found": position_selection[:, :n_cut, :],
            "canvas_size": contour_shape,
        }
        if self.send_dictionaries:
            data_out["features"] = self.clocks
            data_out["phosphene"] = self.phosphene
            self.send_dictionaries = False

        return data_out

    def __del__(self):

        print("OE-Delete: exiting gracefully!")
        self.cap.close_cam()
        try:
            cv2.destroyAllWindows()
        except:
            pass


# TODO no output file
# TODO detect end of file if input is video file

if __name__ == "__main__":

    verbose = "cv2"
    source = 0  # "GoProWireless"
    frame_count = 20
    i_frame = 0

    data_in = None

    oe = OnlineEncoding(source=source, verbose=verbose)

    # Loop over the frames
    while i_frame < frame_count:

        i_frame += 1

        if i_frame == (frame_count // 3):
            oe.dictionary["size_DVA"] = 0.5
            oe.apply_parameter_changes()

        if i_frame == (frame_count * 2 // 3):
            oe.dictionary["size_DVA"] = 2.0
            oe.apply_parameter_changes()

        data_out = oe.update(data_in)
        position_selection = data_out["position_found"]
        contour_shape = data_out["canvas_size"]

        # SENDE/EMPANGSLOGIK:
        #
        # <- PACKET empfangen
        # Parameteränderungen?
        #    in Instanz se übertragen
        #    "apply_parameter_changes" aufrufen
        #    folgende variablen in sendepacket:
        #       se.clocks, se.phosphene, se.out_x, se.out_y
        # "process_frame"
        # folgende variablen in sendepacket:
        #    position_selection, contour_shape
        # -> PACKET zurückgeben

        # build the full image!
        image_clocks = BuildImage(
            canvas_size=contour_shape,
            dictionary=oe.clocks,
            position_found=position_selection,
            default_dtype=oe.default_dtype,
            torch_device=oe.torch_device,
        )
        # image_phosphenes = BuildImage(
        #     canvas_size=contour.shape,
        #     dictionary=dictionary_phosphene,
        #     position_found=position_selection,
        #     default_dtype=default_dtype,
        #     torch_device=torch_device,
        # )

        # normalize to range [0...1]
        m = image_clocks[0].max()
        if m == 0:
            m = 1
        image_clocks_normalized = image_clocks[0] / m

        # embed into frame of desired output size
        out_torch = embed_image(
            image_clocks_normalized, out_height=oe.out_y, out_width=oe.out_x
        )

        # show, if desired
        if verbose:
            if oe.control["show_percept"]:
                show_torch_frame(
                    out_torch, title="Percept", cmap="gray", target=verbose
                )

        # if output_file != None:
        #     out_pixel = (
        #         (out_torch * torch.ones([3, 1, 1]) * 255)
        #         .type(dtype=torch.uint8)
        #         .movedim(0, -1)
        #         .numpy()
        #     )
        #     out.write(out_pixel)

    del oe

    # if output_file != None:
    #     out.release()

# %%
