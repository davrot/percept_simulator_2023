# Sparsifier.py
# ====================================
# matches dictionary patches to contour images
#
# Version V1.0, 07.03.2023:
#   slight parameter scaling changes to David's last code version...
#
# Version V1.1, 07.03.2023:
#   merged David's rebuild code (GUI capable)
#

import torch
import math

import matplotlib.pyplot as plt

import time


class Sparsifier(torch.nn.Module):

    dictionary_filter_fft: torch.Tensor | None = None
    dictionary_filter: torch.Tensor
    dictionary: torch.Tensor

    parameter_ready: bool
    dictionary_ready: bool

    contour_convolved_sum: torch.Tensor | None = None
    use_map: torch.Tensor | None = None
    position_found: torch.Tensor | None = None

    size_exp_deadzone: float

    number_of_patches: int
    padding_deadzone_size_x: int
    padding_deadzone_size_y: int

    plot_use_map: bool

    deadzone_exp: bool
    deadzone_hard_cutout: int  # 0 = not, 1 = round, 2 = box
    deadzone_hard_cutout_size: float

    dictionary_prior: torch.Tensor | None

    pi: torch.Tensor
    torch_device: torch.device
    default_dtype = torch.float32

    def __init__(
        self,
        dictionary_filter: torch.Tensor,
        dictionary: torch.Tensor,
        dictionary_prior: torch.Tensor | None = None,
        number_of_patches: int = 10,  #
        size_exp_deadzone: float = 1.0,  #
        padding_deadzone_size_x: int = 0,  #
        padding_deadzone_size_y: int = 0,  #
        plot_use_map: bool = False,
        deadzone_exp: bool = True,  #
        deadzone_hard_cutout: int = 1,  # 0 = not, 1 = round
        deadzone_hard_cutout_size: float = 1.0,  #
        torch_device: str = "cpu",
    ):
        super().__init__()

        self.dictionary_ready = False
        self.parameter_ready = False

        self.torch_device = torch.device(torch_device)

        self.pi = torch.tensor(
            math.pi,
            device=self.torch_device,
            dtype=self.default_dtype,
        )

        self.plot_use_map = plot_use_map

        self.update_parameter(
            number_of_patches,
            size_exp_deadzone,
            padding_deadzone_size_x,
            padding_deadzone_size_y,
            deadzone_exp,
            deadzone_hard_cutout,
            deadzone_hard_cutout_size,
        )

        self.update_dictionary(dictionary_filter, dictionary, dictionary_prior)

    def update_parameter(
        self,
        number_of_patches: int,
        size_exp_deadzone: float,
        padding_deadzone_size_x: int,
        padding_deadzone_size_y: int,
        deadzone_exp: bool,
        deadzone_hard_cutout: int,
        deadzone_hard_cutout_size: float,
    ) -> None:

        self.parameter_ready = False

        assert size_exp_deadzone > 0.0
        assert number_of_patches > 0
        assert padding_deadzone_size_x >= 0
        assert padding_deadzone_size_y >= 0

        self.number_of_patches = number_of_patches
        self.size_exp_deadzone = size_exp_deadzone
        self.padding_deadzone_size_x = padding_deadzone_size_x
        self.padding_deadzone_size_y = padding_deadzone_size_y

        self.deadzone_exp = deadzone_exp
        self.deadzone_hard_cutout = deadzone_hard_cutout
        self.deadzone_hard_cutout_size = deadzone_hard_cutout_size

        self.parameter_ready = True

    def update_dictionary(
        self,
        dictionary_filter: torch.Tensor,
        dictionary: torch.Tensor,
        dictionary_prior: torch.Tensor | None = None,
    ) -> None:

        self.dictionary_ready = False

        assert dictionary_filter.ndim == 4
        assert dictionary.ndim == 4

        # Odd number of pixels. Please!
        assert (dictionary_filter.shape[-2] % 2) == 1
        assert (dictionary_filter.shape[-1] % 2) == 1

        self.dictionary_filter = dictionary_filter.detach().clone()
        self.dictionary = dictionary

        if dictionary_prior is not None:
            assert dictionary_prior.ndim == 1
            assert dictionary_prior.shape[0] == dictionary_filter.shape[0]

        self.dictionary_prior = dictionary_prior

        self.dictionary_filter_fft = None

        self.dictionary_ready = True

    def fourier_convolution(self, contour: torch.Tensor):
        # Pattern, X, Y
        assert contour.dim() == 4
        assert self.dictionary_filter is not None
        assert contour.shape[-2] >= self.dictionary_filter.shape[-2]
        assert contour.shape[-1] >= self.dictionary_filter.shape[-1]

        t0 = time.time()

        contour_fft = torch.fft.rfft2(contour, dim=(-2, -1))

        t1 = time.time()

        if (
            (self.dictionary_filter_fft is None)
            or (self.dictionary_filter_fft.dim() != 4)
            or (self.dictionary_filter_fft.shape[-2] != contour.shape[-2])
            or (self.dictionary_filter_fft.shape[-1] != contour.shape[-1])
        ):
            dictionary_padded: torch.Tensor = torch.zeros(
                (
                    self.dictionary_filter.shape[0],
                    self.dictionary_filter.shape[1],
                    contour.shape[-2],
                    contour.shape[-1],
                ),
                device=self.torch_device,
                dtype=self.default_dtype,
            )
            dictionary_padded[
                :,
                :,
                : self.dictionary_filter.shape[-2],
                : self.dictionary_filter.shape[-1],
            ] = self.dictionary_filter.flip(dims=[-2, -1])

            dictionary_padded = dictionary_padded.roll(
                (
                    -(self.dictionary_filter.shape[-2] // 2),
                    -(self.dictionary_filter.shape[-1] // 2),
                ),
                (-2, -1),
            )
            self.dictionary_filter_fft = torch.fft.rfft2(
                dictionary_padded, dim=(-2, -1)
            )

        t2 = time.time()

        assert self.dictionary_filter_fft is not None

        # dimension order for multiplication: [pat, feat, ori, x, y]
        self.contour_convolved_sum = torch.fft.irfft2(
            contour_fft.unsqueeze(1) * self.dictionary_filter_fft.unsqueeze(0),
            dim=(-2, -1),
        ).sum(dim=2)
        # --> [pat, feat, x, y]
        t3 = time.time()

        self.use_map: torch.Tensor = torch.ones(
            (contour.shape[0], 1, contour.shape[-2], contour.shape[-1]),
            device=self.torch_device,
            dtype=self.default_dtype,
        )

        if self.padding_deadzone_size_x > 0:
            self.use_map[:, 0, : self.padding_deadzone_size_x, :] = 0.0
            self.use_map[:, 0, -self.padding_deadzone_size_x :, :] = 0.0

        if self.padding_deadzone_size_y > 0:
            self.use_map[:, 0, :, : self.padding_deadzone_size_y] = 0.0
            self.use_map[:, 0, :, -self.padding_deadzone_size_y :] = 0.0

        t4 = time.time()
        print(
            "Sparsifier-convol {:.3f}s: fft-img-{:.3f}s, fft-fil-{:.3f}s, convol-{:.3f}s, pad-{:.3f}s".format(
                t4 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3
            )
        )

        return

    def find_next_element(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.use_map is not None
        assert self.contour_convolved_sum is not None
        assert self.dictionary_filter is not None

        # store feature index, x pos and y pos (3 entries)
        position_found = torch.zeros(
            (self.contour_convolved_sum.shape[0], 3),
            dtype=torch.int64,
            device=self.torch_device,
        )

        overlap_found = torch.zeros(
            (self.contour_convolved_sum.shape[0], 2),
            device=self.torch_device,
            dtype=self.default_dtype,
        )

        for pattern_id in range(0, self.contour_convolved_sum.shape[0]):

            t0 = time.time()
            # search_tensor: torch.Tensor = (
            #    self.contour_convolved[pattern_id] * self.use_map[pattern_id]
            # ).sum(dim=1)
            search_tensor: torch.Tensor = (
                self.contour_convolved_sum[pattern_id] * self.use_map[pattern_id]
            )

            t1 = time.time()

            if self.dictionary_prior is not None:
                search_tensor *= self.dictionary_prior.unsqueeze(-1).unsqueeze(-1)

            t2 = time.time()

            temp, max_0 = search_tensor.max(dim=0)
            temp, max_1 = temp.max(dim=0)
            temp_overlap, max_2 = temp.max(dim=0)

            position_base_3: int = int(max_2)
            position_base_2: int = int(max_1[position_base_3])
            position_base_1: int = int(max_0[position_base_2, position_base_3])
            position_base_0: int = int(pattern_id)

            position_found[position_base_0, 0] = position_base_1
            position_found[position_base_0, 1] = position_base_2
            position_found[position_base_0, 2] = position_base_3

            overlap_found[pattern_id, 0] = temp_overlap
            overlap_found[pattern_id, 1] = self.contour_convolved_sum[
                position_base_0, position_base_1, position_base_2, position_base_3
            ]

            t3 = time.time()

            x_max: int = int(self.contour_convolved_sum.shape[-2])
            y_max: int = int(self.contour_convolved_sum.shape[-1])

            # Center arround the max position
            x_0 = int(-position_base_2)
            x_1 = int(x_max - position_base_2)
            y_0 = int(-position_base_3)
            y_1 = int(y_max - position_base_3)

            temp_grid_x: torch.Tensor = torch.arange(
                start=x_0,
                end=x_1,
                device=self.torch_device,
                dtype=self.default_dtype,
            )

            temp_grid_y: torch.Tensor = torch.arange(
                start=y_0,
                end=y_1,
                device=self.torch_device,
                dtype=self.default_dtype,
            )

            x, y = torch.meshgrid(temp_grid_x, temp_grid_y, indexing="ij")

            # discourage the neigbourhood around for the future
            if self.deadzone_exp is True:
                self.temp_map: torch.Tensor = 1.0 - torch.exp(
                    -(x**2 + y**2) / (2 * self.size_exp_deadzone**2)
                )
            else:
                self.temp_map = torch.ones(
                    (x.shape[0], x.shape[1]),
                    device=self.torch_device,
                    dtype=self.default_dtype,
                )

            assert self.deadzone_hard_cutout >= 0
            assert self.deadzone_hard_cutout <= 1
            assert self.deadzone_hard_cutout_size >= 0

            if self.deadzone_hard_cutout == 1:
                temp = x**2 + y**2
                self.temp_map *= torch.where(
                    temp <= self.deadzone_hard_cutout_size**2,
                    0.0,
                    1.0,
                )

            self.use_map[position_base_0, 0, :, :] *= self.temp_map

            t4 = time.time()

            # Only for keeping it in the float32 range:
            self.use_map[position_base_0, 0, :, :] /= self.use_map[
                position_base_0, 0, :, :
            ].max()

            t5 = time.time()

            # print(
            #     "{}, {}, {}, {}, {}".format(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)
            # )

        return (
            position_found,
            overlap_found,
            torch.tensor((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)),
        )

    def forward(self, input: torch.Tensor) -> None:
        assert self.number_of_patches > 0

        assert self.dictionary_ready is True
        assert self.parameter_ready is True

        self.position_found = torch.zeros(
            (input.shape[0], self.number_of_patches, 3),
            dtype=torch.int64,
            device=self.torch_device,
        )
        self.overlap_found = torch.zeros(
            (input.shape[0], self.number_of_patches, 2),
            device=self.torch_device,
            dtype=self.default_dtype,
        )

        # Folding the images and the dictionary
        self.fourier_convolution(input)

        t0 = time.time()
        dt = torch.tensor([0, 0, 0, 0, 0])
        for patch_id in range(0, self.number_of_patches):
            (
                self.position_found[:, patch_id, :],
                self.overlap_found[:, patch_id, :],
                dt_tmp,
            ) = self.find_next_element()

            dt = dt + dt_tmp

            if self.plot_use_map is True:

                assert self.position_found.shape[0] == 1
                assert self.use_map is not None

                print("Position Saliency:")
                print(self.overlap_found[0, :, 0])
                print("Overlap with Contour Image:")
                print(self.overlap_found[0, :, 1])
                plt.subplot(1, 2, 1)
                plt.imshow(self.use_map[0, 0].cpu(), cmap="gray")
                plt.title(f"patch-number: {patch_id}")
                plt.axis("off")
                plt.colorbar(shrink=0.5)
                plt.subplot(1, 2, 2)
                plt.imshow(
                    (self.use_map[0, 0] * input[0].sum(dim=0)).cpu(), cmap="gray"
                )
                plt.title(f"patch-number: {patch_id}")
                plt.axis("off")
                plt.colorbar(shrink=0.5)
                plt.show()

        # self.overlap_found /= self.overlap_found.max(dim=1, keepdim=True)[0]
        t1 = time.time()
        print(
            "Sparsifier-forward {:.3f}s: usemap-{:.3f}s, prior-{:.3f}s, findmax-{:.3f}s, notch-{:.3f}s, norm-{:.3f}s: (sum-{:.3f})".format(
                t1 - t0, dt[0], dt[1], dt[2], dt[3], dt[4], dt.sum()
            )
        )
