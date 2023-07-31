# ContourExtract.py
# ====================================
# extracts contours from gray-level images
#
# Version V1.0, pre-07.03.2023:
#   no actual changes, is David's last code version...
#
# Version V1.1, 07.03.2023:
#   merged David's rebuild code (GUI capable)
#

import torch
import time
import math


class ContourExtract(torch.nn.Module):
    image_width: int = -1
    image_height: int = -1
    output_threshold: torch.Tensor = torch.tensor(-1)

    n_orientations: int
    sigma_kernel: float = 0
    lambda_kernel: float = 0
    gamma_aspect_ratio: float = 0

    kernel_axis_x: torch.Tensor | None = None
    kernel_axis_y: torch.Tensor | None = None
    target_orientations: torch.Tensor
    weight_vector: torch.Tensor

    image_scale: float | None

    psi_phase_offset_cos: torch.Tensor
    psi_phase_offset_sin: torch.Tensor

    fft_gabor_cos_bank: torch.Tensor | None = None
    fft_gabor_sin_bank: torch.Tensor | None = None

    pi: torch.Tensor
    torch_device: torch.device
    default_dtype = torch.float32

    rebuild_kernels: bool

    def __init__(
        self,
        n_orientations: int,
        sigma_kernel: float,
        lambda_kernel: float,
        gamma_aspect_ratio: float = 1.0,
        image_scale: float | None = 255.0,
        torch_device: str = "cpu",
    ):
        super().__init__()
        self.torch_device = torch.device(torch_device)

        self.pi = torch.tensor(
            math.pi,
            device=self.torch_device,
            dtype=self.default_dtype,
        )

        self.psi_phase_offset_cos = torch.tensor(
            0.0,
            device=self.torch_device,
            dtype=self.default_dtype,
        )
        self.psi_phase_offset_sin = -self.pi / 2
        self.gamma_aspect_ratio = gamma_aspect_ratio
        self.image_scale = image_scale

        self.update_settings(
            n_orientations=n_orientations,
            sigma_kernel=sigma_kernel,
            lambda_kernel=lambda_kernel,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        assert input.ndim == 4, "input must have 4 dims!"

        # We can only handle one color channel
        assert input.shape[1] == 1, "input.shape[1] must be 1!"

        input = input.type(dtype=self.default_dtype).to(device=self.torch_device)
        if self.image_scale is not None:
            # scale grey level [0, 255] to range [0.0, 1.0]
            input /= self.image_scale

        # Do we have valid kernels?
        if input.shape[-2] != self.image_width:
            self.image_width = input.shape[-2]
            self.rebuild_kernels = True

        if input.shape[-1] != self.image_height:
            self.image_height = input.shape[-1]
            self.rebuild_kernels = True

        assert self.image_width > 0
        assert self.image_height > 0

        t0 = time.time()

        # We need to rebuild the kernels
        if self.rebuild_kernels is True:

            self.kernel_axis_x = self.create_kernel_axis(self.image_width)
            self.kernel_axis_y = self.create_kernel_axis(self.image_height)

            assert self.kernel_axis_x is not None
            assert self.kernel_axis_y is not None

            gabor_cos_bank, gabor_sin_bank = self.create_gabor_filter_bank()

            assert gabor_cos_bank is not None
            assert gabor_sin_bank is not None

            self.fft_gabor_cos_bank = torch.fft.rfft2(
                gabor_cos_bank, s=None, dim=(-2, -1), norm=None
            ).unsqueeze(0)

            self.fft_gabor_sin_bank = torch.fft.rfft2(
                gabor_sin_bank, s=None, dim=(-2, -1), norm=None
            ).unsqueeze(0)

            # compute threshold for ignoring non-zero outputs arising due
            # to numerical imprecision (fft, kernel definition)
            assert self.weight_vector is not None
            norm_input = torch.full(
                (1, 1, self.image_width, self.image_height),
                1.0,
                device=self.torch_device,
                dtype=self.default_dtype,
            )
            norm_fft_input = torch.fft.rfft2(
                norm_input, s=None, dim=(-2, -1), norm=None
            )

            norm_output_sin = torch.fft.irfft2(
                norm_fft_input * self.fft_gabor_sin_bank,
                s=None,
                dim=(-2, -1),
                norm=None,
            )
            norm_output_cos = torch.fft.irfft2(
                norm_fft_input * self.fft_gabor_cos_bank,
                s=None,
                dim=(-2, -1),
                norm=None,
            )
            norm_value_matrix = torch.sqrt(norm_output_sin**2 + norm_output_cos**2)

            # norm_output = torch.abs(
            #     (self.weight_vector * norm_value_matrix).sum(dim=-1)
            # ).type(dtype=torch.float32)

            self.output_threshold = norm_value_matrix.max()

            self.rebuild_kernels = False

        assert self.fft_gabor_cos_bank is not None
        assert self.fft_gabor_sin_bank is not None

        t1 = time.time()

        fft_input = torch.fft.rfft2(input, s=None, dim=(-2, -1), norm=None)

        output_sin = torch.fft.irfft2(
            fft_input * self.fft_gabor_sin_bank, s=None, dim=(-2, -1), norm=None
        )

        output_cos = torch.fft.irfft2(
            fft_input * self.fft_gabor_cos_bank, s=None, dim=(-2, -1), norm=None
        )

        t2 = time.time()

        output = torch.sqrt(output_sin**2 + output_cos**2)

        t3 = time.time()

        print(
            "ContourExtract {:.3f}s: prep-{:.3f}s, fft-{:.3f}s, out-{:.3f}s".format(
                t3 - t0, t1 - t0, t2 - t1, t3 - t2
            )
        )

        return output

    def create_collapse(self, input: torch.Tensor) -> torch.Tensor:

        assert self.weight_vector is not None

        output = torch.abs((self.weight_vector * input).sum(dim=1)).type(
            dtype=self.default_dtype
        )

        return output

    def create_kernel_axis(self, axis_size: int) -> torch.Tensor:

        lower_bound_axis: int = -int(math.floor(axis_size / 2))
        upper_bound_axis: int = int(math.ceil(axis_size / 2))

        kernel_axis = torch.arange(
            lower_bound_axis,
            upper_bound_axis,
            device=self.torch_device,
            dtype=self.default_dtype,
        )
        kernel_axis = torch.roll(kernel_axis, int(math.ceil(axis_size / 2)))

        return kernel_axis

    def create_gabor_filter_bank(self) -> tuple[torch.Tensor, torch.Tensor]:

        assert self.kernel_axis_x is not None
        assert self.kernel_axis_y is not None
        assert self.target_orientations is not None

        orientation_matrix = (
            self.target_orientations.unsqueeze(-1).unsqueeze(-1).detach().clone()
        )
        x_kernel_matrix = self.kernel_axis_x.unsqueeze(0).unsqueeze(-1).detach().clone()
        y_kernel_matrix = self.kernel_axis_y.unsqueeze(0).unsqueeze(0).detach().clone()

        r2 = x_kernel_matrix**2 + self.gamma_aspect_ratio * y_kernel_matrix**2

        kr = x_kernel_matrix * torch.cos(
            orientation_matrix
        ) + y_kernel_matrix * torch.sin(orientation_matrix)

        c0 = torch.exp(-2 * (self.pi * self.sigma_kernel / self.lambda_kernel) ** 2)

        gauss: torch.Tensor = torch.exp(-r2 / 2 / self.sigma_kernel**2)

        gabor_cos_bank: torch.Tensor = gauss * (
            torch.cos(2 * self.pi * kr / self.lambda_kernel + self.psi_phase_offset_cos)
            - c0 * torch.cos(self.psi_phase_offset_cos)
        )

        gabor_sin_bank: torch.Tensor = gauss * (
            torch.cos(2 * self.pi * kr / self.lambda_kernel + self.psi_phase_offset_sin)
            - c0 * torch.cos(self.psi_phase_offset_sin)
        )

        return gabor_cos_bank, gabor_sin_bank

    def update_settings(
        self,
        n_orientations: int,
        sigma_kernel: float,
        lambda_kernel: float,
    ):

        self.rebuild_kernels = True

        self.n_orientations = n_orientations
        self.sigma_kernel = sigma_kernel
        self.lambda_kernel = lambda_kernel

        # generate orientation axis and axis for complex summation
        self.target_orientations: torch.Tensor = (
            torch.arange(
                start=0,
                end=int(self.n_orientations),
                device=self.torch_device,
                dtype=self.default_dtype,
            )
            * torch.tensor(
                math.pi,
                device=self.torch_device,
                dtype=self.default_dtype,
            )
            / self.n_orientations
        )

        self.weight_vector: torch.Tensor = (
            torch.exp(
                2.0
                * torch.complex(torch.tensor(0.0), torch.tensor(1.0))
                * self.target_orientations
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(0)
        )
