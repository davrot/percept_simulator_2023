# %%
# PatchGenerator.py
# ====================================
# generates dictionaries (currently: phosphenes or clocks)
#
# Version V1.0, pre-07.03.2023:
#   no actual changes, is David's last code version...
#
# Version V1.1, 07.03.2023:
#   merged David's rebuild code (GUI capable)
#   (there was not really anything to merge :-))
#

import torch
import math


class PatchGenerator:

    pi: torch.Tensor
    torch_device: torch.device
    default_dtype = torch.float32

    def __init__(self, torch_device: str = "cpu"):
        self.torch_device = torch.device(torch_device)

        self.pi = torch.tensor(
            math.pi,
            device=self.torch_device,
            dtype=self.default_dtype,
        )

    def alphabet_phosphene(
        self,
        sigma_width: float = 2.5,
        patch_size: int = 41,
    ) -> torch.Tensor:

        n: int = int(patch_size // 2)
        temp_grid: torch.Tensor = torch.arange(
            start=-n,
            end=n + 1,
            device=self.torch_device,
            dtype=self.default_dtype,
        )

        x, y = torch.meshgrid(temp_grid, temp_grid, indexing="ij")

        phosphene: torch.Tensor = torch.exp(-(x**2 + y**2) / (2 * sigma_width**2))
        phosphene /= phosphene.sum()

        return phosphene.unsqueeze(0).unsqueeze(0)

    def alphabet_clocks(
        self,
        n_dir: int = 8,
        n_open: int = 4,
        n_filter: int = 4,
        patch_size: int = 41,
        segment_width: float = 2.5,
        segment_length: float = 15.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # for n_dir directions, there are n_open_max opening angles possible...
        assert n_dir % 2 == 0, "n_dir must be multiple of 2"
        n_open_max: int = n_dir // 2

        # ...but this number can be reduced by integer multiples:
        assert (
            n_open_max % n_open == 0
        ), "n_open_max = n_dir // 2 must be multiple of n_open"
        mul_open: int = n_open_max // n_open

        # filter planes must be multiple of number of orientations implied by n_dir
        assert n_filter % n_open_max == 0, "n_filter must be multiple of (n_dir // 2)"
        mul_filter: int = n_filter // n_open_max
        # compute single segments
        segments: torch.Tensor = torch.zeros(
            (n_dir, patch_size, patch_size),
            device=self.torch_device,
            dtype=self.default_dtype,
        )
        dirs: torch.Tensor = (
            2
            * self.pi
            * torch.arange(
                start=0, end=n_dir, device=self.torch_device, dtype=self.default_dtype
            )
            / n_dir
        )

        for i_dir in range(n_dir):
            segments[i_dir] = self.draw_segment(
                patch_size=patch_size,
                phi=float(dirs[i_dir]),
                segment_length=segment_length,
                segment_width=segment_width,
            )

        # compute patches from segments
        clocks = torch.zeros(
            (n_open, n_dir, patch_size, patch_size),
            device=self.torch_device,
            dtype=self.default_dtype,
        )
        clocks_filter = torch.zeros(
            (n_open, n_dir, n_filter, patch_size, patch_size),
            device=self.torch_device,
            dtype=self.default_dtype,
        )

        for i_dir in range(n_dir):
            for i_open in range(n_open):

                seg1 = segments[i_dir]
                seg2 = segments[(i_dir + (i_open + 1) * mul_open) % n_dir]
                clocks[i_open, i_dir] = torch.where(seg1 > seg2, seg1, seg2)

                i_filter_seg1 = (i_dir * mul_filter) % n_filter
                i_filter_seg2 = (
                    (i_dir + (i_open + 1) * mul_open) * mul_filter
                ) % n_filter

                if i_filter_seg1 == i_filter_seg2:
                    clock_merged = torch.where(seg1 > seg2, seg1, seg2)
                    clocks_filter[i_open, i_dir, i_filter_seg1] = clock_merged
                else:
                    clocks_filter[i_open, i_dir, i_filter_seg1] = seg1
                    clocks_filter[i_open, i_dir, i_filter_seg2] = seg2

        clocks_filter = clocks_filter.reshape(
            (n_open * n_dir, n_filter, patch_size, patch_size)
        )
        clocks_filter = clocks_filter / clocks_filter.sum(
            axis=(-3, -2, -1), keepdims=True
        )
        clocks = clocks.reshape((n_open * n_dir, 1, patch_size, patch_size))
        clocks = clocks / clocks.sum(axis=(-2, -1), keepdims=True)

        return clocks_filter, clocks, segments

    def draw_segment(
        self,
        patch_size: float,
        phi: float,
        segment_length: float,
        segment_width: float,
    ) -> torch.Tensor:

        # extension of patch beyond origin
        n: int = int(patch_size // 2)

        # grid for the patch
        temp_grid: torch.Tensor = torch.arange(
            start=-n,
            end=n + 1,
            device=self.torch_device,
            dtype=self.default_dtype,
        )

        x, y = torch.meshgrid(temp_grid, temp_grid, indexing="ij")

        r: torch.Tensor = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=2)

        # target orientation of basis segment
        phi90: torch.Tensor = phi + self.pi / 2

        # vector pointing to the ending point of segment (direction)
        #
        # when result is displayed with plt.imshow(segment),
        # phi=0 points to the right, and increasing phi rotates
        # the segment counterclockwise
        #
        e: torch.Tensor = torch.tensor(
            [torch.cos(phi90), torch.sin(phi90)],
            device=self.torch_device,
            dtype=self.default_dtype,
        )

        # tangential vectors
        e_tang: torch.Tensor = e.flip(dims=[0]) * torch.tensor(
            [-1, 1], device=self.torch_device, dtype=self.default_dtype
        )

        # compute distances to segment: parallel/tangential
        d = torch.maximum(
            torch.zeros(
                (r.shape[0], r.shape[1]),
                device=self.torch_device,
                dtype=self.default_dtype,
            ),
            torch.abs(
                (r * e.unsqueeze(0).unsqueeze(0)).sum(dim=-1) - segment_length / 2
            )
            - segment_length / 2,
        )

        d_tang = (r * e_tang.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        # compute minimum distance to any of the two pointers
        dr = torch.sqrt(d**2 + d_tang**2)

        segment = torch.exp(-(dr**2) / 2 / segment_width**2)
        segment = segment / segment.sum()

        return segment


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pg = PatchGenerator()

    patch1 = pg.draw_segment(
        patch_size=81, phi=math.pi / 4, segment_length=30, segment_width=5
    )
    patch2 = pg.draw_segment(patch_size=81, phi=0, segment_length=30, segment_width=5)
    plt.imshow(torch.where(patch1 > patch2, patch1, patch2).cpu())

    phos = pg.alphabet_phosphene()
    plt.imshow(phos[0, 0].cpu())

    n_filter = 8
    n_dir = 8
    clocks_filter, clocks, segments = pg.alphabet_clocks(n_dir=n_dir, n_filter=n_filter)

    n_features = clocks_filter.shape[0]
    print(n_features, "clock features generated!")
    for i_feature in range(n_features):
        for i_filter in range(n_filter):
            plt.subplot(1, n_filter, i_filter + 1)
            plt.imshow(clocks_filter[i_feature, i_filter].cpu())
            plt.title("Feature #{}, Dir #{}".format(i_feature, i_filter))
        plt.show()


# %%
