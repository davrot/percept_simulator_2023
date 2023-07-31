import torch


def clip_coordinates(x_canvas: int, dx_canvas: int, dx_dict: int):

    x_canvas = int(x_canvas)
    dx_canvas = int(dx_canvas)
    dx_dict = int(dx_dict)
    dr_dict = int(dx_dict // 2)

    x0_canvas = int(x_canvas - dr_dict)
    # placement outside right boundary?
    if x0_canvas >= dx_canvas:
        return None

    x1_canvas = int(x_canvas + dr_dict + (dx_dict % 2))
    # placement outside left boundary?
    if x1_canvas <= 0:
        return None

    # clip to the left?
    if x0_canvas < 0:
        x0_dict = -x0_canvas
        x0_canvas = 0
    else:
        x0_dict = 0

    # clip to the right?
    if x1_canvas > dx_canvas:
        x1_dict = dx_dict - (x1_canvas - dx_canvas)
        x1_canvas = dx_canvas
    else:
        x1_dict = dx_dict

    # print(x0_canvas, x1_canvas, x0_dict, x1_dict)
    assert (x1_canvas - x0_canvas) == (x1_dict - x0_dict)

    return x0_canvas, x1_canvas, x0_dict, x1_dict


def BuildImage(
    canvas_size: torch.Size,
    dictionary: torch.Tensor,
    position_found: torch.Tensor,
    default_dtype,
    torch_device,
):

    assert position_found is not None
    assert dictionary is not None

    canvas_size_copy = torch.tensor(canvas_size)
    assert canvas_size_copy.shape[0] == 4
    canvas_size_copy[1] = 1
    output = torch.zeros(
        canvas_size_copy.tolist(),
        device=torch_device,
        dtype=default_dtype,
    )

    dx_canvas = canvas_size[-2]
    dy_canvas = canvas_size[-1]
    dx_dict = dictionary.shape[-2]
    dy_dict = dictionary.shape[-1]

    for pattern_id in range(0, position_found.shape[0]):
        for patch_id in range(0, position_found.shape[1]):

            x_canvas = position_found[pattern_id, patch_id, 1]
            y_canvas = position_found[pattern_id, patch_id, 2]

            xv = clip_coordinates(x_canvas, dx_canvas, dx_dict)
            if xv == None:
                break

            yv = clip_coordinates(y_canvas, dy_canvas, dy_dict)
            if yv == None:
                break

            if dictionary.shape[0] > 1:
                elem_idx = int(position_found[pattern_id, patch_id, 0])
            else:
                elem_idx = 0

            output[pattern_id, 0, xv[0] : xv[1], yv[0] : yv[1]] += dictionary[
                elem_idx, 0, xv[2] : xv[3], yv[2] : yv[3]
            ]

    return output
