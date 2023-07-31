import torch
import os
import torchvision as tv
import time

from models.yolo import Detect, Model

# Warning: The line "#matplotlib.use('Agg')  # for writing to files only"
# in utils/plots.py prevents the further use of matplotlib


class Yolo5Segmentation(torch.nn.Module):

    default_dtype = torch.float32

    conf: float = 0.25  # NMS confidence threshold
    iou: float = 0.45  # NMS IoU threshold
    agnostic: bool = False  # NMS class-agnostic
    multi_label: bool = False  # NMS multiple labels per box
    max_det: int = 1000  # maximum number of detections per image
    number_of_maps: int = 32
    imgsz: tuple[int, int] = (640, 640)  # inference size (height, width)

    device: torch.device = torch.device("cpu")
    weigh_path: str = ""

    class_names: dict
    stride: int

    found_class_id: torch.Tensor | None = None

    def __init__(self, mode: int = 3, torch_device: str = "cpu"):
        super().__init__()

        model_pretrained_path: str = "segment_pretrained"
        assert mode < 5
        assert mode >= 0
        if mode == 0:
            model_pretrained_weights: str = "yolov5n-seg.pt"
        elif mode == 1:
            model_pretrained_weights = "yolov5s-seg.pt"
        elif mode == 2:
            model_pretrained_weights = "yolov5m-seg.pt"
        elif mode == 3:
            model_pretrained_weights = "yolov5l-seg.pt"
        elif mode == 4:
            model_pretrained_weights = "yolov5x-seg.pt"

        self.weigh_path = os.path.join(model_pretrained_path, model_pretrained_weights)

        self.device = torch.device(torch_device)

        self.network = self.attempt_load(
            self.weigh_path, device=self.device, inplace=True, fuse=True
        )
        self.stride = max(int(self.network.stride.max()), 32)  # model stride
        self.network.float()
        self.class_names = dict(self.network.names)  # type: ignore

    # classes: (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    def forward(self, input: torch.Tensor, classes=None) -> torch.Tensor:

        assert input.ndim == 4
        assert input.shape[0] == 1
        assert input.shape[1] == 3

        input_resized, (
            remove_left,
            remove_top,
            remove_height,
            remove_width,
        ) = self.scale_and_pad(
            input,
        )

        network_output = self.network(input_resized)
        number_of_classes = network_output[0].shape[2] - self.number_of_maps - 5
        assert len(self.class_names) == number_of_classes

        maps = network_output[1]

        # results matrix:
        # Fist Dimension:
        # Image Number
        # ...
        # Last Dimension:
        # center_x: 0
        # center_y: 1
        # width: 2
        # height: 3
        # obj_conf (object): 4
        # cls_conf (class): 5

        results = non_max_suppression(
            network_output[0],
            self.conf,
            self.iou,
            classes,
            self.agnostic,
            self.multi_label,
            max_det=self.max_det,
            nm=self.number_of_maps,
        )

        image_id = 0

        if results[image_id].shape[0] > 0:
            masks = self.process_mask_native(
                maps[image_id],
                results[image_id][:, 6:],
                results[image_id][:, :4],
            )
            self.found_class_id = results[image_id][:, 5]

            output = tv.transforms.functional.resize(
                tv.transforms.functional.crop(
                    img=masks,
                    top=int(remove_top),
                    left=int(remove_left),
                    height=int(remove_height),
                    width=int(remove_width),
                ),
                size=(input.shape[-2], input.shape[-1]),
            )
        else:
            output = None
            self.found_class_id = None

        return output

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    # code stolen and/or modified from Yolov5 ->
    def scale_and_pad(
        self,
        input,
    ):
        ratio = min(self.imgsz[0] / input.shape[-2], self.imgsz[1] / input.shape[-1])

        shape_new_x = int(input.shape[-2] * ratio)
        shape_new_y = int(input.shape[-1] * ratio)

        dx = self.imgsz[0] - shape_new_x
        dy = self.imgsz[1] - shape_new_y

        dx_0 = dx // 2
        dy_0 = dy // 2

        image_resize = tv.transforms.functional.pad(
            tv.transforms.functional.resize(input, size=(shape_new_x, shape_new_y)),
            padding=[dy_0, dx_0, int(dy - dy_0), int(dx - dx_0)],
            fill=float(114.0 / 255.0),
        )

        return image_resize, (dy_0, dx_0, shape_new_x, shape_new_y)

    def process_mask_native(self, protos, masks_in, bboxes):
        masks = (
            (masks_in @ protos.float().view(protos.shape[0], -1))
            .sigmoid()
            .view(-1, protos.shape[1], protos.shape[2])
        )

        masks = torch.nn.functional.interpolate(
            masks[None],
            (self.imgsz[0], self.imgsz[1]),
            mode="bilinear",
            align_corners=False,
        )[
            0
        ]  # CHW

        x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = torch.arange(masks.shape[2], device=masks.device, dtype=x1.dtype)[
            None, None, :
        ]  # rows shape(1,w,1)
        c = torch.arange(masks.shape[1], device=masks.device, dtype=x1.dtype)[
            None, :, None
        ]  # cols shape(h,1,1)

        masks = masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

        return masks.gt_(0.5)

    def attempt_load(self, weights, device=None, inplace=True, fuse=True):
        # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a

        model = Ensemble()
        for w in weights if isinstance(weights, list) else [weights]:
            ckpt = torch.load(w, map_location="cpu")  # load
            ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

            # Model compatibility updates
            if not hasattr(ckpt, "stride"):
                ckpt.stride = torch.tensor([32.0])
            if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
                ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

            model.append(
                ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval()
            )  # model in eval mode

        # Module compatibility updates
        for m in model.modules():
            t = type(m)
            if t in (
                torch.nn.Hardswish,
                torch.nn.LeakyReLU,
                torch.nn.ReLU,
                torch.nn.ReLU6,
                torch.nn.SiLU,
                Detect,
                Model,
            ):
                m.inplace = inplace  # torch 1.7.0 compatibility
                if t is Detect and not isinstance(m.anchor_grid, list):
                    delattr(m, "anchor_grid")
                    setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
            elif t is torch.nn.Upsample and not hasattr(m, "recompute_scale_factor"):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        # Return model
        if len(model) == 1:
            return model[-1]

        # Return detection ensemble
        print(f"Ensemble created with {weights}\n")
        for k in "names", "nc", "yaml":
            setattr(model, k, getattr(model[0], k))
        model.stride = model[
            torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
        ].stride  # max stride
        assert all(
            model[0].nc == m.nc for m in model
        ), f"Models have different class counts: {[m.nc for m in model]}"
        return model


class Ensemble(torch.nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4]
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = tv.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def box_iou(box1, box2, eps=1e-7):
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# <- code stolen and/or modified from Yolov5
