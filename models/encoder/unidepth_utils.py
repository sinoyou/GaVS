"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from math import ceil
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange

MAP_BACKBONES = {"ViTL14": "vitl14", "ConvNextL": "cnvnxtl"}
IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)

@torch.jit.script
def generate_rays(
    camera_intrinsics: torch.Tensor, image_shape: Tuple[int, int], noisy: bool = False
):
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    # Calculate ray directions
    intrinsics_inv = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics_inv[:, 0, 0] = 1.0 / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 1] = 1.0 / camera_intrinsics[:, 1, 1]
    intrinsics_inv[:, 0, 2] = -camera_intrinsics[:, 0, 2] / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 2] = -camera_intrinsics[:, 1, 2] / camera_intrinsics[:, 1, 1]
    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )  # (H, W, 3)
    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1)
    )  # (3, H*W)
    ray_directions = F.normalize(ray_directions, dim=1)  # (B, 3, H*W)
    ray_directions = ray_directions.permute(0, 2, 1)  # (B, H*W, 3)

    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    # pitch = torch.asin(ray_directions[..., 1])
    # roll = torch.atan2(ray_directions[..., 0], - ray_directions[..., 1])
    angles = torch.stack([theta, phi], dim=-1)
    return ray_directions, angles


@torch.jit.script
def spherical_zbuffer_to_euclidean(spherical_tensor: torch.Tensor) -> torch.Tensor:
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    z = spherical_tensor[..., 2]  # Extract zbuffer depth

    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    # =>
    # r = z / sin(phi) / cos(theta)
    # y = z / (sin(phi) / cos(phi)) / cos(theta)
    # x = z * sin(theta) / cos(theta)
    x = z * torch.tan(theta)
    y = z / torch.tan(phi) / torch.cos(theta)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor

def _paddings(image_shape, network_shape):
    cur_h, cur_w = image_shape
    h, w = network_shape
    pad_top, pad_bottom = (h - cur_h) // 2, h - cur_h - (h - cur_h) // 2
    pad_left, pad_right = (w - cur_w) // 2, w - cur_w - (w - cur_w) // 2
    return pad_left, pad_right, pad_top, pad_bottom

def _shapes(image_shape, network_shape):
    h, w = image_shape
    input_ratio = w / h
    output_ratio = network_shape[1] / network_shape[0]
    if output_ratio > input_ratio:
        ratio = network_shape[0] / h
    elif output_ratio <= input_ratio:
        ratio = network_shape[1] / w
    return (ceil(h * ratio - 0.5), ceil(w * ratio - 0.5)), ratio

def _preprocess(rgbs, intrinsics, shapes, pads, ratio, output_shapes):
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    rgbs = F.interpolate(
        rgbs, size=shapes, mode="bilinear", align_corners=False, antialias=True
    )
    rgbs = F.pad(rgbs, (pad_left, pad_right, pad_top, pad_bottom), mode="constant")
    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio + pad_left
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio + pad_top
        return rgbs, intrinsics
    return rgbs, None


def _postprocess(predictions, intrinsics, shapes, pads, ratio, original_shapes):
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    # pred mean, trim paddings, and upsample to input dim
    predictions = sum(
        [
            F.interpolate(
                x.clone(),
                size=shapes,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            for x in predictions
        ]
    ) / len(predictions)
    predictions = predictions[
        ..., pad_top : shapes[0] - pad_bottom, pad_left : shapes[1] - pad_right
    ]
    predictions = F.interpolate(
        predictions,
        size=original_shapes,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    intrinsics[:, 0, 0] = intrinsics[:, 0, 0] / ratio
    intrinsics[:, 1, 1] = intrinsics[:, 1, 1] / ratio
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] - pad_left) / ratio
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] - pad_top) / ratio
    return predictions, intrinsics


def infer(self, rgbs: torch.Tensor, intrinsics=None, skip_camera=False):
    if rgbs.ndim == 3:
        rgbs = rgbs.unsqueeze(0)
    if intrinsics is not None and intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    B, _, H, W = rgbs.shape

    rgbs = rgbs.to(self.device)
    if intrinsics is not None:
        intrinsics = intrinsics.to(self.device)

    # process image and intrinsiscs (if any) to match network input (slow?)
    if rgbs.max() > 5 or rgbs.dtype == torch.uint8:
        rgbs = rgbs.to(torch.float32).div(255)
    if rgbs.min() >= 0.0 and rgbs.max() <= 1.0:
        rgbs = TF.normalize(
            rgbs,
            mean=IMAGENET_DATASET_MEAN,
            std=IMAGENET_DATASET_STD,
        )

    (h, w), ratio = _shapes((H, W), self.image_shape)
    pad_left, pad_right, pad_top, pad_bottom = _paddings((h, w), self.image_shape)
    rgbs, gt_intrinsics = _preprocess(
        rgbs,
        intrinsics,
        (h, w),
        (pad_left, pad_right, pad_top, pad_bottom),
        ratio,
        self.image_shape,
    )

    # run encoder
    encoder_outputs, cls_tokens = self.pixel_encoder(rgbs)
    if "dino" in self.pixel_encoder.__class__.__name__.lower():
        encoder_outputs = [
            (x + y.unsqueeze(1)).contiguous()
            for x, y in zip(encoder_outputs, cls_tokens)
        ]

    # get data for decoder and adapt to given camera
    inputs = {}
    inputs["encoder_outputs"] = encoder_outputs
    inputs["cls_tokens"] = cls_tokens
    inputs["image"] = rgbs
    if gt_intrinsics is not None:
        rays, angles = generate_rays(
            gt_intrinsics, self.image_shape, noisy=self.training
        )
        inputs["rays"] = rays
        inputs["angles"] = angles
        inputs["K"] = gt_intrinsics
        self.pixel_decoder.test_fixed_camera = True
        self.pixel_decoder.skip_camera = skip_camera

    # decode all
    pred_intrinsics, predictions, _ = self.pixel_decoder(inputs, {})

    # undo the reshaping and get original image size (slow)
    predictions, pred_intrinsics = _postprocess(
        predictions,
        pred_intrinsics,
        self.image_shape,
        (pad_left, pad_right, pad_top, pad_bottom),
        ratio,
        (H, W),
    )

    # final 3D points backprojection
    intrinsics = gt_intrinsics if gt_intrinsics is not None else pred_intrinsics
    angles = generate_rays(intrinsics, (H, W), noisy=False)[-1]
    angles = rearrange(angles, "b (h w) c -> b c h w", h=H, w=W)
    points_3d = torch.cat((angles, predictions), dim=1)
    points_3d = spherical_zbuffer_to_euclidean(
        points_3d.permute(0, 2, 3, 1)
    ).permute(0, 3, 1, 2)

    # output data
    outputs = {
        "intrinsics": pred_intrinsics,
        "points": points_3d,
        "depth": predictions[:, -1:],
    }
    self.pixel_decoder.test_fixed_camera = False
    self.pixel_decoder.skip_camera = False
    return outputs