import torch
import torch.nn.functional as F
import os
from pathlib import Path
import cv2
from PIL import Image

# add the dictectory of this file to the python system path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from flow_comp_raft import RAFT_bi

fix_raft = RAFT_bi()

def compute_bidirectional_optical_flow(src_imgs, tgt_imgs, scale=2, raft_iters=10):
    # read src image
    # src_img = Image.fromarray(cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB))

    # read tgt images
    # tgt_imgs = [Image.fromarray(cv2.cvtColor(cv2.imread(tgt_path), cv2.COLOR_BGR2RGB)) for tgt_path in tgt_paths]
    #     
    # size check and resizing
    W, H = src_imgs.shape[-1], src_imgs.shape[-2]
    assert src_imgs.shape == tgt_imgs.shape, "Source and target images must have the same shape."
    H_resized, W_resized = H // scale, W // scale
    H_resized, W_resized = H_resized - H_resized % 8, W_resized - W_resized % 8
    src_imgs = src_imgs.to(fix_raft.device)
    tgt_imgs = tgt_imgs.to(fix_raft.device)

    # raft inference
    # src_imgs = torch.cat(src_imgs, dim=0)  # (N, C, H, W)
    # tgt_imgs = torch.cat(tgt_imgs, dim=0)  # (N, C, H, W)
    flows_f, flows_b = fix_raft(F.interpolate(src_imgs, (H_resized, W_resized), mode='bilinear', align_corners=True), 
                                F.interpolate(tgt_imgs, (H_resized, W_resized), mode='bilinear', align_corners=True), 
                                iters=raft_iters)  # N, H, W, 2

    # consistency mask
    fwd_occ, bwd_occ = forward_backward_consistency_check(flows_f, flows_b)  # (N, H, W, 1)

    # resize
    def resize_flow(flow_data, ori_size):
        if ori_size != (flow_data.shape[1], flow_data.shape[2]):
            inference_size = (flow_data.shape[1], flow_data.shape[2])
            flow_data = flow_data.permute(0, 3, 1, 2)  # N, 2, H, W
            flow_data = F.interpolate(flow_data, ori_size, mode='bilinear', align_corners=True)
            flow_data[:, 0] = flow_data[:, 0] * ori_size[-1] / inference_size[-1]
            flow_data[:, 1] = flow_data[:, 1] * ori_size[-2] / inference_size[-2]
        return flow_data # N, 2, H, W
        
    def resize_occ_mask(occ_mask, ori_size):
        if ori_size != (occ_mask.shape[1], occ_mask.shape[2]):
            occ_mask  = occ_mask.permute(0, 3, 1, 2)  # N, 1, H, W
            occ_mask = F.interpolate(occ_mask, ori_size, mode='bilinear', align_corners=True)
            occ_mask = (occ_mask > 0.5).float()
        return occ_mask
    
    flows_f = resize_flow(flows_f, (H, W))
    flows_b = resize_flow(flows_b, (H, W))
    fwd_occ = resize_occ_mask(fwd_occ, (H, W))
    bwd_occ = resize_occ_mask(bwd_occ, (H, W))

    return flows_f.detach(), flows_b.detach(), fwd_occ.detach(), bwd_occ.detach()

def read_frames(frame_root):
    frames = []
    fr_lst = sorted(os.listdir(frame_root))
    for fr in fr_lst:
        frame = cv2.imread(os.path.join(frame_root, fr))
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(frame)
    size = frames[0].size

    return frames, fr_lst, size


"""
ref: unidepth implementation
"""

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img

def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)

def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=1.0
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    fwd_flow = fwd_flow.permute(0, 3, 1, 2)  # [B, 2, H, W]
    bwd_flow = bwd_flow.permute(0, 3, 1, 2)
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ.unsqueeze(-1), bwd_occ.unsqueeze(-1)  # [B, H, W, 1]