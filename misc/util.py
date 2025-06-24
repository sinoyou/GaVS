import torch
import torch.nn.functional as F

def get_source_frame_ids():
    return [0]

def add_source_frame_id(novel_frames):
    return get_source_frame_ids() + novel_frames

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

def get_fg_mask(fg_mask):
    """
    Convert a foreground mask to a binary mask
    fg_mask: B, H, W, 3
    """
    fg_mask = torch.max(fg_mask, dim=1, keepdim=True)[0]  # B 1 H W
    fg_mask = (fg_mask > 0.5).float()
    return fg_mask.float()

def get_fg_to_bg_mask(fg_mask):
    """
    Convert a foreground mask to a background mask
    fg_mask: B, H, W, 3
    """
    fg_mask = torch.max(fg_mask, dim=1, keepdim=True)[0]  # B 1 H W
    fg_mask = (fg_mask > 0.5).float()
    bg_mask = 1 - fg_mask
    return bg_mask.float()

def warp_image(img, flow, align_corners=True):
    """
    Warp an image using optical flow
    img: B, C, H, W
    flow: B, H, W, 2
    """
    B, C, H, W = img.shape
    grid = flow.permute(0, 3, 1, 2).clone()
    grid[:, 0] = 2 * grid[:, 0] / (W - 1) - 1
    grid[:, 1] = 2 * grid[:, 1] / (H - 1) - 1
    grid = grid.permute(0, 2, 3, 1)
    warped_img = torch.nn.functional.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=align_corners)
    return warped_img   

def forward_flow_warp(frames, flows):
    """
    Forward warp an image using a flow map (PyTorch implementation).
    
    Args:
        frames (torch.Tensor): Input image N x C x H x W.
        flows (torch.Tensor): Flow map (N x H x W x 2), where the last dimension represents (u, v), tensor format.
    
    Returns:
        G (torch.Tensor): Forward warped image (same shape as I).
    """
    ret = []
    forward_mask = []
    for I, flow in zip(frames, flows):
        I = I.permute(1, 2, 0)  # H x W x C
        h, w = I.shape[:2]

        # Create a grid of pixel coordinates
        y, x = torch.meshgrid(torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32), indexing='ij')
        y = y.to(I)
        x = x.to(I)
        
        # Create an empty output image and a weight map
        G = torch.zeros_like(I, dtype=torch.float32).to(I)
        weight_map = torch.zeros((h, w), dtype=torch.float32).to(I)

        x_shift = x + flow[..., 0]
        y_shift = y + flow[..., 1]

        # Compute target coordinates
        for x_round, y_round in [(0, 0), 
                                (0, 1), 
                                (1, 0), 
                                (1, 1)]:
        
            # round indices
            x_new = torch.floor(x_shift + x_round).long()
            y_new = torch.floor(y_shift + y_round).long()

            # area weights
            area_weights = (1 - (x_new.float() - x_shift).abs()) * (1 - (y_new.float() - y_shift).abs())

            # Ensure target coordinates are within bounds
            valid = (x_new >= 0) & (x_new < w) & (y_new >= 0) & (y_new < h)

            # Filter valid coordinates
            x_new = x_new[valid]
            y_new = y_new[valid]
            I_valid = I[valid]
            area_weights_valid = area_weights[valid]

            # Accumulate values at the target coordinates
            for ci in range(I.shape[2]):
                # G[y_new, x_new, ci] += I_valid[..., ci] * area_weights_valid
                G[..., ci].index_put_((y_new, x_new), I_valid[..., ci] * area_weights_valid, accumulate=True)
            
            # G.index_put_((y_new, x_new), I_valid * area_weights_valid, accumulate=True)
            weight_map.index_put_((y_new, x_new), torch.ones_like(I_valid[..., 0], dtype=torch.float32) * area_weights_valid, accumulate=True)

        # Normalize to handle overlaps
        G[weight_map > 0] /= weight_map[weight_map > 0].unsqueeze(-1)
        ret.append(G)
        forward_mask.append(weight_map > 0)
    
    ret = torch.stack(ret, dim=0).permute(0, 3, 1, 2)
    forward_mask = torch.stack(forward_mask, dim=0)
    return ret, forward_mask.float().unsqueeze(1)