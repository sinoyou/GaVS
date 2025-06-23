import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from RAFT import RAFT

def initialize_RAFT(model_path='./optical_flow_raft/weights/raft-things.pth', device='cuda'):
    """Initializes the RAFT model.
    """
    args = argparse.ArgumentParser()
    args.raft_model = model_path
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.raft_model, map_location='cpu'))
    model = model.module

    model.to(device)

    return model


class RAFT_bi(nn.Module):
    """Flow completion loss"""
    def __init__(self, model_path='weights/raft-things.pth', device='cuda'):
        super().__init__()
        # get file's absolute path
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        self.fix_raft = initialize_RAFT(model_path, device=device)

        for p in self.fix_raft.parameters():
            p.requires_grad = False

        self.l1_criterion = nn.L1Loss()
        self.device = device
        self.eval()

    def forward(self, src_frames, tgt_frames, iters=20):
        """Computes the bidirectional optical flow between frames.
        """
        assert src_frames.size() == tgt_frames.size(), 'Input frames must have the same size {} != {}'.format(src_frames.size(), tgt_frames.size())
        l, c, h, w = src_frames.size()

        with torch.no_grad():
            _, gt_flows_forward = self.fix_raft(src_frames, tgt_frames, iters=iters, test_mode=True)
            _, gt_flows_backward = self.fix_raft(tgt_frames, src_frames, iters=iters, test_mode=True)

        gt_flows_forward = gt_flows_forward.view(l, 2, h, w).permute(0, 2, 3, 1)
        gt_flows_backward = gt_flows_backward.view(l, 2, h, w).permute(0, 2, 3, 1)

        return gt_flows_forward, gt_flows_backward