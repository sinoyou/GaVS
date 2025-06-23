import os
import random
import torch
import json
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from tqdm import tqdm

from datasets.data import pil_loader
from datasets.colmap_misc import load_sparse_pcl_colmap, read_colmap_pose, get_sparse_depth
from datasets.smooth import smooth_trajectory
from misc.depth import estimate_depth_scale
from datasets.size_tool import rescale_and_crop_field

class GaVSDataset(data.Dataset):
    def __init__(self, cfg, split, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.split = split

        self.data_dir = Path(cfg.dataset.data_path)
        print(f"Loading dataset from {self.data_dir}")

        # record padding
        self.board_aug_pad = self.cfg.dataset.pad_border_aug
        if self.cfg.dataset.pad_border_aug != 0:
            assert self.cfg.dataset.use_inpainted_images, "Padding border augmentation is only supported for inpainted images."
            print('padding boarder', self.board_aug_pad)
        
        # load json file 
        with open(self.data_dir / "blender.json", "r") as f:
            self.data = json.load(f)
            # sort the list of dicts by key colmap_id
            self.data['raw_frames'] = sorted(self.data['raw_frames'], key=lambda img: img['file_path'])

        # novel frame config
        self.novel_frames = list(cfg.model.gauss_novel_frames)

        # stable frames cache
        self.stable_frames = []

        # mod 32 image resolution
        self.mod32_w = cfg.dataset.width
        self.mod32_h = cfg.dataset.height
        print(f"expected mod32 image resolution: {self.mod32_w}x{self.mod32_h}")

        # stability
        self.stability = cfg.dataset.stability
        print(f"dataset loading stability: {self.stability}")

        # depth precompute
        self.precompute_depth()

        # depth scale information
        self.depth_scale = cfg.dataset.predefined_depth_scale
        if self.depth_scale:
            print(f"Using predefined depth scale: {self.depth_scale}")
        else:
            self.prepare_colmap_data()
            scale = self.estimate_depth_scale()
            self.depth_scale = scale

    @property
    def src_size(self):
        if self.cfg.dataset.use_inpainted_images:
            return self.mod32_h + self.cfg.dataset.pad_border_aug * 2, self.mod32_w + self.cfg.dataset.pad_border_aug * 2
        else:
            return self.mod32_h, self.mod32_w
    
    @property
    def tgt_size(self):
        return self.mod32_h, self.mod32_w

    def precompute_depth(self):
        print('precompute depth for all frames ... ')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        unidepth = torch.hub.load(
            "lpiccinelli-eth/UniDepth", "UniDepth", version=self.cfg.model.depth.version, 
            backbone=self.cfg.model.depth.backbone, pretrained=True, trust_repo=True, 
            force_reload=False
        ).to(device)

        # image intrinsics
        K_ndc = self.get_ndc_intrinsics_before_crop(meta_data=self.data)
        if self.cfg.dataset.use_inpainted_images:
            K_ndc[0, 2] *= (self.mod32_w + 2 * self.board_aug_pad) / self.mod32_w
            K_ndc[1, 2] *= (self.mod32_h + 2 * self.board_aug_pad) / self.mod32_h

        # list files
        if self.cfg.dataset.use_inpainted_images:
            src_img_dir = self.data_dir / 'inpainted' / 'images'
        else:
            src_img_dir = self.data_dir / 'images'
        
        # compute depths and save
        self.depths = {}
        image_names = sorted(os.listdir(src_img_dir))
        print(f"Found {len(image_names)} images in {src_img_dir}")
        for image_name in tqdm(image_names):
            src_img_path = src_img_dir / image_name
            src_img = T.ToTensor()(pil_loader(src_img_path)).unsqueeze(0).to(device)  # 1, C, H, W
            # infer depth
            with torch.no_grad():
                depth_outs = unidepth.infer(src_img, intrinsics=K_ndc.unsqueeze(0).to(device))
            depth = depth_outs['depth'][0]
            # save depth
            self.depths[src_img_path.stem] = depth.cpu().squeeze()  # H, W

    def estimate_depth_scale(self, k=64):
        print('calling metric depth and compute the depth scale from colmap sparse results ... ')
        random.seed(1737)
        random_indices = random.sample(range(len(self)), k)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        scales = []
        for idx in tqdm(random_indices):
            data = self.__getitem__(idx)
            depth = data[('unidepth', 0, 0)]
            scale = estimate_depth_scale(depth, data["depth_sparse", 0].to(device))
            scales.append(scale.item())

        print(sorted(scales))
        scale = np.mean(scales)
        print('use median depth scale for all frames:', scale)
        
        return scale

    def get_ndc_intrinsics_before_crop(self, meta_data):
        K = torch.zeros((3, 3))
        K[0, 0] = meta_data['fl_x'] / meta_data['w']
        K[1, 1] = meta_data['fl_y'] / meta_data['h']
        K[0, 2] = meta_data['cx'] / meta_data['w']
        K[1, 2] = meta_data['cy'] / meta_data['h']
        K[2, 2] = 1.0
        return K

    def get_camera_intrinsics(self, K: torch.Tensor):
        # K_scale_target
        K_scale_target = K.clone()
        K_scale_target[0, :] *= self.mod32_w
        K_scale_target[1, :] *= self.mod32_h

        # K_scale_source
        K_scale_source = K.clone()
        K_scale_source[0, 0] *= self.mod32_w
        K_scale_source[1, 1] *= self.mod32_h
        K_scale_source[0, 2] *= (self.mod32_w + 2 * self.board_aug_pad)
        K_scale_source[1, 2] *= (self.mod32_h + 2 * self.board_aug_pad)

        # inv_k_source
        inv_K_source = torch.linalg.inv(K_scale_source)

        return K_scale_target, K_scale_source, inv_K_source

    def prepare_colmap_data(self):
        colmap_dir = self.data_dir / 'sparse'
        if (colmap_dir / '0').exists():
            colmap_dir = colmap_dir / '0'
        self.colmap_sparse_pcl = load_sparse_pcl_colmap(colmap_dir)
        self.colmap_pose_data = {'poses': []}
        for image_k in sorted(self.colmap_sparse_pcl['images'].keys()):
            image = self.colmap_sparse_pcl['images'][image_k]
            self.colmap_pose_data['poses'].append(read_colmap_pose(image))

    def get_sparse_depth_from_colmap(self, idx, raw_size):
        # normal data
        pose_data = self.colmap_pose_data
        sparse_pcl = self.colmap_sparse_pcl
        frame_idx = idx

        # consider center cropping effects
        W_raw, H_raw = raw_size
        H_src, W_src= self.src_size
        scale = min(W_raw / W_src, H_raw / H_src)
        W_crop, H_crop = round((W_src - 2 * self.board_aug_pad) * scale), round((H_src - 2 * self.board_aug_pad) * scale)

        crop_size = (W_crop, H_crop)

        sparse_depth = get_sparse_depth(pose_data, crop_size, sparse_pcl, frame_idx)  # N, 3

        # in the case of padding, we need to shift the x and y coordinates (x, y are in range [-1, 1])
        if self.board_aug_pad > 0:
            sparse_depth[:, 0] *= W_src / (W_src + 2 * self.board_aug_pad)
            sparse_depth[:, 1] *= H_src / (H_src + 2 * self.board_aug_pad)

        return sparse_depth

    def get_depth_scale(self):
        return {'depth_scale': self.depth_scale}

    def __len__(self):
        return len(self.data['raw_frames'])

    def __getitem__(self, idx):
        # check idx size
        assert idx < len(self), f"Index {idx} out of range {len(self)}"

        # load image and pose
        inputs = {}

        # get ndc intrincis before crop
        meta_data = self.data
        K_ndc = self.get_ndc_intrinsics_before_crop(meta_data=meta_data)

        # src image and mask
        src_img_path = self.data_dir / meta_data['raw_frames'][idx]['file_path']
        if self.cfg.dataset.use_inpainted_images:
            src_img_path = src_img_path.parent.parent / 'inpainted' / 'images' / src_img_path.name
            src_mask_path = src_img_path.parent.parent / 'fg_mask' / src_img_path.name
        src_img = T.ToTensor()(pil_loader(src_img_path))
        src_mask = T.ToTensor()(pil_loader(src_mask_path))
        assert src_img.shape[1] == src_mask.shape[1] and src_img.shape[2] == src_mask.shape[2], f"Image and mask size not match {src_img.shape} vs {src_mask.shape}"
        H_input, W_input = src_img.shape[1], src_img.shape[2]
        # scale and crop
        src_img_crop, K_ndc_crop, src_mask_crop = rescale_and_crop_field(src_img, self.src_size, K_ndc, mask=src_mask)
        inputs[("frame_id", 0)] = src_img_path.stem
        inputs[("K_tgt", 0)], inputs[("K_src", 0)], inputs[("inv_K_src", 0)] = self.get_camera_intrinsics(K_ndc_crop)
        inputs[("color", 0, 0)] = src_img_crop[..., self.cfg.dataset.pad_border_aug: -self.cfg.dataset.pad_border_aug, self.cfg.dataset.pad_border_aug: -self.cfg.dataset.pad_border_aug]
        inputs[("color_aug", 0, 0)] = src_img_crop
        inputs[("fg_mask", 0)] = src_mask_crop[..., self.cfg.dataset.pad_border_aug: -self.cfg.dataset.pad_border_aug, self.cfg.dataset.pad_border_aug: -self.cfg.dataset.pad_border_aug]
        inputs[("fg_mask_aug", 0)] = src_mask_crop
        
        # pose and scale
        c2w = torch.tensor(meta_data['raw_frames'][idx]['transform_matrix'])
        c2w_opencv = c2w @ torch.tensor([[1., 0, 0, 0], [0, -1., 0, 0], [0, 0, -1., 0], [0, 0, 0, 1.]])
        inputs[("T_c2w", 0)] = c2w_opencv
        inputs[("T_w2c", 0)] = c2w_opencv.inverse()
        if self.depth_scale: # if depth scale is defined, then no need to prepare sparse depth
            inputs[('scale_colmap', 0)] = self.depth_scale
        else:
            inputs[("depth_sparse", 0)] = self.get_sparse_depth_from_colmap(idx, raw_size=(W_input, H_input))

        # depth data
        mono_depth_crop, _ = rescale_and_crop_field(self.depths[src_img_path.stem].unsqueeze(0), self.src_size, K_ndc)
        inputs[('unidepth', 0, 0)] = mono_depth_crop

        # target image
        if self.split == 'train':
            num_novel_views = len(self.novel_frames)
            idx_candidates = []
            window_size = 10
            if idx > 0:
                idx_candidates += list(range(max(0, idx - window_size), idx))
            if idx < len(self) - 1:
                idx_candidates += list(range(idx + 1, min(len(self), idx + window_size)))
            
            tgt_idx_list = random.sample(idx_candidates, num_novel_views)

            # prepare data for computing flows
            flow_src_imgs, flow_tgt_imgs = [], []
            
            for i, tgt_idx in enumerate(tgt_idx_list):
                tgt_img_path = self.data_dir / meta_data['raw_frames'][tgt_idx]['file_path']
                tgt_mask_path = tgt_img_path.parent.parent / 'fg_mask' / tgt_img_path.name
                tgt_img = T.ToTensor()(pil_loader(tgt_img_path))
                tgt_mask = T.ToTensor()(pil_loader(tgt_mask_path))
                # rescale and crop
                tgt_img_crop, K_ndc_crop, tgt_mask_crop = rescale_and_crop_field(tgt_img, self.tgt_size, K_ndc, mask=tgt_mask)
                inputs[("frame_id", i + 1)] = tgt_img_path.stem
                inputs[("K_tgt", i + 1)], inputs[("K_src", i + 1)], inputs[("inv_K_src", i + 1)] = self.get_camera_intrinsics(K_ndc_crop)
                inputs[("color", i + 1, 0)] = tgt_img_crop
                inputs[("color_aug", i + 1, 0)] = tgt_img_crop
                c2w = torch.tensor(meta_data['raw_frames'][tgt_idx]['transform_matrix'])
                c2w_opencv = c2w @ torch.tensor([[1., 0, 0, 0], [0, -1., 0, 0], [0, 0, -1., 0], [0, 0, 0, 1.]])
                inputs[("T_c2w", i + 1)] = c2w_opencv
                inputs[("T_w2c", i + 1)] = c2w_opencv.inverse()
                inputs[("fg_mask", i + 1)] = tgt_mask_crop

                flow_src_imgs.append(inputs[("color", 0, 0)])
                flow_tgt_imgs.append(inputs[("color", i + 1, 0)])
            
            inputs['flow_src_imgs'] = torch.stack(flow_src_imgs, dim=0)
            inputs['flow_tgt_imgs'] = torch.stack(flow_tgt_imgs, dim=0)
        else:
            if len(self.stable_frames) == 0:
                self.stable_frames = smooth_trajectory(meta_data['raw_frames'], smooth_window=self.cfg.dataset.stability_window, stability=self.stability)

            # placeholder for target image
            tgt_img = torch.zeros_like(inputs[("color", 0, 0)])
            inputs[("K_tgt", 1)], inputs[("K_src", 1)], inputs[("inv_K_src", 1)] = self.get_camera_intrinsics(K_ndc_crop)
            inputs[("color", 1, 0)] = tgt_img
            inputs[("color_aug", 1, 0)] = tgt_img
            c2w_stable = torch.tensor(self.stable_frames[idx], dtype=torch.float32)
            c2w_stable_opencv = c2w_stable @ torch.tensor([[1., 0, 0, 0], [0, -1., 0, 0], [0, 0, -1., 0], [0, 0, 0, 1.]])
            inputs[("T_c2w", 1)] = c2w_stable_opencv
            inputs[("T_w2c", 1)] = c2w_stable_opencv.inverse()

        return inputs
