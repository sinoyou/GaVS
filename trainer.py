import time
import torch
import torch.nn as nn
import numpy as np
import wandb
from einops import rearrange

from models.model import GaussianPredictor
from models.encoder.layers import SSIM
from models.encoder.layers import Project3DSimple, BackprojectDepth

from preprocess.raft.preprocessing_raft import compute_bidirectional_optical_flow
from misc.depth import normalize_depth_for_display
from misc.util import sec_to_hm_str, forward_flow_warp, get_fg_mask, warp_image

class Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.step = 0
        self.model = GaussianPredictor(cfg)
        if cfg.loss.ssim.weight > 0:
            self.ssim = SSIM()

        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def forward(self, inputs, sampler):
        outputs = self.model.forward(inputs)
        losses = self.compute_losses(inputs, outputs, sampler)
        return losses, outputs
    
    def compute_reconstruction_loss(self, pred, target, losses):
        """Computes reprojection loss between a batch of predicted and target images
        """
        cfg = self.cfg
        rec_loss = 0.0
        # pixel level loss
        if cfg.loss.mse.weight > 0:
            if cfg.loss.mse.type == "l1":
                mse_loss = (pred-target).abs()
            elif cfg.loss.mse.type == "l2":
                mse_loss = ((pred-target)**2)
            losses["loss/mse"] = mse_loss.mean()
            rec_loss += cfg.loss.mse.weight * losses["loss/mse"]
        
        # patch level loss
        if cfg.loss.ssim.weight > 0:
            ssim_loss = self.ssim(pred, target)
            losses["loss/ssim"] = ssim_loss.mean()
            rec_loss += cfg.loss.ssim.weight * losses["loss/ssim"]
        
        return rec_loss

    def compute_depth_warping_flow(self, inputs, outputs, frame_id):
        # frame_id = 2 # frame id 1 is always the src frame
        B, _, H, W = inputs['color_aug', 0, 0].shape
        _, _, H_out, W_out = inputs['color', 0, 0].shape
        projector = Project3DSimple(batch_size=B, height=H_out, width=W_out)
        backprojector = BackprojectDepth(batch_size=B, height=H, width=W)

        # src_points = outputs['unidepth-points']  # B, 3, H, W
        src_points = backprojector(outputs['unidepth'], inv_K = inputs[('inv_K_src', 0)])  # B, 3, H*W

        # remove points in the padding region
        if self.cfg.dataset.pad_border_aug > 0:
            src_points = src_points.view(B, -1, H, W)
            src_points = src_points[:, :, self.cfg.dataset.pad_border_aug:-self.cfg.dataset.pad_border_aug, self.cfg.dataset.pad_border_aug:-self.cfg.dataset.pad_border_aug]
            src_points = src_points.reshape(B, -1, H_out * W_out)

        src_tgt_K = inputs['K_tgt', 0]  # B, 3, 3
        near_tgt_K = inputs[('K_tgt', frame_id)]  # B, 3, 3
        key_T_near = outputs[('cam_T_cam', 0, frame_id)]  # B, 4, 4

        # transform points to near frame camera coordinates
        src_points_near = torch.einsum('bij, bjl -> bil', key_T_near, src_points)  # B, 4, H*W
        src_points_near = src_points_near[:, :3, :]  # B, 3, H*W

        # project points to near frame image plane
        src_pix_coords = projector(src_points[:, :3], src_tgt_K)  # B, H, W, 2
        src_pix_coords_near = projector(src_points_near, near_tgt_K)  # B, H, W, 2
        
        # compute flow loss
        computed_flow = src_pix_coords_near - src_pix_coords  # B, H, W, 2
        # flow_loss = ((computed_flow - near_flow) ** 2)

        return computed_flow

    
    def compute_losses(self, inputs, outputs, sampler):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        cfg = self.cfg
        losses = {}
        total_loss = 0.0

        if cfg.model.gaussian_rendering:
            # check middle three frames are consecutive
            if (window_g_lmbd := cfg.loss.window.weight) > 0 and sampler is not None and sampler.stride > 0:
                # infer correspondence from dense optical flow
                middle_idx = len(inputs['frame_id', 0]) // 2
                with torch.no_grad():
                    B = len(inputs['frame_id', 0])
                    middle_fw_flow, middle_bw_flow, middle_fw_occ, middle_bw_occ = compute_bidirectional_optical_flow(
                        inputs[('color_aug', 0, 0)][middle_idx].unsqueeze(0).repeat(B, 1, 1, 1),  # B, 3, H, W
                        inputs[('color_aug', 0, 0)], # B, 3, H, W
                        scale=2, raft_iters=10,
                    )
                    # add original identical coordinates to the flow for backward warping
                    middle_frame_flow = middle_fw_flow.permute(0, 2, 3, 1)  # B, H, W, 2
                    y, x = torch.meshgrid(
                        torch.arange(0, middle_frame_flow.shape[1], device=middle_frame_flow.device),
                        torch.arange(0, middle_frame_flow.shape[2], device=middle_frame_flow.device),
                        indexing='ij'
                    )
                    middle_frame_flow[..., 0] += x.unsqueeze(0)  # add x coordinates
                    middle_frame_flow[..., 1] += y.unsqueeze(0)  # add y coordinates
                    fw_occ_mask = (middle_fw_occ > 0.5).float()  # B, 1, H, W

                # batch size should be odd. 
                assert len(inputs['frame_id', 0]) % 2 == 1, f'batch size should be odd, but got {len(inputs["frame_id", 0])}'
                # prepare neighboring frame's gaussians to the middle frame with backward warping. 
                gauss_opacity = rearrange(outputs["gauss_opacity"], "(b n) c ... -> b (c n) ...", n=cfg.model.gaussians_per_pixel)
                gauss_scaling = rearrange(outputs["gauss_scaling"], "(b n) c ... -> b (c n) ...", n=cfg.model.gaussians_per_pixel)
                gauss_rotation = rearrange(outputs["gauss_rotation"], "(b n) c ... -> b (c n) ...", n=cfg.model.gaussians_per_pixel)
                gauss_feature_dc = rearrange(outputs["gauss_features_dc"], "(b n) c ... -> b (c n) ...", n=cfg.model.gaussians_per_pixel)
                # apply backward warping
                warped_gauss_opacity = warp_image(gauss_opacity, middle_frame_flow)  # B, C, H, W
                warped_gauss_scaling = warp_image(gauss_scaling, middle_frame_flow)  # B, C, H, W
                warped_gauss_rotation = warp_image(gauss_rotation, middle_frame_flow)  # B, C, H, W
                warped_gauss_feature_dc = warp_image(gauss_feature_dc, middle_frame_flow)  # B, C, H, W
                # compute loss with occlusion mask
                valid_mask = (1 - fw_occ_mask).float()
                losses['loss/window_opacity'] = torch.mean((warped_gauss_opacity - gauss_opacity[[middle_idx]].detach()) ** 2 * valid_mask)
                losses['loss/window_scaling'] = torch.mean((warped_gauss_scaling - gauss_scaling[[middle_idx]].detach()) ** 2 * valid_mask)
                losses['loss/window_rotation'] = torch.mean((warped_gauss_rotation - gauss_rotation[[middle_idx]].detach()) ** 2 * valid_mask)
                losses['loss/window_feature_dc'] = torch.mean((warped_gauss_feature_dc - gauss_feature_dc[[middle_idx]].detach()) ** 2 * valid_mask)
                losses['loss/window_loss'] = (losses['loss/window_opacity'] * cfg.loss.window.opacity_weight +
                                              losses['loss/window_scaling'] * cfg.loss.window.scaling_weight +
                                              losses['loss/window_rotation'] * cfg.loss.window.rotation_weight +
                                              losses['loss/window_feature_dc'] * cfg.loss.window.feature_dc_weight)
                total_loss += window_g_lmbd * losses['loss/window_loss']

            # regularize too large gaussian scales
            if (big_g_lmbd := cfg.loss.gauss_scale.weight) > 0:
                scaling = outputs["gauss_scaling"]
                # quantile 99% of scaling
                threshold = torch.quantile(scaling, 0.90)
                big_gaussians = torch.where(scaling > threshold)
                if len(big_gaussians[0]) > 0:
                    big_gauss_reg_loss = torch.mean(scaling[big_gaussians])
                else:
                    big_gauss_reg_loss = 0
                losses["loss/big_gauss_reg_loss"] = big_gauss_reg_loss
                total_loss += big_g_lmbd * big_gauss_reg_loss

            # regularize too big offset
            if cfg.model.predict_offset and (offs_lmbd := cfg.loss.gauss_offset.weight) > 0:
                BL, _, H, W = outputs["gauss_offset"].shape
                offset = outputs["gauss_offset"]  # B*L, 3, H, W
                unidepth = outputs["unidepth"].detach()  # B, 1, H, W
                L = offset.shape[0] // unidepth.shape[0]
                offset = rearrange(offset, "(b l) c h w -> b l c h w", l=L) # B, L, 3, H, W
                unidepth = unidepth.unsqueeze(1)  # B, 1, 1, H, W
                threshold = torch.quantile(offset ** 2, 0.90)
                big_offset = torch.where(offset**2 > threshold)
                
                if len(big_offset[0]) > 0:
                    big_offset_reg_loss = torch.mean(offset[big_offset]**2)
                else:
                    big_offset_reg_loss = 0.0
                losses["loss/gauss_offset_reg"] = big_offset_reg_loss
                total_loss += offs_lmbd * big_offset_reg_loss

            # regularize too big rendered depth
            if (depth_reg_lmbd := cfg.loss.depth_reg.weight) > 0:
                unidepth = outputs["unidepth"].detach() # B, 1, H, W
                if cfg.dataset.pad_border_aug > 0:
                    unidepth = unidepth[:,:,cfg.dataset.pad_border_aug:-cfg.dataset.pad_border_aug,cfg.dataset.pad_border_aug:-cfg.dataset.pad_border_aug]
                threshold = unidepth * cfg.loss.depth_reg.ratio
                depth_gauss = outputs["depth_gauss", 0, 0] # B, 1, H, W
                error = torch.abs(depth_gauss - unidepth)  # B, 1, H, W
                if cfg.loss.only_fg_regularization:
                    fg_mask = get_fg_mask(inputs[("fg_mask", 0)])
                    big_depth_error = torch.where((error > threshold) * fg_mask)
                else:
                    big_depth_error = torch.where(error > threshold)
                
                if len(big_depth_error[0]) > 0:
                    big_depth_error_reg_loss = torch.mean(error[big_depth_error])
                else:
                    big_depth_error_reg_loss = 0
                
                losses['loss/depth_reg'] = big_depth_error_reg_loss
                total_loss += depth_reg_lmbd * big_depth_error_reg_loss

            # reconstruction loss
            frame_ids = self.model.all_frame_ids(inputs)
            rec_loss = 0

            # src frame bg mask
            src_fg_mask = get_fg_mask(inputs[("fg_mask", 0)])  # B, 1, H, W

            # compute flow if needed
            if cfg.train.handle_dynamic_by_mask or cfg.train.handle_dynamic_by_flow:
                B, N, _, H, W = inputs['flow_src_imgs'].shape  # B, N, C, H, W]
                with torch.no_grad():
                    fw_flows, bw_flows, fw_occs, bw_occs = compute_bidirectional_optical_flow(inputs['flow_src_imgs'].reshape(B*N,3,H,W), inputs['flow_tgt_imgs'].reshape(B*N,3,H,W), scale=2, raft_iters=10)
                fw_flows = fw_flows.reshape(B, N, 2, H, W)
                bw_flows = bw_flows.reshape(B, N, 2, H, W)
                fw_occs = fw_occs.reshape(B, N, 1, H, W)
                bw_occs = bw_occs.reshape(B, N, 1, H, W)

            for frame_id in frame_ids:
                if frame_id != 0:
                    camera_flow = self.compute_depth_warping_flow(inputs, outputs, frame_id)  # (B, H, W, 2)
                    camera_flow = camera_flow.detach()
                    if cfg.train.handle_dynamic_by_mask:
                        # compute dynamic masking
                        tgt_fg_mask = get_fg_mask(inputs[("fg_mask", frame_id)])  # B, 1, H, W
                        src_fg_mask_warp, _ = forward_flow_warp(src_fg_mask, camera_flow)  # B, 1, H, W
                        fg_mask = tgt_fg_mask + src_fg_mask_warp
                        bg_mask = 1 - (fg_mask > 0).float()
                        outputs[("color_loss_pred", frame_id, 0)] = outputs[("color_gauss", frame_id, 0)] * bg_mask
                        outputs[("color_loss_gt", frame_id, 0)] = inputs[("color", frame_id, 0)] * bg_mask
                    elif cfg.train.handle_dynamic_by_flow:
                        forward_flow, backward_flow, fw_occ_mask, bw_occ_mask = fw_flows[:, frame_id - 1], bw_flows[:, frame_id - 1], fw_occs[:, frame_id - 1], bw_occs[:, frame_id - 1]
                        camera_flow = camera_flow.permute(0, 3, 1, 2)  # N, 2, H, W
                        backward_camera_flow, motion_flow_valid_mask = forward_flow_warp(-camera_flow, forward_flow.permute(0, 2, 3, 1))  # N, 2, H, W; N, 1, H, W
                        backward_motion_flow = backward_flow - backward_camera_flow  # N, 2, H, W

                        # warp target frame to non-motion status
                        tgt_dynamic_mask = get_fg_mask(inputs[("fg_mask", frame_id)])  # N, 1, H, W
                        tgt_dynamic_img = inputs[("color", frame_id, 0)] * tgt_dynamic_mask  # N, 3, H, W
                        tgt_img_feature = torch.concat([tgt_dynamic_img, tgt_dynamic_mask, bw_occ_mask], dim=1)  # N, 4, H, W
                        warped_tgt_img_feature, valid_mask = forward_flow_warp(tgt_img_feature, backward_motion_flow.permute(0, 2, 3, 1))  # N, 4, H, W
                        warped_tgt_img = warped_tgt_img_feature[:, :3]  # N, 3, H, W
                        warped_tgt_dynamic_mask = warped_tgt_img_feature[:, [3]]  # N, 1, H, W
                        warped_bw_occ_mask = warped_tgt_img_feature[:, [4]]  # N, 1, H, W

                        # activated warp region
                        warp_fusion_mask = ((tgt_dynamic_mask + warped_tgt_dynamic_mask) > 0.5).float()  # N, 1, H, W
                        warp_fusion_img = warped_tgt_img * warp_fusion_mask + inputs[("color", frame_id, 0)] * (1 - warp_fusion_mask)  # N, 3, H, W
                        # mask annotation 
                        # 1. occulusion region: due to inconsistency (low confidence and occluded regions.)
                        # 2. invalid forward region: due to forward warping may introduce holes. 
                        # 3. boundary region: static scenes should not penerate into dynamic regions (due to flow error). 
                        # invalid_mask_occ = ((bw_occ_mask * (1.0 - warp_fusion_mask) + warped_bw_occ_mask * warp_fusion_mask) > 0.5).float()
                        invalid_mask_occ = warped_bw_occ_mask * warp_fusion_mask
                        invalid_mask_forward = ((1.0 - valid_mask) * warp_fusion_mask > 0.5).float()
                        invalid_mask_boundary = ((1.0 - warped_tgt_dynamic_mask) * tgt_dynamic_mask > 0.5).float() 
                        invalid_comp_region = invalid_mask_occ + invalid_mask_forward + invalid_mask_boundary
                        invalid_comp_region = (invalid_comp_region > 0.5).float()  # N, 1, H, W

                        outputs[("color_loss_pred", frame_id, 0)] = outputs[("color_gauss", frame_id, 0)] * (1 - invalid_comp_region)
                        outputs[("color_loss_gt", frame_id, 0)] = warp_fusion_img * (1 - invalid_comp_region)

                        # supplementary materials
                        outputs[("color_mask_occ", frame_id, 0)] = warp_fusion_img * (1 - invalid_mask_occ)
                        outputs[("color_mask_forward", frame_id, 0)] = warp_fusion_img * (1 - invalid_mask_forward)
                        outputs[("color_mask_boundary", frame_id, 0)] = warp_fusion_img * (1 - invalid_mask_boundary)
                    else:
                        bg_mask = 1 - torch.zeros_like(src_fg_mask).to(src_fg_mask.device)
                        outputs[("color_loss_pred", frame_id, 0)] = outputs[("color_gauss", frame_id, 0)] * bg_mask
                        outputs[("color_loss_gt", frame_id, 0)] = inputs[("color", frame_id, 0)] * bg_mask
                else:
                    bg_mask = 1 - torch.zeros_like(src_fg_mask).to(src_fg_mask.device)
                    outputs[("color_loss_gt", frame_id, 0)] = inputs[("color", frame_id, 0)] * bg_mask
                    outputs[("color_loss_pred", frame_id, 0)] = outputs[("color_gauss", frame_id, 0)] * bg_mask

                # compute gaussian reconstruction loss
                target = outputs[("color_loss_gt", frame_id, 0)]
                pred = outputs[("color_loss_pred", frame_id, 0)]

                rec_loss += self.compute_reconstruction_loss(pred, target, losses)
            
            rec_loss /= len(frame_ids)
            
            losses["loss/rec"] = rec_loss
            total_loss += rec_loss

        losses["loss/total"] = total_loss
        return losses
    
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.cfg.optimiser.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    
    def log_scalars(self, mode, outputs, losses, lr):
        """log the scalars"""
        cfg = self.cfg
        logger = self.logger
        if logger is None:
            return
        
        # print(f"{self.step}: {losses['loss/total']:.4f}")

        logger.log({f"{mode}-scalar/learning_rate": lr}, self.step)
        logger.log({f"{mode}-scalar/{l}": v for l, v in losses.items()}, self.step)
        if cfg.model.gaussian_rendering:
            logger.log({f"{mode}-scalar/gauss-scale-mean": torch.mean(outputs["gauss_scaling"])}, self.step)
            for i in range(self.cfg.model.gaussians_per_pixel):
                # given a matrix shaped as [N, ], only slice the [i, i+k, i+2*k, ...] elements
                logger.log({f"{mode}-scalar/gauss-opacity-{i}": torch.mean(outputs["gauss_opacity"][i::self.cfg.model.gaussians_per_pixel])}, self.step)
            logger.log({f"{mode}-scalar/gauss-depth-inc": torch.mean(outputs["depth_inc", 0])}, self.step)

            if self.cfg.model.predict_offset:
                offset_mag = torch.linalg.vector_norm(outputs["gauss_offset"], dim=1)
                mean_offset = offset_mag.mean()
                logger.log({f"{mode}-scalar/gauss-offset-mean": mean_offset}, self.step)
        
        if cfg.dataset.scale_pose_by_depth:
            depth_scale = outputs[("depth_scale", 0)]
            logger.log({f"{mode}-scalar/depth_scale": depth_scale.mean().item()}, self.step)

    def log(self, mode, inputs, outputs):
        """Write images to Neptune
        """
        cfg = self.cfg
        frame_ids = self.model.all_frame_ids(inputs)
        scales = cfg.model.scales
        logger = self.logger
        if logger is None:
            return
        
        for j in range(min(1, cfg.data_loader.batch_size)): # write a maxmimum of 4 images
            for s in scales:
                assert cfg.model.gaussian_rendering
                color = {}
                masks = {}
                opacity = {}
                
                for frame_id in frame_ids:
                    # warpped / masked images for loss computing
                    color_loss_gt = outputs[("color_loss_gt", frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
                    color_loss_pred = outputs[("color_loss_pred", frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
                    # rendered original images
                    color_aug = inputs[("color", frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
                    color_pred = outputs[("color_gauss", frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
                    # concat then horizontally
                    color_loss_concat = np.concatenate((color_loss_pred, color_loss_gt), axis=1)
                    color_original_concat = np.concatenate((color_pred, color_aug), axis=1)
                    color_concat = np.concatenate((color_loss_concat, color_original_concat), axis=0)
                    color[f"{mode}-color/{j}/{frame_id}"] = wandb.Image(color_concat)

                    # masks data
                    if ('color_mask_occ', frame_id, 0) in outputs:
                        color_mask_occ = outputs[('color_mask_occ', frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
                        color_mask_forward = outputs[('color_mask_forward', frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
                        color_mask_boundary = outputs[('color_mask_boundary', frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
                        mask_concat = np.concatenate((color_mask_occ, color_mask_forward, color_mask_boundary), axis=1)
                        masks[f"{mode}-mask/{j}/{frame_id}"] = wandb.Image(mask_concat)
                
                logger.log(color, self.step)
                if masks:    
                    logger.log(masks, self.step)

                for i in range(self.cfg.model.gaussians_per_pixel):
                    opacity[f"{mode}-gauss_opacity_gaussian/{i}/{j}"] = wandb.Image(outputs["gauss_opacity"][j * self.cfg.model.gaussians_per_pixel + i].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy())
                logger.log(opacity, self.step)

                depth = rearrange(outputs[("depth", 0)], "(b n) ... -> b n ...", n=self.cfg.model.gaussians_per_pixel)
                depth_sliced = depth[j][0].detach().cpu().numpy()
                depth_img, normalizer = normalize_depth_for_display(depth_sliced, return_normalizer=True)
                depth_img = np.clip(depth_img, 0, 1)

                logger.log({f"{mode}-depth/{s}/{j}": wandb.Image(depth_img)}, self.step)