import numpy as np
import torch
import torch.nn as nn
import scipy
from tqdm import tqdm
from scipy.optimize import linprog, minimize
from scipy.spatial.transform import Rotation as R
import sys
import math

def get_smooth_depth_kernel():
    kernel_size = 49
    sigma = 50
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size // 2) * 1.0
    var = sigma ** 2

    gaussian_kernel = (1./(2.*math.pi*var)) * torch.exp(\
                        -torch.sum((xy_grid-mean)**2., dim=-1) / (2*var))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    gaussian_filters = nn.Sequential(nn.ReflectionPad2d(kernel_size//2), gaussian_filter)
    return gaussian_filters

def mat2vec(poses_mat, rotation_mode):
    
    # r = R.from_dcm(poses_mat[:, :3, :3])
    r = R.from_matrix(poses_mat[:, :3, :3])
    if rotation_mode == 'euler':
        r_vec = r.as_euler('xyz')
    else:
        r_vec = r.as_quat()
    t_vec = poses_mat[:, :3, 3]
    vec = np.concatenate([t_vec, r_vec], axis=1)
    return vec

def vec2mat(poses_vec, rotation_mode):
    if rotation_mode == 'euler':
        r = R.from_euler('xyz', poses_vec[:, 3:])
    elif rotation_mode == 'quat':
        r = R.from_quat(poses_vec[:, 3:])
    # r_mat = r.as_dcm()
    r_mat = r.as_matrix()
    mat = np.concatenate([r_mat, np.expand_dims(poses_vec[:, :3], 2)], axis=2)
    mat = np.concatenate([mat, np.zeros((poses_vec.shape[0], 1, 4))], axis=1)
    mat[:, 3, 3] = 1.
    return mat

def inverse_posemat(posemat):
    R = posemat[:, :3, :3]
    t = posemat[:, :3, 3:]
    R_T = np.transpose(R, (0, 2, 1))
    t_inv = -R_T @ t
    pose_inv = np.concatenate([R_T, t_inv], axis=2)
    bot = np.zeros([posemat.shape[0], 1, 4])
    bot[:, :, -1] = 1.
    pose_inv = np.concatenate([pose_inv, bot], axis=1)
    return pose_inv

def weighted_pose(window, weights, rotation_mode):
    t = np.average(window[:, :3], axis=0, weights=weights)
    if rotation_mode == 'quat':
        r = R.from_quat(window[:, 3:]).mean(weights=weights).as_quat()
    else:
        r = R.from_euler('xyz', window[:, 3:]).mean(weights=weights).as_euler('xyz')
    pose = np.concatenate([t, r], axis=0)
    return pose

def smooth_trajectory(raw_frames, smooth_window=59, stability=5):
    poses_mat = np.array([frame['transform_matrix'] for frame in raw_frames])

    poses_vec = mat2vec(poses_mat, rotation_mode='euler')
    
    window = smooth_window
    half = window // 2

    L = poses_vec.shape[0]
    
    smooth_vec = poses_vec.copy()
    for t in range(0, L):
        window_begin, window_end = max(0, t - half), min(L, t + half + 1)
        min_half = min(t - window_begin, window_end - 1 - t)
        window_begin, window_end = t - min_half, t + min_half + 1 
        weights = scipy.signal.windows.gaussian(2 * min_half + 1, stability)
        weights /= weights.sum()

        vec_window = poses_vec[window_begin:window_end]
        smooth_vec[t] = weighted_pose(vec_window, weights, rotation_mode='euler')
    
    smooth_mat = vec2mat(smooth_vec, rotation_mode='euler')
    compensate_mat = inverse_posemat(smooth_mat) @ poses_mat
    
    smooth_frames = []
    for t in range(0, L):
        smooth_frames.append(smooth_mat[t])

    return smooth_frames
