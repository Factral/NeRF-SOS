import os
import json
import numpy as np
import torch
import cv2
from pathlib import Path

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_spectral_data(basedir):
    """
    Load spectral data from the nerfacto format dataset.
    
    Args:
        basedir (str): Base directory of the dataset.
    
    Returns:
        tuple: Contains the following elements:
            - imgs (np.ndarray): Loaded images.
            - poses (np.ndarray): Camera poses.
            - render_poses (torch.Tensor): Poses for rendering.
            - hwf (list): Height, width, and focal length.
            - K (np.ndarray): Intrinsic matrix.
            - i_split (list): Train/val/test split indices.
    """
    transforms_file = os.path.join(basedir, 'transforms.json')
    with open(transforms_file, 'r') as f:
        meta = json.load(f)

    w = meta['w']
    h = meta['h']
    fl_x = meta['fl_x']
    fl_y = meta['fl_y']
    cx = meta['cx']
    cy = meta['cy']

    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])

    frames = meta['frames']
    imgs = []
    poses = []
    i_split = [[], [], []]  # train, val, test

    for i, frame in enumerate(frames):
        fname = os.path.join(basedir, frame['file_path'])
        imgs.append(cv2.imread(fname)[:, :, ::-1] / 255.0)
        poses.append(np.array(frame['transform_matrix']))
        
        # Assuming a simple 80-10-10 split for train-val-test
        if i < int(len(frames) * 0.8):
            i_split[0].append(i)
        elif i < int(len(frames) * 0.9):
            i_split[1].append(i)
        else:
            i_split[2].append(i)

    imgs = np.stack(imgs, 0)
    poses = np.stack(poses, 0)

    # based on load_tankstemple.py line 32
    # why dont we take the mean of the focal length in x and y?
    focal = float(K[0,0]) 

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    return imgs, poses, render_poses, [h, w, focal], K, i_split