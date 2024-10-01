import os, sys
import numpy as np
import json
import random
import time

from tqdm import tqdm, trange
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as st


if __name__ == '__main__':
    sys.path.append('..')

from utils.ray import get_persp_rays, get_persp_intrinsic

from data.load_llff import load_llff_data
from data.load_deepvoxels import load_dv_data
from data.load_LINEMOD import load_LINEMOD_data
from data.load_blender import load_blender_data
from data.load_tankstemple import load_tankstemple_data
from data.load_toydesk import load_toydesk_data
from data.load_toydesk_custom import load_toydesk_custom_data
from data.load_nerfstudio_data import load_nerfstudio_data
from data.utils_ply import save_pc

import configargparse


class AABBBoxCollider():
    """Module for colliding rays with the scene box to compute near and far values.

    Args:
        scene_box: scene box to apply to dataset
    """

    def __init__(self, scene_box=torch.tensor([[-1,-1,-1],[1,1,1]]), near_plane: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scene_box = scene_box
        self.near_plane = near_plane

    def _intersect_with_aabb(
        self, rays_o, rays_d, aabb
    ):
        """Returns collection of valid rays within a specified near/far bounding box along with a mask
        specifying which rays are valid

        Args:
            rays_o: (num_rays, 3) ray origins
            rays_d: (num_rays, 3) ray directions
            aabb: (2, 3) This is [min point (x,y,z), max point (x,y,z)]
        """
        # avoid divide by zero
        dir_fraction = 1.0 / (rays_d + 1e-6)

        # x

        print(rays_o.shape, rays_d.shape, aabb.shape)

        t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        # y
        t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        # z
        t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
        t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

        nears = torch.max(
            torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=1), dim=1
        ).values
        fars = torch.min(
            torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=1), dim=1
        ).values

        # clamp to near plane
        near_plane = self.near_plane
        nears = torch.clamp(nears, min=near_plane)
        fars = torch.maximum(fars, nears + 1e-6)

        return nears, fars

    def set_nears_and_fars(self, ray_bundle):
        """Intersects the rays with the scene box and updates the near and far values.
        Populates nears and fars fields and returns the ray_bundle.

        Args:
            ray_bundle: specified ray bundle to operate on
        """
        aabb = self.scene_box
        nears, fars = self._intersect_with_aabb(ray_bundle.origins, ray_bundle.directions, aabb)
        ray_bundle.nears = nears[..., None]
        ray_bundle.fars = fars[..., None]
        return ray_bundle


def create_arg_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--data_type', '--dataset_type', type=str, required=True, help='Dataset type',
        choices=['llff', 'blender', 'LINEMOD', 'deepvoxels', 'tankstemple', 'toydesk', 'toydesk_custom', 'dtu',
        'tankstemple_custom', 'synthetic_custom', 'nerfstudio'])
    parser.add_argument('--data_path', '--datadir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_path', type=str, default='', help='Path to save processed dataset directory')

    # flags for llff
    parser.add_argument('--ndc', action='store_true', default=False,
        help='Turn on NDC device. Only for llff dataset')
    parser.add_argument('--spherify', action='store_true', default=False,
        help='Turn on spherical 360 scenes. Only for llff dataset')
    parser.add_argument('--factor', type=int, default=8,
        help='Downsample factor for LLFF images. Only for llff dataset')
    parser.add_argument('--llffhold', type=int, default=8,
        help='Hold out every 1/N images as test set. Only for llff dataset')

    # flags for blend
    parser.add_argument('--half_res', action='store_true', default=False,
        help='Load half-resolution (400x400) images instead of full resolution (800x800). Only for blender dataset.')
    parser.add_argument('--white_bkgd', action='store_true', default=False,
        help='Render synthetic data on white background. Only for blender/LINEMOD dataset')
    parser.add_argument('--test_skip', type=int, default=8,
        help='will load 1/N images from test/val sets. Only for large datasets like blender/LINEMOD/deepvoxels.')

    ## flags for deepvoxels
    parser.add_argument('--dv_scene', type=str, default='greek',
        help='Shape of deepvoxels scene. Only for deepvoxels dataset', choices=['armchair', 'cube', 'greek', 'vase'])

    parser.add_argument("--inverse_y", default=False,
                        help='inverse y when generating dataset and render')

    parser.add_argument("--w_pose", action="store_true", default=False, help='save poses')
    parser.add_argument("--mask", action="store_true", default=False, help='mask or not')


    return parser

def generate_dataset(args, output_path):

    if not os.path.exists(args.data_path):
        print('Dataset path not exists:', args.data_path)
        exit(-1)
    print(f"[spherify]: {args.spherify}")
    K = None # intrinsic matrix
    if args.data_type == 'llff':
        images, poses, bds, render_poses, i_test, masks = load_llff_data(args.data_path, factor=args.factor,
            recenter=True, bd_factor=.75, spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.data_path)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)

    elif args.data_type == 'blender':
        print("blender")
        images, poses, render_poses, hwf, i_split = load_blender_data(args.data_path, args.half_res, args.test_skip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.data_path)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.data_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.data_path, args.half_res, args.test_skip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]


    elif args.data_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.dv_scene, basedir=args.data_path, testskip=args.test_skip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.data_path)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    elif args.data_type == 'tankstemple':
        images, poses, render_poses, hwf, K, i_split = load_tankstemple_data(args.data_path)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.data_type == 'toydesk':
        images, poses, render_poses, masks, i_split, hwf = load_toydesk_data(args.data_path)
        i_train, i_val, i_test = i_split
        near = 0.
        far = 1
        if hwf is None:
            hwf = [353, 640, 466.772]

    elif args.data_type in ['toydesk_custom', 'tankstemple_custom', 'synthetic_custom']:
        images, poses, bds, render_poses, i_test, masks = load_toydesk_custom_data(args.data_path, factor=args.factor,
            recenter=True, bd_factor=.75, spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.data_path)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)

    elif args.data_type == 'nerfstudio':
        images, poses, render_poses, hwf, i_split = load_nerfstudio_data(args.data_path)
        print('Loaded nerfstudio', images.shape, render_poses.shape, hwf, args.data_path)
        i_train, i_val, i_test = i_split
        
        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

    else:
        print('Unknown dataset type:', args.data_type)
        exit(-1)



    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    if K is None:
        K = get_persp_intrinsic(H, W, focal)
    print('Intrinsic matrix:', K)
    print('Train/valid/test split', i_train, i_val, i_test)

    print('Calculating train/valid/test rays ...')
    rays_raw = torch.stack([get_persp_rays(H, W, K, torch.tensor(p)) for p in tqdm(poses[:,:3,:4])], 0) # [N, ro+rd, H, W, 3]
    
    rays = rays_raw.permute([0, 2, 3, 1, 4]).numpy().astype(np.float32) # [N, H, W, ro+rd, 3]


    # Calcular los valores de near y far usando AABBBoxCollider
    collider = AABBBoxCollider()
    ray_bundle = type('', (), {})()  # Crear un objeto vacío
    ray_bundle.origins = torch.tensor(rays[..., 0, :])  # Asignar los orígenes de los rayos
    ray_bundle.directions = torch.tensor(rays[..., 1, :])  # Asignar las direcciones de los rayos

    # Calcular los valores de near y far usando AABBBoxCollider
    ray_bundle = collider.set_nears_and_fars(ray_bundle)
    
    # Asignar los valores calculados de near y far a los rayos
    near = ray_bundle.nears.numpy()  # Convertir a numpy
    far = ray_bundle.fars.numpy()  # Convertir a numpy

    print('Done.', rays.shape)


    print('Splitting train/valid/test rays ...')
    if args.mask:
        rays_train, rgbs_train, masks_train = rays[i_train], images[i_train], masks[i_train]
        rays_val, rgbs_val, masks_val = rays[i_val], images[i_val], masks[i_val]
        rays_test, rgbs_test, masks_test = rays[i_test], images[i_test], masks[i_test]
    else:
        rays_train, rgbs_train = rays[i_train], images[i_train]
        rays_val, rgbs_val = rays[i_val], images[i_val]
        rays_test, rgbs_test = rays[i_test], images[i_test]

    print('Calculating exhibition rays ...')
    if render_poses is None:
        print(f'> Warning!  render_poses is None')
        render_poses = poses[i_train]
    rays_exhibit = torch.stack([get_persp_rays(H, W, K, torch.tensor(p)) for p in tqdm(render_poses[:,:3,:4])], 0) # [N, ro+rd, H, W, 3]
    rays_exhibit = rays_exhibit.permute([0, 2, 3, 1, 4]).numpy().astype(np.float32) # [N, H, W, ro+rd, 3]
    print('Done.', rays_exhibit.shape)

    print('Training set:', rays_train.shape, rgbs_train.shape)
    print('Validation set:', rays_val.shape, rgbs_val.shape)
    print('Testing set:', rays_test.shape, rgbs_test.shape)
    print('Exhibition set:', rays_exhibit.shape)

    print('Saving to: ', output_path)
    np.save(os.path.join(output_path, 'rays_train.npy'), rays_train)
    np.save(os.path.join(output_path, 'rgbs_train.npy'), rgbs_train)
    if args.mask:
        np.save(os.path.join(output_path, 'masks_train.npy'), masks_train)

    np.save(os.path.join(output_path, 'rays_val.npy'), rays_val)
    np.save(os.path.join(output_path, 'rgbs_val.npy'), rgbs_val)
    if args.mask:
        np.save(os.path.join(output_path, 'masks_val.npy'), masks_val)

    np.save(os.path.join(output_path, 'rays_test.npy'), rays_test)
    np.save(os.path.join(output_path, 'rgbs_test.npy'), rgbs_test)
    if args.mask:
        np.save(os.path.join(output_path, 'masks_test.npy'), masks_test)

    np.save(os.path.join(output_path, 'rays_exhibit.npy'), rays_exhibit)

    if args.w_pose:
        print("> Save poses")
        poses_train = poses[i_train]
        poses_val = poses[i_val]
        poses_test = poses[i_test]
        np.save(os.path.join(output_path, 'poses_train.npy'), poses_train)
        np.save(os.path.join(output_path, 'poses_val.npy'), poses_val)
        np.save(os.path.join(output_path, 'poses_test.npy'), poses_test)

    # Save meta data
    meta_dict = {
        'H': H, 'W': W, 'focal': float(focal),
        'near': near.tolist(), 'far': far.tolist(),

        'i_train': i_train.tolist() if isinstance(i_train, np.ndarray) else i_train,
        'i_val': i_val.tolist() if isinstance(i_val, np.ndarray) else i_val,
        'i_test': i_test.tolist() if isinstance(i_test, np.ndarray) else i_test,

        'ndc': args.ndc, 'factor': args.factor,
        'spherify': args.spherify, 'llffhold': args.llffhold,

        'half_res': args.half_res, 'white_bkgd': args.white_bkgd,
        'test_skip': args.test_skip, 'dv_scene': args.dv_scene
    }
    print("Meta data:", meta_dict)
    with open(os.path.join(output_path, 'meta.json'), 'w') as f:
        json.dump(meta_dict, f)


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far


if __name__ == '__main__':

    parser = create_arg_parser()
    args, _ = parser.parse_known_args()

    output_path = args.output_path
    if not args.output_path:
        output_path = args.data_path
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    generate_dataset(args, output_path)
