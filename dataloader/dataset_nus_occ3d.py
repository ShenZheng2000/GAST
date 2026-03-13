import os, json, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix


class NuscenesOcc3D(data.Dataset):
    def __init__(self, config, dataset_type):
        self.config = config
        self.return_len = self.config['dataset_params']['return_len']
        self.start_frame = self.config['dataset_params']['start_frame']
        self.mid_frame = self.config['dataset_params']['mid_frame']
        self.end_frame = self.config['dataset_params']['end_frame']

        self.label_path = self.config['dataset_params']['label_path']
        self.anno_path = self.config['dataset_params']['anno_path']

        self.class_names = [ 'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',  
                             'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',  
                             'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free']

        with open(self.anno_path, 'r') as anno_file:
            self.annotations = json.load(anno_file)
        if dataset_type == "train":
            self.scene_list = self.annotations['train_split']
            self.pkl_file = self.config['dataset_params']['train_data_loader']['pkl_file']
        elif dataset_type == "valid":
            self.scene_list = self.annotations['val_split']
            self.pkl_file = self.config['dataset_params']['val_data_loader']['pkl_file']

        self.sample_infos = {}
        for scene_index in self.scene_list:
            scene_length = len(self.annotations['scene_infos'][scene_index])
            for sample_idx, sample_token in enumerate(self.annotations['scene_infos'][scene_index].keys()):
                sample_info = {
                    'scene_length': scene_length,
                    'sample_index': sample_idx
                }
                self.sample_infos[sample_token] = sample_info
        
        with open(self.pkl_file, 'rb') as f:
            data = pickle.load(f)
        self.data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))

        self.valid_list = []
        for idx in range(len(self.data_infos)):
            sample_token = self.data_infos[idx]['token']
            scene_length = self.sample_infos[sample_token]['scene_length']
            sample_index = self.sample_infos[sample_token]['sample_index']
            if scene_length - sample_index >= self.return_len:
                self.valid_list.append(idx)
            
            
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.valid_list)


    def __getitem__(self, index):
        
        start_index = self.valid_list[index]

        occ_list, ego_pose = [], []
        ego_to_global_translation, ego_to_global_rotation = [], []
        gt_ego_fut_cmd, gt_ego_lcf_feat = [], []
        for idx in range(self.return_len):
            info = self.data_infos[start_index + idx]
            gt = np.load(os.path.join(info['occ_path'].replace("./data/nuscenes/", self.label_path), "labels.npz"))
            occ_list.append(gt['semantics'])
            ego_to_global = transform_matrix(info['ego2global_translation'], Quaternion(info['ego2global_rotation']))
            ego_pose.append(ego_to_global)
            ego_to_global_translation.append(info['ego2global_translation'])
            ego_to_global_rotation.append(info['ego2global_rotation'])
            gt_ego_fut_cmd.append(info['gt_ego_fut_cmd'])
            ego_feat = info['gt_ego_lcf_feat']
            ego_vx, ego_vy, ego_w = ego_feat[0], ego_feat[1], ego_feat[4]
            gt_ego_lcf_feat.append([ego_vx, ego_vy, ego_w])
        
        occ_all = torch.stack([torch.from_numpy(occ) for occ in occ_list], 0).to(torch.int64)
        ego_pose = torch.stack([torch.from_numpy(pose) for pose in ego_pose], 0).to(torch.float32)
        ego_to_global_translation = torch.tensor(np.array(ego_to_global_translation)).to(torch.float32)
        ego_to_global_rotation = torch.tensor(np.array(ego_to_global_rotation)).to(torch.float32)
        gt_ego_fut_cmd = torch.tensor(np.array(gt_ego_fut_cmd)).to(torch.float32)
        gt_ego_lcf_feat = torch.tensor(np.array(gt_ego_lcf_feat)).to(torch.float32)

        ego_to_global_delta_translation = ego_to_global_translation[1:] - ego_to_global_translation[:-1]  # (f-1, 3)
        ego_to_global_delta_translation = ego_to_global_delta_translation[..., :2]  # (f-1, 2)
        his_e2g_delta_translation = ego_to_global_delta_translation[self.start_frame : self.mid_frame - 1]
        his_e2g_delta_translation = F.pad(his_e2g_delta_translation, pad=(0, 0, 0, 1), mode='constant', value=0)
        fut_e2g_delta_translation = ego_to_global_delta_translation[self.mid_frame - 1 : self.end_frame - 1]
        ego_to_global_relative_rotation = compute_relative_rotation(ego_to_global_rotation) # (f-1, 4)
        his_e2g_relative_rotation = ego_to_global_relative_rotation[self.start_frame : self.mid_frame - 1]
        his_e2g_relative_rotation = F.pad(his_e2g_relative_rotation, pad=(0, 0, 0, 1), mode='constant', value=0)
        fut_e2g_relative_rotation = ego_to_global_relative_rotation[self.mid_frame - 1 : self.end_frame - 1]

        data_dict = {}
        data_dict['his_occ'] = occ_all[self.start_frame : self.mid_frame]
        data_dict['fut_occ'] = occ_all[self.mid_frame : self.end_frame]
        data_dict['his_ego_pose'] = ego_pose[self.start_frame : self.mid_frame]
        data_dict['fut_ego_pose'] = ego_pose[self.mid_frame : self.end_frame]
        data_dict['his_e2g_translation'] = ego_to_global_translation[self.start_frame : self.mid_frame]
        data_dict['fut_e2g_translation'] = ego_to_global_translation[self.mid_frame : self.end_frame]
        data_dict['his_e2g_rotation'] = ego_to_global_rotation[self.start_frame : self.mid_frame]
        data_dict['fut_e2g_rotation'] = ego_to_global_rotation[self.mid_frame : self.end_frame]
        data_dict['his_ego_cmd'] = gt_ego_fut_cmd[self.start_frame : self.mid_frame]
        data_dict['fut_ego_cmd'] = gt_ego_fut_cmd[self.mid_frame : self.end_frame]
        data_dict['his_ego_lcf_feat'] = gt_ego_lcf_feat[self.start_frame : self.mid_frame]
        data_dict['fut_ego_lcf_feat'] = gt_ego_lcf_feat[self.mid_frame : self.end_frame]
        data_dict['his_e2g_delta_translation'] = his_e2g_delta_translation.to(torch.float32)
        data_dict['fut_e2g_delta_translation'] = fut_e2g_delta_translation.to(torch.float32)
        data_dict['his_e2g_relative_rotation'] = his_e2g_relative_rotation.to(torch.float32)
        data_dict['fut_e2g_relative_rotation'] = fut_e2g_relative_rotation.to(torch.float32)
        return data_dict
    
def compute_relative_rotation(ego_to_global_rotation):
    current = ego_to_global_rotation[1:, :]
    previous = ego_to_global_rotation[:-1, :]
    previous_conjugate = previous.clone()
    previous_conjugate[:, 1:] *= -1

    w1, x1, y1, z1 = previous_conjugate[..., 0], previous_conjugate[..., 1], previous_conjugate[..., 2], previous_conjugate[..., 3]
    w2, x2, y2, z2 = current[..., 0], current[..., 1], current[..., 2], current[..., 3]
    relative_rotation = torch.zeros_like(current)
    relative_rotation[..., 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    relative_rotation[..., 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    relative_rotation[..., 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    relative_rotation[..., 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return relative_rotation


def collate_fn_default(data):

    his_occ_batch = [d['his_occ'] for d in data]
    fut_occ_batch = [d['fut_occ'] for d in data]
    his_ego_pose_batch = [d['his_ego_pose'] for d in data]
    fut_ego_pose_batch = [d['fut_ego_pose'] for d in data]
    his_e2g_translation_batch = [d['his_e2g_translation'] for d in data]
    fut_e2g_translation_batch = [d['fut_e2g_translation'] for d in data]
    his_e2g_rotation_batch = [d['his_e2g_rotation'] for d in data]
    fut_e2g_rotation_batch = [d['fut_e2g_rotation'] for d in data]
    his_ego_cmd_batch = [d['his_ego_cmd'] for d in data]
    fut_ego_cmd_batch = [d['fut_ego_cmd'] for d in data]
    his_ego_lcf_feat_batch = [d['his_ego_lcf_feat'] for d in data]
    fut_ego_lcf_feat_batch = [d['fut_ego_lcf_feat'] for d in data]
    his_e2g_delta_translation_batch = [d['his_e2g_delta_translation'] for d in data]
    fut_e2g_delta_translation_batch = [d['fut_e2g_delta_translation'] for d in data]
    his_e2g_relative_rotation_batch = [d['his_e2g_relative_rotation'] for d in data]
    fut_e2g_relative_rotation_batch = [d['fut_e2g_relative_rotation'] for d in data]

    return {
        'his_occ': torch.stack(his_occ_batch, 0),
        'fut_occ': torch.stack(fut_occ_batch, 0),
        'his_ego_pose': torch.stack(his_ego_pose_batch, 0),
        'fut_ego_pose': torch.stack(fut_ego_pose_batch, 0),
        'his_e2g_translation': torch.stack(his_e2g_translation_batch, 0),
        'fut_e2g_translation': torch.stack(fut_e2g_translation_batch, 0),
        'his_e2g_rotation': torch.stack(his_e2g_rotation_batch, 0),
        'fut_e2g_rotation': torch.stack(fut_e2g_rotation_batch, 0),
        'his_ego_cmd': torch.stack(his_ego_cmd_batch, 0),
        'fut_ego_cmd': torch.stack(fut_ego_cmd_batch, 0),
        'his_ego_lcf_feat': torch.stack(his_ego_lcf_feat_batch, 0),
        'fut_ego_lcf_feat': torch.stack(fut_ego_lcf_feat_batch, 0),
        'his_e2g_delta_translation': torch.stack(his_e2g_delta_translation_batch, 0),
        'fut_e2g_delta_translation': torch.stack(fut_e2g_delta_translation_batch, 0),
        'his_e2g_relative_rotation': torch.stack(his_e2g_relative_rotation_batch, 0),
        'fut_e2g_relative_rotation': torch.stack(fut_e2g_relative_rotation_batch, 0),
    }