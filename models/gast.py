import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from einops import rearrange
from models.occ_blocks import BEVEncoder2D, BEVDecoder2D, ResnetBlock, get_2d_sincos_pos_embed, get_1d_sincos_temp_embed, TemporalBlock, SpatialBlock
from models.deformable_transformer import DeformableTransformer
from models.pose_encoder import PoseEncoder, quaternion_multiply, quaternion_to_rotation_matrix


class GAST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_gt = config['model_params']['use_gt']
        self.num_classes = config['model_params']['num_classes']
        self.expansion = config['model_params']['expansion']
        self.embed_dim = config['model_params']['embed_dim']
        self.num_his_frames = config['model_params']['num_his_frames']
        self.num_fut_frames = config['model_params']['num_fut_frames']
        self.num_patches = config['model_params']['num_patches']
        self.scene_range = torch.tensor(config['model_params']['scene_range'])
        self.bev_sizes = config['model_params']['bev_sizes']

        self.pose_encoder = PoseEncoder(
            in_channels=(2+3+4),
            embed_dims=self.embed_dim,
            history_frame_number=self.num_his_frames,
        )
        self.plan_attn = nn.MultiheadAttention(self.embed_dim, 8, batch_first=True)
        self.plan_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.plan_translation_output = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.Softplus(),
            nn.Linear(self.embed_dim // 2, (2+4)*3*self.num_fut_frames)
        )
        self.pose_embeds = nn.Sequential(
            nn.Linear(2+4, self.embed_dim),
            nn.ReLU(True),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=True)
        )

        self.temporal_block = TemporalBlock(
            input_dim=self.embed_dim, hidden_dim=self.embed_dim, return_all=True)
        self.spatial_block = SpatialBlock(
            input_dim=self.embed_dim, hidden_dim=self.embed_dim, num_layers=len(self.bev_sizes))

        self.class_embeds = nn.Embedding(self.num_classes, self.expansion)
        self.encoder = BEVEncoder2D(**config['model_params']['encoder'])
        self.decoder = BEVDecoder2D(**config['model_params']['decoder'])
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=False)
        self.temp_embed = nn.Parameter(
            torch.zeros(1, self.num_his_frames + self.num_fut_frames, self.embed_dim), requires_grad=False)
        self.update_block = ResnetBlock(in_channels=self.embed_dim*3, out_channels=self.embed_dim, temb_channels=0, dropout=0.1)
        self.cross_deform_attn = DeformableTransformer(**config['model_params']['cross_deform_attn'])
        self.self_deform_attn = DeformableTransformer(**config['model_params']['self_deform_attn'])
        self.initialize_weights()
        self.initialize_reference_points()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))
    
    def initialize_reference_points(self):
        H = W = int(self.num_patches ** 0.5)
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32))
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        self.cross_ref_2d = ref_2d.unsqueeze(-2).repeat(1, 1, self.config['model_params']['cross_deform_attn']['n_levels'], 1)
        self.self_ref_2d = ref_2d.unsqueeze(-2).repeat(1, 1, self.config['model_params']['self_deform_attn']['n_levels'], 1)

    def process_transformation_matrix(self, plan_output, cur_translation, cur_rotation):
        bs = cur_translation.shape[0]
        pred_delta_translation = plan_output[:, :2]
        pred_relative_rotation = plan_output[:, 2:]
        pred_relative_rotation = F.normalize(pred_relative_rotation, p=2, dim=-1)
        pred_next_rotation = quaternion_multiply(pred_relative_rotation, cur_rotation)
        pred_next_translation = pred_delta_translation + cur_translation[:, :2]
        pred_next_translation = torch.cat([pred_next_translation, cur_translation[:, 2:]], dim=-1)

        pred_rotation_3x3 = quaternion_to_rotation_matrix(pred_next_rotation)
        pred_next_ego_to_global = torch.zeros(bs, 4, 4, device=cur_translation.device, dtype=cur_translation.dtype)
        pred_next_ego_to_global[:, :3, :3] = pred_rotation_3x3
        pred_next_ego_to_global[:, :3, 3] = pred_next_translation
        pred_next_ego_to_global[:, 3, 3] = 1

        trans_info = dict(
            pred_delta_translation=pred_delta_translation,      # (b, 2)
            pred_next_translation=pred_next_translation,        # (b, 3)
            pred_relative_rotation=pred_relative_rotation,      # (b, 4)
            pred_next_rotation=pred_next_rotation,              # (b, 4)
            pred_next_ego_to_global=pred_next_ego_to_global,    # (b, 4, 4)
        )
        return trans_info

    def forward(self, data_dict):
        occ = data_dict['his_occ'] # (b, f, h, w, d)
        b, f, h, w, d = occ.shape

        plan_query = self.pose_encoder(
            data_dict['his_e2g_delta_translation'],
            data_dict['his_e2g_relative_rotation'],
            data_dict['his_ego_lcf_feat']
        )
        
        # encode
        occ_feats = self.class_embeds(occ) # (b, f, h, w, d, c)
        occ_feats = rearrange(occ_feats, 'b f h w d c -> (b f) (d c) h w') # (b*f, d*c, h, w)
        latent_occ = self.encoder(occ_feats) # (b*f, c1, h1, w1)

        # generate
        h1, w1 = latent_occ.shape[-2], latent_occ.shape[-1] 
        latent_occ_reshape = latent_occ.reshape(b, f, -1, h1, w1).permute(0, 1, 3, 4, 2).flatten(1, 3)
        plan_query, _ = self.plan_attn(plan_query, latent_occ_reshape, latent_occ_reshape)
        plan_query = self.plan_out(plan_query)
        plan_output = self.plan_translation_output(plan_query.squeeze(1)).reshape(b, self.num_fut_frames, -1, 6)    # (b, f, 3, 6)
       
        cur_translation = data_dict['his_e2g_translation'][:, -1]
        cur_rotation = data_dict['his_e2g_rotation'][:, -1]
        pred_delta_translations, pred_relative_rotations = [], []
        pred_ego_pose = []
        for idx in range(self.num_fut_frames):
            cur_plan_output = plan_output[:, idx]
            curr_ego_mode = data_dict['fut_ego_cmd'][:, idx]
            cur_plan_output = cur_plan_output[curr_ego_mode.bool()] # (b, 6)
            trans_info = self.process_transformation_matrix(cur_plan_output, cur_translation, cur_rotation)
            pred_delta_translations.append(trans_info['pred_delta_translation'])
            pred_relative_rotations.append(trans_info['pred_relative_rotation'])
            cur_translation = trans_info['pred_next_translation']
            cur_rotation = trans_info['pred_next_rotation']
            if self.use_gt:
                pred_ego_pose.append(data_dict['fut_ego_pose'][:, idx])
            else:
                pred_ego_pose.append(trans_info['pred_next_ego_to_global'])
        pred_delta_translations = torch.stack(pred_delta_translations, dim=1)
        pred_relative_rotations = torch.stack(pred_relative_rotations, dim=1)
        
        his_occ = latent_occ.reshape(b, f, -1, h1, w1)[:, -1]
        his_pose = data_dict['his_ego_pose'][:, -1]

        # wrapping
        his_occ_fut = his_occ.unsqueeze(1).repeat(1, self.num_fut_frames, 1, 1, 1)
        his_occ_fut = his_occ_fut.flatten(0, 1)
        his_pose_fut = his_pose.unsqueeze(1).repeat(1, self.num_fut_frames, 1, 1)
        his_pose_fut = his_pose_fut.flatten(0, 1)
        pred_ego_pose = torch.stack(pred_ego_pose, dim=1)
        pred_ego_pose = pred_ego_pose.flatten(0, 1)
        offset_fut, global_xyz = self.transform_coor_system(h1, w1, 1, his_pose_fut, pred_ego_pose)
        future_xyz = global_xyz[..., :2].reshape(b, self.num_fut_frames, -1, 2)
        offset_fut = offset_fut[..., :2].squeeze(1)
        next_occ = F.grid_sample(
            his_occ_fut, 
            offset_fut,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True)
        next_occ = next_occ.reshape(b, self.num_fut_frames, -1, h1, w1)
        
        # shift + scale
        if self.use_gt:
            pose_embeddings = self.pose_embeds(torch.cat([data_dict['fut_e2g_delta_translation'], data_dict['fut_e2g_relative_rotation']], dim=-1))
        else:
            pose_embeddings = self.pose_embeds(torch.cat([pred_delta_translations, pred_relative_rotations], dim=-1))
        shift, scale = self.adaLN_modulation(pose_embeddings).chunk(2, dim=-1)
        next_occ = next_occ * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        future_occ = next_occ.flatten(0, 1)

        # DCA
        offset = (offset_fut + 1.0) / 2.0
        offset = offset.flatten(1, 2).unsqueeze(2)
        future_occ_q = rearrange(future_occ, 'b c h w -> b (h w) c')
        future_occ_q = self.cross_deform_attn(
            query=future_occ_q, feats=[his_occ_fut], reference_points=offset)
        future_occ = future_occ_q.reshape(b, self.num_fut_frames, h1, w1, -1).permute(0, 1, 4, 2, 3)

        # merge
        latent_occ = latent_occ.reshape(b, f, -1, h1, w1)
        all_occ = torch.cat((latent_occ, future_occ), dim=1).flatten(0, 1)
        all_occ = all_occ + self.pos_embed.permute(0, 2, 1).reshape(-1, self.embed_dim, h1, w1)
        all_occ = all_occ.reshape(b, -1, all_occ.shape[1], h1, w1)
        all_occ = all_occ + self.temp_embed.unsqueeze(-1).unsqueeze(-1) # (b,f1+f2,c,h,w)

        # temporal
        tem_occ = self.temporal_block(all_occ)
        tem_feats = tem_occ[:, self.num_his_frames:]

        # spatial
        his_pose_repeat = his_pose.unsqueeze(1).repeat(1, self.num_his_frames, 1, 1).flatten(0, 1)
        his_pose_all = data_dict['his_ego_pose'].flatten(0, 1)
        _, global_xyz = self.transform_coor_system(h1, w1, 1, his_pose_repeat, his_pose_all)
        history_xyz = global_xyz[..., :2].reshape(b, self.num_his_frames, -1, 2)
        all_xyz = torch.cat((history_xyz, future_xyz), dim=1)   # (b,f1+f2,n,2)
        all_xyz = all_xyz.flatten(1, 2) # (b,(f1+f2)*n,2)
        all_occ = all_occ.flatten(3, 4) # (b,f1+f2,c,h*w)
        all_occ = all_occ.permute(0, 1, 3, 2).flatten(1, 2) # (b,(f1+f2)*h*w,c)

        bev_feats = []
        for size in self.bev_sizes:
            feat, grid_coords = self.points_to_bev(all_xyz, all_occ, bev_size=size)    # (b,(f1+f2)*n,1,2)
            bev_feats.append(feat)
        bev_feat = self.spatial_block(bev_feats)
        sam_feat = F.grid_sample(     # (b,c,(f1+f2)*n,1)
            bev_feat, 
            grid_coords, 
            mode='bilinear',
            padding_mode='zeros', 
            align_corners=True)
        sam_feat = sam_feat.squeeze(-1) # (b,c,(f1+f2)*n)
        sam_feat = sam_feat.reshape(b, self.embed_dim, self.num_his_frames+self.num_fut_frames, h1, w1)
        sam_feat = sam_feat.permute(0, 2, 1, 3, 4)
        fut_feat = sam_feat[:, self.num_his_frames:]    # (b,f2,c,h,w)

        updated_occ = torch.cat((future_occ, fut_feat, tem_feats), dim=2)  # (b,f2,2c,h,w)
        updated_occ = updated_occ.flatten(0, 1) # (b*f2,2c,h,w)
        updated_occ = self.update_block(updated_occ)    # (b*f2,c,h,w)
        updated_occ = updated_occ.reshape(b, -1, self.embed_dim, h1, w1)    # (b,f2,c,h,w)

        # DSA
        _, ff, _, hf, wf = updated_occ.shape
        updated_occ = rearrange(updated_occ, 'b f c h w -> (b f) c h w')
        updated_occ_q = rearrange(updated_occ, 'bf c h w -> bf (h w) c')
        self_ref_points = self.self_ref_2d.repeat(b*ff, 1, 1, 1).to(updated_occ.device)
        updated_occ_q = self.self_deform_attn(
            query=updated_occ_q, feats=[updated_occ], reference_points=self_ref_points)
        updated_occ = updated_occ_q.reshape(b, ff, hf, wf, -1).permute(0, 1, 4, 2, 3)

        pred_occ = torch.cat((latent_occ, updated_occ), dim=1).flatten(0, 1)

        # decode
        logits = self.decoder(pred_occ) # (b*f, d*c, h, w)
        logits = logits.permute(0,2,3,1).reshape(-1, d, self.expansion) # (b*f*h*w, d, c)
        cls_embed = self.class_embeds.weight.T.unsqueeze(0) # (1, expansion, num_classes)
        logits = torch.matmul(logits, cls_embed) # (b*f*h*w, d, num_classes)
        logits = logits.reshape(b, -1, h, w, d, self.num_classes).permute(0,1,5,2,3,4) # (b, f, num_classes, h, w, d)
        return logits, pred_delta_translations, pred_relative_rotations


    def transform_coor_system(self, H, W, Z, src_matrix, dst_matrix):
        device = src_matrix.device
        origin = ((self.scene_range[:3] + self.scene_range[3:]) / 2).to(device)
        scene_size = (self.scene_range[3:] - self.scene_range[:3]).to(device)
        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * scene_size[0] + self.scene_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * scene_size[1] + self.scene_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * scene_size[2] + self.scene_range[2]
        xx = x[None, None, :].expand(Z, H, W)
        yy = y[None, :, None].expand(Z, H, W)
        zz = z[:, None, None].expand(Z, H, W)
        coors = torch.stack([xx, yy, zz], dim=-1)
        coors = coors.reshape(-1, 3)

        B = src_matrix.shape[0]
        coors_exp = coors.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
        R = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32, device=device)  # (3, 3)

        # Step 1: ego (dst) -> global
        processed = coors_exp @ R       # (B, N, 3)
        rot_dst = dst_matrix[:, :3, :3] # (B, 3, 3)
        pose_dst = dst_matrix[:, :3, 3] # (B, 3)
        global_xyz = torch.bmm(processed, rot_dst.transpose(-1, -2)) + pose_dst.unsqueeze(1)  # (B, N, 3)

        # Step 2: global -> ego (src)
        pose_src = src_matrix[:, :3, 3]   # (B, 3)
        ego_centered = global_xyz - pose_src.unsqueeze(1)  # (B, N, 3)
        rot_src = src_matrix[:, :3, :3]   # (B, 3, 3)
        ego_rotated = torch.bmm(ego_centered, rot_src)     # (B, N, 3)
        ego_xyz = ego_rotated @ R

        offset = (ego_xyz - origin) / scene_size * 2.0  # (B, N, 3)
        offset = torch.clamp(offset, -1.0, 1.0)
        offsets = offset.reshape(B, Z, H, W, 3)
        return offsets, global_xyz
    

    def points_to_bev(self,
        points: torch.Tensor,      # (B, N, 2)
        features: torch.Tensor,    # (B, N, C)
        bev_size: tuple = (200, 200)  # (H, W)
    ):
        B, N, C = features.shape
        H, W = bev_size

        bev_list = []
        grid_list = []

        for b in range(B):
            pts = points[b]       # (N, 2)
            feats = features[b]   # (N, C)

            x_min = pts[:, 0].min().item()
            x_max = pts[:, 0].max().item()
            y_min = pts[:, 1].min().item()
            y_max = pts[:, 1].max().item()
            u_cont = (pts[:, 0] - x_min) * W / (x_max - x_min)  # (N,)
            v_cont = (pts[:, 1] - y_min) * H / (y_max - y_min)  # (N,)

            xs_idx = torch.clamp(u_cont.long(), 0, W - 1)
            ys_idx = torch.clamp(v_cont.long(), 0, H - 1)
            idx = ys_idx * W + xs_idx

            u_norm = (u_cont / (W - 1)) * 2 - 1
            v_norm = (v_cont / (H - 1)) * 2 - 1
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

            feat_flat = feats.transpose(0, 1).reshape(C, N)  # (C, N)
            idx_flat = idx.unsqueeze(0).repeat(C, 1)         # (C, N)
            bev_flat = torch_scatter.scatter_mean(feat_flat, idx_flat, dim_size=H * W)  # (C, H*W)
            bev = bev_flat.view(1, C, H, W)

            bev_list.append(bev)
            grid_list.append(grid)

        bev = torch.cat(bev_list, dim=0)   # (B, C, H, W)
        grid = torch.cat(grid_list, dim=0) # (B, N, 1, 2)
        return bev, grid