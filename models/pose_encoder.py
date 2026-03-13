import torch
import torch.nn as nn
import torch.nn.init as init

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=1)

def quaternion_to_rotation_matrix(quaternion):
    # Quterion to Rotation Matrix
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    batch_size = quaternion.size(0)
    R = torch.zeros((batch_size, 3, 3), device=quaternion.device, dtype=quaternion.dtype)

    R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)

    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - w * x)

    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

    return R


class PoseEncoder(nn.Module):
    def __init__(
            self,
            in_channels=(2+3+4),
            embed_dims=128,
            history_frame_number=4,
    ):
        super(PoseEncoder, self).__init__()

        self.plan_input = nn.Sequential(
            nn.Linear(in_channels, embed_dims//2),
            nn.ReLU(True),
            nn.Linear(embed_dims//2, embed_dims//2),
        )
        self.pose_time_embedding = nn.Parameter(
            torch.zeros(1, history_frame_number, embed_dims//2)
        )
        self.plan_encoder = nn.Linear(history_frame_number*embed_dims//2, embed_dims)
        init.trunc_normal_(self.pose_time_embedding, std=0.02)

    def forward(self, history_delta_translation, history_relative_rotation, history_ego_lcf_feat):
        bs = history_delta_translation.shape[0]
        plan_query = torch.cat([history_delta_translation, history_relative_rotation, history_ego_lcf_feat], dim=-1)    # (b, num_his_frame, 9)
        plan_query = self.plan_input(plan_query)            # (b, num_his_frame, 64)
        plan_query = plan_query + self.pose_time_embedding  # (b, num_his_frame, 64)
        plan_query = plan_query.reshape(bs, -1)             # (b, num_his_frame*64)
        plan_query = self.plan_encoder(plan_query)          # (b, 128)
        plan_query = plan_query.unsqueeze(1)                # (b, 1, 128)
        return plan_query                                   # (b, 1, 128)
