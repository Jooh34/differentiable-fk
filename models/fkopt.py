import torch
import torch.nn as nn
from graphics.rotation import quaternion_to_matrix

class fkopt_net(nn.Module):
    def __init__(self, num_joint):
        super().__init__()
        self.num_joint = num_joint
        self.out_feature = num_joint * 4

        self.simple_encoder = simple_encoder(self.out_feature)
        

    def forward(self, input):
        device = input.device

        # _input = (B, 3)
        B = input.shape[0]
        
        # (B, out_feature)
        joints = self.simple_encoder(input)

        # (B, num_joints, 4 * 4)
        joint_coordinates = torch.zeros((B, self.num_joint, 4, 4)).to(device)

        # (B, 4, 4)
        T = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
        ], dtype=torch.float, device=device)
        T = T.unsqueeze(dim=0).repeat((B,1,1))

        for j in range(self.num_joint):
            # (B, 4, 4)
            R = quaternion_to_matrix(joints[:, j*4:j*4+4])

            # (B, 4, 4)
            coordi = torch.bmm(R,T)

            if j == 0:
                joint_coordinates[:, j] = coordi
            else:
                joint_coordinates[:, j] = torch.bmm(coordi, joint_coordinates[:, j-1].clone())

        # (B, num_joint, 3)
        fake_points = torch.zeros((B, self.num_joint, 3), device=device)

        O = torch.tensor([0,0,0,1], dtype=torch.float, device=device).unsqueeze(dim=-1).repeat((B,1,1))
        for j in range(self.num_joint):
            # (B, num_joint, 4, 1)
            points = torch.bmm(joint_coordinates[:, j], O)

            # (B, num_joint, 3)
            fake_points[:, j] = (points[:, :3].squeeze(dim=-1) / (points[:, 3] + 1e-9))

        # fake_points = (B, num_joint, 3)
        return fake_points

        

class simple_encoder(nn.Module):
    def __init__(self, out_feature):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layer = nn.Sequential(
            nn.Linear(3, 1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(1024, out_feature),
            nn.Tanh(),
        ).to(device)
    
    def forward(self, x):
        return self.layer(x)