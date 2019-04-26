import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetworkDueling(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Init params and build model

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of action space
            seed (int): Random seed
            fc1_units (int): Nodes in hidden layer fc1
            fc2_units (int): Nodes in hidden layer fc2
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2_val = nn.Linear(fc2_units, fc2_units)
        self.fc3_val = nn.Linear(fc2_units, 1)
        self.fc2_adv = nn.Linear(fc2_units, fc2_units)
        self.fc3_adv = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Maps state to action values

        Args:
            state: State of the environment

        Returns:
            Action values corresponding to state.
        """
        y = F.elu(self.fc2(F.elu(self.fc1(state))))
        value = F.elu(self.fc3_val(F.elu(self.fc2_val(y))))
        adv = F.elu(self.fc3_adv(F.elu(self.fc2_adv(y))))
        q = value + adv - adv.mean()
        return q
