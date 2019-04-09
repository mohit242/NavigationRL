import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetworkSimple(nn.Module):

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
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Maps state to action values

        Args:
            state: State of the environment

        Returns:
            Action values corresponding to state.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
