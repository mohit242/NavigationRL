import torch
import random
import numpy as np
from .agent import RLAgent
from navigationrl.models import QNetFactory
from navigationrl.utils import ReplayBuffer

class DQNAgent(RLAgent):

    def __init__(self, state_size, action_size, seed, device, qnet_params, buffer_size=int(1e5), batch_size=64,
                 gamma=0.99, tau=1e-3, lr=5e-4, update_every=4, logger=None, double_dqn=False):
        """A RL agent that utilizes Deep Q-Networks for Q-value approximation.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of actions
            seed (int): random seed
            device: cpu or gpu
            qnet_params: params of Q-network
            buffer_size (int): size of memory buffer (no. of records)
            batch_size (int): size of training batch
            gamma (float): discount factor
            tau (float): soft update factor
            lr (float): learning rate
            update_every (int): number of steps before every update
        """
        self.logger = logger
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.device = device
        self.double_dqn = double_dqn
        # Q-Network
        qnet_factory = QNetFactory(state_size, action_size, seed)
        self.qnet_local = qnet_factory(qnet_params)
        self.qnet_target = qnet_factory(qnet_params)
        self.optimizer = torch.optim.Adam(self.qnet_local.parameters(), self.lr)

        torch.manual_seed(seed)
        # Replay Memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # init time step
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards , next_states, dones = experiences

        if not self.double_dqn:
            Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        else:

            _, next_state_actions = self.qnet_local(next_states).max(1, keepdim=True).detach()
            Q_targets_next = self.qnet_target(next_states).gather(1, next_state_actions).detach()

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnet_local(states).gather(1, actions)

        loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.logger is not None:
            self.logger.log_metric("loss", loss)
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)

