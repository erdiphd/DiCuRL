import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import time
from typing import Union
import warnings
import matplotlib.pyplot as plt


class utils:

    @staticmethod
    def weight_init(m):
        """Custom weight init for Conv2D and Linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            gain = nn.init.calculate_gain('relu')
            nn.init.orthogonal_(m.weight.data, gain)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

def wasserstein_reward(d: torch.Tensor) -> torch.Tensor:
    """
    return the wasserstein reward
    """
    return d


reward_mapping = {'aim': wasserstein_reward,
                  }


class MlpNetwork(nn.Module):
    """
    Basic feedforward network uesd as building block of more complex policies
    """

    def __init__(self, input_dim, output_dim=1, activ=F.relu, output_nonlinearity=None, n_units=64, tanh_constant=1.0):
        super(MlpNetwork, self).__init__()

        self.h1 = nn.Linear(input_dim, n_units)
        self.h2 = nn.Linear(n_units, n_units)
        # self.h3 = nn.Linear(n_units, n_units)
        self.out = nn.Linear(n_units, output_dim)
        self.out_nl = output_nonlinearity
        self.activ = activ
        self.tanh_constant = tanh_constant
        self.apply(utils.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of network
        :param x:
        :return:
        """
        x = self.activ(self.h1(x))
        x = self.activ(self.h2(x))
        # x = self.activ(self.h3(x))
        x = self.out(x)
        if self.out_nl is not None:
            if self.out_nl == F.log_softmax:
                x = F.log_softmax(x, dim=-1)
            else:
                if self.out_nl == torch.tanh:
                    x = self.out_nl(self.tanh_constant * x)
                else:
                    x = self.out_nl(x)
        return x


class DiscriminatorEnsemble(nn.Module):
    def __init__(self, n_ensemble, x_dim=1, reward_type='aim', lr=1e-4, lipschitz_constant=0.1, output_activation=None,
                 device='cuda:0',
                 env_name=None, tanh_constant=1.0, lambda_coef=10.0, adam_eps=1e-8, optim='adam'):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.adam_eps = adam_eps
        self.optim = optim
        self.discriminator_ensemble = nn.ModuleList(
            [Discriminator(x_dim, reward_type, lr, lipschitz_constant, output_activation, device,
                           env_name, tanh_constant, lambda_coef, adam_eps, optim) for i in range(n_ensemble)])

        self.apply(utils.weight_init)

    def forward(self, inputs):
        h = inputs
        outputs = torch.stack([discriminator(h) for discriminator in self.discriminator_ensemble],
                              dim=1)  # [bs, n_ensemble, dim(1)]
        outputs = torch.mean(outputs, dim=1)  # [bs, 1]
        return outputs

    def std(self, inputs):
        aim_outputs = torch.stack(self.forward(inputs), dim=1)  # [bs, n_ensemble, 1]
        return torch.std(aim_outputs, dim=1, keepdim=False)  # [bs, 1]

    def reward(self, x: torch.Tensor) -> np.ndarray:
        return np.stack([discriminator.reward(x) for discriminator in self.discriminator_ensemble], axis=1).mean(axis=1)

    def optimize_discriminator(self, *args, **kwargs):
        loss_list = []
        wgan_loss_list = []
        graph_penalty_list = []
        # min_aim_f_loss_list = []

        for discriminator in self.discriminator_ensemble:
            loss, wgan_loss, graph_penalty, min_aim_f_loss = discriminator.optimize_discriminator(*args, **kwargs)
            loss_list.append(loss)
            wgan_loss_list.append(wgan_loss)
            graph_penalty_list.append(graph_penalty)
            # min_aim_f_loss_list.append(min_aim_f_loss)
        return torch.stack(loss_list, dim=0).mean(0), torch.stack(wgan_loss_list, dim=0).mean(0), torch.stack(
            graph_penalty_list, dim=0).mean(0), None


class Discriminator(nn.Module):
    def __init__(self, x_dim=1, reward_type='aim', lr=1e-4, lipschitz_constant=0.1, output_activation=None,
                 device='cuda:0',
                 env_name=None, tanh_constant=1.0, lambda_coef=10.0, adam_eps=1e-8, optim='adam'):
        self.use_cuda = torch.cuda.is_available()
        self.device = device  # torch.device("cuda" if self.use_cuda else "cpu")

        self.adam_eps = adam_eps
        self.optim = optim
        super(Discriminator, self).__init__()
        self.input_dim = x_dim
        assert reward_type in ['aim', 'gail', 'airl', 'fairl']
        self.reward_type = reward_mapping[reward_type]
        if self.reward_type == 'aim':
            self.d = MlpNetwork(self.input_dim, n_units=64)  # , activ=f.tanh)
        else:
            if output_activation is None:
                self.d = MlpNetwork(self.input_dim, n_units=64, activ=torch.tanh)
            elif output_activation == 'tanh':
                self.d = MlpNetwork(self.input_dim, n_units=64, activ=torch.relu, output_nonlinearity=torch.tanh,
                                    tanh_constant=tanh_constant)

        self.d.to(self.device)
        self.lr = lr
        if optim == 'adam':
            self.discriminator_optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=adam_eps)

        self.lipschitz_constant = lipschitz_constant
        self.env_name = env_name
        self.lambda_coef = lambda_coef
        self.apply(utils.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        return discriminator output
        :param x:
        :return:
        """

        output = self.d(x)
        return output

    def reward(self, x: torch.Tensor) -> np.ndarray:
        """
        return the reward
        """

        r = self.forward(x)
        if self.reward_type is not None:
            r = self.reward_type(r)
        return r.cpu().detach().numpy()

    def compute_graph_pen(self,
                          prev_state: torch.Tensor,
                          next_state_state: torch.Tensor):
        """
        Computes values of the discriminator at different points
        and constraints the difference to be 0.1
        """
        if self.use_cuda:
            prev_state = prev_state.cuda()
            next_state_state = next_state_state.cuda()
            zero = torch.zeros(size=[int(next_state_state.size(0))]).cuda()
        else:
            zero = torch.zeros(size=[int(next_state_state.size(0))])
        prev_out = self(prev_state)
        next_out = self(next_state_state)
        penalty = self.lambda_coef * torch.max(torch.abs(next_out - prev_out) - self.lipschitz_constant, zero).pow(
            2).mean()
        return penalty

    def optimize_discriminator(self, target_states, policy_states, policy_next_states,
                               replay_buffer=None, goal_env=None, ):
        """
        Optimize the discriminator based on the memory and
        target_distribution
        :return:
        """
        self.discriminator_optimizer.zero_grad()

        ones = target_states  # [bs, dim([ag,dg])] #[g,g]
        zeros = policy_next_states  # [bs, dim([ag,dg])] #[s',g]
        zeros_prev = policy_states  # [bs, dim([ag,dg])] #[s,g]

        pred_ones = self(ones)
        pred_zeros = self(zeros)
        graph_penalty = self.compute_graph_pen(zeros_prev, zeros)
        min_aim_f_loss = None
        wgan_loss = torch.mean(pred_zeros) + torch.mean(pred_ones * (-1.))
        loss = wgan_loss + graph_penalty

        loss.backward()
        self.discriminator_optimizer.step()
        return loss, wgan_loss, graph_penalty, min_aim_f_loss


if __name__ == "__main__":
    n_ensemble = 5
    x_dim = 6
    env_name = 'sawyer_reach'
    tmp = DiscriminatorEnsemble(n_ensemble=n_ensemble,x_dim=x_dim,env_name=env_name)
    obs = torch.load("/home/Desktop/Storage/Projects/silinecek/test.pt")
    tmp.optimize_discriminator(obs,obs,obs)
    tmp.reward(obs)

    print("test")