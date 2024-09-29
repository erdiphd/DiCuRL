from algorithm.diffusion.utils.diffusion import Diffusion
from algorithm.diffusion.utils.model import MLP
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from collections import OrderedDict, deque
from tqdm import tqdm
import os
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import plotly.express as px
from algorithm.diffusion.visualizer import Visualizer

class utils:
    @staticmethod
    def make_dir(*path_parts):
        dir_path = os.path.join(*path_parts)
        try:
            os.mkdir(dir_path)
        except OSError:
            pass
        return dir_path


class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)


class DiffusionGoalSampler:
    def __init__(self, state_dim, action_dim, max_action, min_action, device, agent, diffusion_configuration):
        self.diffusion_training_iteration = diffusion_configuration.diffusion_iteration
        self.diffusion_n_timesteps = diffusion_configuration.diffusion_n_timesteps
        self.diffusion_lr = diffusion_configuration.diffusion_lr
        self.diffusion_loss_coef = diffusion_configuration.diffusion_loss_coef
        self.critic_loss_coef = diffusion_configuration.critic_loss_coef
        self.aim_reward_loss_coef = diffusion_configuration.aim_reward_loss_coef
        self.debug = diffusion_configuration.debug
        # self.debug_saved_path = utils.make_dir(diffusion_configuration.save_path_prefix ,"diffusion_debug")
        self.model_state_dim = state_dim
        self.model_action_dim = action_dim
        self.device = device
        self.agent = agent
        self.model = MLP(state_dim=self.model_state_dim, action_dim=self.model_action_dim, device=self.device)
        # max_action = torch.Tensor([2, 2, 2]).to(self.device)
        # min_action = torch.Tensor([-2, -2, -2]).to(self.device)
        self.max_action = max_action
        self.min_action = min_action
        max_action = torch.Tensor(max_action).to(device)
        min_action = torch.Tensor(min_action).to(device)
        #TODO
        self.loss_container = deque(maxlen=self.diffusion_training_iteration)
        self.diffusion_goals_container = deque(maxlen=self.diffusion_training_iteration)
        self.critic_loss_container = deque(maxlen=self.diffusion_training_iteration)

        #TODO change loss_type and n_timesteps hard-coded parts
        self.diffusion = Diffusion(state_dim=self.model_state_dim, action_dim=self.model_action_dim, model=self.model,
                                   loss_type='l2', min_action=min_action, max_action=max_action, beta_schedule='vp',
                                   n_timesteps=self.diffusion_n_timesteps).to(self.device)

        #TODO change lr hard-coded part
        lr = self.diffusion_lr
        self.diffusion_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=lr)
        self.diffusion_lr_scheduler = CosineAnnealingLR(self.diffusion_optimizer, T_max=1000, eta_min=0.)

        self.counter = 0

        if diffusion_configuration.env in ['FetchPush-v1', 'FetchSlide-v1']:
            visualizer_config = {'number_of_points': 50,
                                 'x_min': self.min_action[0],
                                 'y_min': self.min_action[1],
                                 'x_max': self.max_action[0],
                                 'y_max': self.max_action[1],
                                 'env_name': diffusion_configuration.env,
                                 'device': self.device,
                                 'sub_folder': diffusion_configuration.log_subfolder_name
                                 }
        elif diffusion_configuration.env in ['FetchPickAndPlace-v1', 'FetchReach-v1']:
            visualizer_config = {'number_of_points': 50,
                                 'x_min': self.min_action[0],
                                 'y_min': self.min_action[1],
                                 'z_min': self.min_action[2],
                                 'x_max': self.max_action[0],
                                 'y_max': self.max_action[1],
                                 'z_max': self.max_action[2],
                                 'env_name': diffusion_configuration.env,
                                 'device': self.device,
                                 'sub_folder': diffusion_configuration.log_subfolder_name
                                 }
        else:
            raise   NotImplementedError

        self.visualizer = Visualizer(**visualizer_config)


    def sampler(self, input):
        self.diffusion_goals = self.diffusion(input)
        return self.diffusion_goals

    def loss(self, output, input):
        diffusion_loss = self.diffusion.loss(output, input)
        return diffusion_loss

    def train(self, agent, aim_discriminator, hgg_pool, desired_goals, episode_number):
        if not len(hgg_pool.pool):
            return
        achieved_states, achieved_pool_init_state = hgg_pool.pad()
        if agent.env_name == 'FetchReach-v1':
            achieved_trajectories = np.array(achieved_states)[:,:,0:3]
        else:
            achieved_trajectories = np.array(achieved_states)[:,:,3:6]
        self.observation_array = obs = torch.from_numpy(np.array(achieved_states, np.float32)).to(self.device)
        achieved_trajectories =  torch.from_numpy(np.array(achieved_trajectories, np.float32)).to(self.device)
        achieved_init_states = torch.tensor(achieved_pool_init_state,dtype=torch.float32).to(self.device)
        self.diffusion_optimizer.zero_grad()
        pbar = tqdm(range(self.diffusion_training_iteration), desc="Training loop")
        for _ in pbar:  # Training loop
            if agent.env_name == 'FetchPush-v1':
                self.diffusion_goals = self.diffusion(obs[:, -1, :])
                diffusion_loss = self.loss(achieved_trajectories[:,-1,:2].reshape(-1,2), obs[:, -1, :].reshape(-1,25))
                #This normalized state for the value function
                desired_goal_z = desired_goals[np.random.randint(len(desired_goals))][-1]
                self.diffusion_goals = torch.cat((self.diffusion_goals, torch.Tensor([desired_goal_z]).to("cuda:0").repeat(self.diffusion_goals.shape[0], 1)),dim=1)
            elif agent.env_name == 'FetchPickAndPlace-v1':
                self.diffusion_goals = self.diffusion(obs[:, -1, :])
                diffusion_loss = self.loss(achieved_trajectories[:, -1, :3].reshape(-1, 3),obs[:, -1, :].reshape(-1, 25))
                # This normalized state for the value function
            elif agent.env_name == 'FetchSlide-v1':
                self.diffusion_goals = self.diffusion(obs[:, -1, :])
                diffusion_loss = self.loss(achieved_trajectories[:,-1,:2].reshape(-1,2), obs[:, -1, :].reshape(-1,25))
                #This normalized state for the value function
                desired_goal_z = desired_goals[np.random.randint(len(desired_goals))][-1]
                self.diffusion_goals = torch.cat((self.diffusion_goals, torch.Tensor([desired_goal_z]).to("cuda:0").repeat(self.diffusion_goals.shape[0], 1)),dim=1)
            elif agent.env_name == 'FetchReach-v1':
                self.diffusion_goals = self.diffusion(obs[:, -1, :])
                diffusion_loss = self.loss(achieved_trajectories[:, -1, :3].reshape(-1, 3),obs[:, -1, :].reshape(-1, agent.args.obs_dims[0] - 3))
                # This normalized state for the value function
            else:
                raise NotImplementedError
            critic_input_tensor = torch.hstack((obs[:, -1, :], self.diffusion_goals))
            normalized_critic_input = agent.obs_normalizer.normalize(critic_input_tensor)
            actions = agent.pi(normalized_critic_input)
            critic_value = agent.q(normalized_critic_input,actions)
            critic_loss = critic_value.mean()
            desired_goal =  desired_goals[np.random.randint(len(desired_goals))]
            aim_input_tensor = torch.hstack((self.diffusion_goals, torch.tile(torch.Tensor(desired_goal).to(self.device),[self.diffusion_goals.shape[0], 1]))).to(self.device)
            aim_reward = aim_discriminator.forward(aim_input_tensor)
            aim_reward_loss = aim_reward.mean()
            loss = self.diffusion_loss_coef * diffusion_loss - self.critic_loss_coef * critic_loss - self.aim_reward_loss_coef * aim_reward_loss
            pbar.set_description(f"Diffusion Loss: {loss:.4f}")
            self.diffusion_optimizer.zero_grad()
            loss.backward()
            self.diffusion_optimizer.step()

            self.loss_container.append(
                [loss.clone().tolist(), diffusion_loss.clone().tolist(), critic_loss.clone().tolist(),
                 aim_reward_loss.clone().tolist()])

        if self.debug:
            self.visualizer.plotly_loss_graph(self.loss_container, episode_number)
            self.visualizer.critic_visualizer(obs[:, -1,  :], agent, episode_number)
            self.visualizer.aim_reward_visualizer(aim_discriminator, episode_number, desired_goal)

    def sample_goal(self, obs):
        # self.diffusion_goals = self.diffusion(obs)
        return self.diffusion(torch.Tensor(obs['observation']).to(self.device).reshape(-1, 25)).detach().cpu().numpy()


if __name__ == "__main__":
    max_action = [0.7, 1.0, 0.5]
    min_action = [-0.7, 0.2, 0]
    device = "cuda:0"
    agent = None
    diffusion_kwargs ={'diffusion': True, 'train_diffusion_per_epoch': 25, 'diffusion_training_iteration': 300, 'diffusion_n_timesteps': 10, 'loss_type': 'l2', 'lr': 0.0003, 'debug': True, 'diffusion_debugging_queue_size': 300, 'diffusion_plotly_freq': 300, 'save_path_prefix': '${save_path_prefix}/${env}/${now:%Y.%m.%d}/${now:%H%M%S}_test'}

    diffusion_model = DiffusionGoalSampler(state_dim=3, action_dim=3,
                                                max_action=max_action, min_action=min_action, device=device,
                                                agent=agent, diffusion_configuration=diffusion_kwargs)

    diffusion_model.test()
