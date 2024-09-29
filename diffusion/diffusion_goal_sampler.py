from diffusion.utils.diffusion import Diffusion
from diffusion.utils.model import MLP
import torch

from torch.optim.lr_scheduler import CosineAnnealingLR
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from collections import OrderedDict, deque
from tqdm import tqdm
from utils import make_dir

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
        self.diffusion_training_iteration = diffusion_configuration.diffusion_training_iteration
        self.debug_saved_path = make_dir(diffusion_configuration.save_path_prefix ,"diffusion_debug")
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
        self.diffusion = Diffusion(state_dim=self.model_state_dim, action_dim=self.model_action_dim, model=self.model,
                                   loss_type=diffusion_configuration.loss_type, min_action=min_action, max_action=max_action, beta_schedule='vp',
                                   n_timesteps=diffusion_configuration.diffusion_n_timesteps, ).to(self.device)
        lr = diffusion_configuration.lr
        self.diffusion_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=lr)
        self.diffusion_lr_scheduler = CosineAnnealingLR(self.diffusion_optimizer, T_max=1000, eta_min=0.)
        self.counter = 0
        self.debug = diffusion_configuration.debug
        if diffusion_configuration.debug:
            self.loss_container = deque(maxlen=diffusion_configuration.diffusion_debugging_queue_size)
            self.diffusion_goals_container = deque(maxlen=diffusion_configuration.diffusion_debugging_queue_size)
            self.critic_value_container = deque(maxlen=diffusion_configuration.diffusion_debugging_queue_size)
            self.plotly_select_env()
            self.plotly_debug_freq = diffusion_configuration.diffusion_plotly_freq



    def sampler(self, input):
        self.diffusion_goals = self.diffusion(input)
        return self.diffusion_goals

    def loss(self, output, input):
        diffusion_loss = self.diffusion.loss(output, input)
        return diffusion_loss

    def train(self, replay_buffer, goal_env, hgg_achieved_trajectory_pool):
        achieved_pool, achieved_pool_init_state = hgg_achieved_trajectory_pool.pad()
        #here is the observation size =>  goal_env.obs_dim + goal_env.goal_dim + goal_env.goal_dim + action_size
        obs = torch.from_numpy(np.array(achieved_pool, np.float32)).to(self.device)
        # unnormalized obs into the diffusion model (just for testing)
        self.diffusion_optimizer.zero_grad()
        pbar = tqdm(range(self.diffusion_training_iteration), desc="Training loop")
        for _ in pbar:  # Training loop
            self.diffusion_goals = self.diffusion(obs[:, -1, :goal_env.obs_dim])
            diffusion_loss = self.loss(self.diffusion_goals,
                                       obs[:, -1, :goal_env.obs_dim])

            critic_input_tensor = torch.hstack(
                (obs[:, -1, :goal_env.obs_dim + goal_env.goal_dim], self.diffusion_goals))
            critic_input_normalized = self.agent.normalize_obs(torch.clone(critic_input_tensor), self.agent.env_name,
                                                               "diffusion")
            # action = obs[:, -1, goal_env.obs_dim + 2 * goal_env.goal_dim:]
            action = self.agent.actor(critic_input_tensor).sample()
            action = action.clamp(*self.agent.action_range)
            Q1, Q2 = self.agent.critic(critic_input_normalized, action)
            critic_mean_value = Q1.mean()
            critic_loss = critic_mean_value

            aim_input_tensor = torch.hstack((self.diffusion_goals, obs[:, -1,
                                                                   goal_env.obs_dim + goal_env.goal_dim:goal_env.obs_dim + 2 * goal_env.goal_dim]))
            aim_input_normalized = self.agent.normalize_obs(aim_input_tensor, self.agent.env_name, "diffusion")
            aim_reward = self.agent.aim_discriminator.forward(aim_input_normalized)
            if self.agent.aim_reward_normalize:
                aim_reward = (aim_reward - self.agent.aim_rew_mean) / (self.agent.aim_rew_std ** 2.)

            loss = diffusion_loss - 10 * critic_loss - aim_reward.mean()
            pbar.set_description(f"Diffusion Loss: {loss:.4f}")
            # print(loss)
            self.diffusion_optimizer.zero_grad()
            loss.backward()
            self.diffusion_optimizer.step()
            self.counter += 1
            if self.debug:
                self.loss_container.append(
                    [loss.clone().tolist(), diffusion_loss.clone().tolist(), critic_loss.clone().tolist(),
                     aim_reward.mean().clone().tolist()])
                if self.counter % self.plotly_debug_freq == 0:
                    #For debugging purposes
                    self.plotly_value_function_graph(critic_input_tensor, action, self.diffusion_goals,generate_same_size=False, animation=True)
                    self.plotly_loss_graph(self.loss_container)

        return loss

    def sample_goal(self, obs):
        self.diffusion_goals = self.diffusion(obs)
        return self.diffusion_goals

    def plotly_loss_graph(self, loss):
        # first value is the total_lost, then diffusion_lost, critic_loss, aim_reward loss
        fig = make_subplots(rows=2, cols=2)
        loss = np.array(loss)
        step = loss.shape[0]
        total_loss = loss[:, 0]
        diffusion_loss = loss[:, 1]
        critic_loss = loss[:, 2]
        aim_reward = loss[:, 3]
        fig.add_trace(go.Scatter(x=np.arange(step), y=total_loss, name='total_loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(step), y=diffusion_loss, name='diffusion_loss'), row=1, col=2)
        fig.add_trace(go.Scatter(x=np.arange(step), y=critic_loss, name='critic_loss'), row=2, col=1)
        fig.add_trace(go.Scatter(x=np.arange(step), y=aim_reward, name='aim_reward_loss'), row=2, col=2)
        fig.update_layout(width=2600, height=1400, title_text="Loss Graphs")
        fig.write_html(self.debug_saved_path + "/loss_graph"+ str(self.counter)+".html")

    def plotly_value_function_graph(self, critic_input_tensor, action, diffusion_goals, generate_same_size=True, animation=False):
        if generate_same_size:
            # generate data as same size critic_input_tensor
            number_of_points = np.sqrt(critic_input_tensor.shape[0]).astype(int)
            x_debug = torch.linspace(self.min_action[0], self.max_action[0], number_of_points)
            y_debug = torch.linspace(self.min_action[1], self.max_action[1], number_of_points)
            X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
            x_input_debug = X_debug.reshape(-1, 1)
            y_input_debug = Y_debug.reshape(-1, 1)
            virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug)).to(self.device)
            repeated_state_debug = torch.clone(critic_input_tensor).detach()
            repeated_state_debug[:, -2:] = virtual_diffusion_goals_debug
            repeated_state_debug = self.agent.normalize_obs(repeated_state_debug, self.agent.env_name, "diffusion")
            repeated_action_debug = action.clone()
            critic_value_debug = self.agent.critic(repeated_state_debug, repeated_action_debug)[0]
        else:
            number_of_points = 100
            x_debug = torch.linspace(self.min_action[0], self.max_action[0], number_of_points)
            y_debug = torch.linspace(self.min_action[1], self.max_action[1], number_of_points)
            X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
            x_input_debug = X_debug.reshape(-1, 1)
            y_input_debug = Y_debug.reshape(-1, 1)
            obs_debug = critic_input_tensor[0, :].clone().detach()  # copy first row from obs
            virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug)).to(self.device)
            repeated_state_debug = torch.tile(obs_debug, [len(virtual_diffusion_goals_debug), 1])
            repeated_state_debug[:, -2:] = virtual_diffusion_goals_debug
            repeated_state_debug = self.agent.normalize_obs(repeated_state_debug, self.agent.env_name, "diffusion")
            action_debug = action[0, :].clone().detach()
            repeated_action_debug = torch.tile(action_debug, [len(virtual_diffusion_goals_debug), 1])

            critic_value_debug = self.agent.critic(repeated_state_debug, repeated_action_debug)[0]
        fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("2.5k"),
                                            specs=[[{"type": "scatter3d"}]])
        fig.add_trace(self.env_frame , row=1, col=1)

        fig.update_layout(
            width=1600,
            height=1400,
            autosize=False,
            scene=dict(
                xaxis=dict(nticks=4, range=[-30, 30], ),
                yaxis=dict(nticks=4, range=[-30, 30], ),
                zaxis=dict(nticks=4, range=[-25, 5], ),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='manual'
            ),
        )
        critic_value_surface = np.array(critic_value_debug.detach().cpu()).reshape(X_debug.shape)
        surface_plot = go.Surface(x=X_debug, y=Y_debug,
                                  z=critic_value_surface,
                                  colorscale='Viridis')
        fig.add_trace((surface_plot), row=1, col=1)
        diffusion_goals_tmp = np.array(self.diffusion_goals.tolist())

        fig.add_trace(
            go.Scatter3d(x=diffusion_goals_tmp[:, 0], y=diffusion_goals_tmp[:, 1], z=diffusion_goals_tmp[:, 1] * 0,
                         name='diffusion_goals', mode='markers', marker=dict(size=8, color='brown', opacity=0.8)),
            row=1, col=1)

        inputs_norm_tensor_tmp = torch.hstack((virtual_diffusion_goals_debug, repeated_state_debug[:, 8:10]))

        inputs_norm_tensor_tmp = self.agent.normalize_obs(inputs_norm_tensor_tmp, self.agent.env_name, "diffusion")
        aim_reward = self.agent.aim_discriminator.forward(inputs_norm_tensor_tmp)
        surface_plot = go.Surface(x=X_debug, y=Y_debug, z=np.array(aim_reward.detach().cpu()).reshape(X_debug.shape),
                                  colorscale='spectral')
        fig.add_trace((surface_plot), row=1, col=1)

        fig.write_html(self.debug_saved_path + "/value_function"+ str(self.counter) +".html")
        self.diffusion_goals_container.append(diffusion_goals_tmp)
        self.critic_value_container.append(critic_value_surface)

        def plotly_animation():
            def frame_args(duration):
                return {
                    "frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": False,
                    "transition": {"duration": duration, "easing": "linear"}
                }

            number_of_points = 100
            x_debug = torch.linspace(-2, 18, number_of_points)
            y_debug = torch.linspace(-2, 18, number_of_points)
            X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)

            fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Diffusion-model"], specs=[[{"type": "scatter3d"}]])
            diffusion_goals_animation = np.array(self.diffusion_goals_container)
            critic_surface = np.array(self.critic_value_container)
            # test data
            # diffusion_goals_animation = np.random.randint(-2, 15, size=(10, 100, 2))
            # critic_surface = np.random.randint(-2, 2, size=(10, 100, 100))
            fig.add_trace(go.Scatter3d(
                x=diffusion_goals_animation[0, :, 0],
                y=diffusion_goals_animation[0, :, 1],
                z=diffusion_goals_animation[0, :, 1] * 0,
                name='pool',
                mode='markers',
                marker=dict(size=8, color=plotly.colors.DEFAULT_PLOTLY_COLORS[0], opacity=0.8)
            ), row=1, col=1)

            fig.add_trace(self.env_frame, row=1, col=1)

            surface_plot = go.Surface(x=X_debug, y=Y_debug, z=critic_surface[0, :, :], colorscale='Viridis',
                                      opacity=0.7)
            fig.add_trace((surface_plot), row=1, col=1)
            frames = [
                go.Frame(
                    data=[
                        go.Scatter3d(
                            x=pool_goals[:, 0],
                            y=pool_goals[:, 1],
                            z=pool_goals[:, 1] * 0,
                            mode='markers'
                        ),
                        go.Surface(x=X_debug, y=Y_debug, z=critic_surface[k, :, :], colorscale='Viridis', opacity=0.7),
                        self.env_frame,
                    ],
                    traces=[0, 1, 2],  # Update both the scatter plot and surface plot in each frame
                    name=f'frame{k}'
                )
                for k, pool_goals in enumerate(diffusion_goals_animation)
            ]

            fig.update(frames=frames)

            sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ]
                }
            ]

            fig.update_layout(
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(50)],
                                "label": "Play",
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "Pause",
                                "method": "animate",
                            }
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                ],
                sliders=sliders
            )

            fig.update_layout(scene=dict(
                xaxis=dict(nticks=4, range=[-30, 30], ),
                yaxis=dict(nticks=4, range=[-30, 30], ),
                zaxis=dict(nticks=4, range=[-25, 5], ),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='manual'
            ))

            fig.write_html(self.debug_saved_path + "/value_functon_animation"+str(self.counter) +".html")

        if animation:
            plotly_animation()

        return None

    def plotly_select_env(self):
        if self.agent.env_name == 'PointUMaze-v0':
            self.env_frame = go.Mesh3d(
                # 8 vertices of a cube
                x=np.array([0, 20, 0, 20, 0, 0, 20, 20, 4, 4, 4, 4, 16, 16, 16, 16, 4, 4, 4, 4, 12, 12, 12, 12]) - 6,
                y=np.array([20, 20, 20, 20, 0, 0, -0, -0, 16, 12, 16, 12, 16, 16, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12]) - 6,
                z=np.array([2, 2, 0, 0, 2, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, ]),

                i=[0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 9, 11, 1, 3, 18, 16, 16, 14,
                   14, 12, 12, 8, 8, 9, 9, 22, 11, 21, 11, 11, 10, 10, 13, 13, 15, 15, 17, 17],
                j=[1, 1, 0, 0, 4, 4, 9, 9, 8, 8, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22, 6, 6, 0, 0, 4, 4, 6, 6,
                   1, 1, 0, 0, 18, 18, 23, 19, 19, 5, 5, 2, 2, 3, 3, 7, 7, 5],
                k=[2, 3, 5, 2, 7, 5, 10, 11, 13, 10, 15, 13, 17, 15, 19, 17, 21, 19, 23, 21, 11, 23, 3, 7, 16, 4, 14, 6,
                   12, 1, 8, 0, 9, 18, 22, 20, 21, 11, 5, 10, 2, 13, 3, 15, 7, 17, 5, 19],
                opacity=0.2,
                color='#DC143C',
                name='input',
            )
        elif self.agent.env_name == 'PointNMaze-v0':
            self.env_frame = go.Mesh3d(
                # 8 vertices of a cube
                x=np.array([20., -0., 20., 0., 20., 20., -0., 0., 8., 16., 8., 16., 8.,
                            8., 16., 16., 16., 16., 4., 4., 4., 4., 12., 12., 12., 12.,
                            4., 4., 4., 4., 16., 16.]) - 6,

                y=np.array([0., 0., 0., 0., 28., 28., 28., 28., 20., 20., 20., 20., 16.,
                            16., 16., 16., 4., 4., 4., 4., 8., 8., 8., 8., 12., 12.,
                            12., 12., 24., 24., 24., 24.]) - 6,

                z=np.array([2., 2., 0., 0., 2., 0., 2., 0., 2., 2., 0., 0., 2., 0., 2., 0., 2.,
                            0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.]),

                i=[0, 2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                   31, 9, 11, 1, 3, 14, 20, 0, 1, 1, 22, 20, 20, 6, 6, 4, 4, 8, 9, 9, 4, 27, 29, 29, 31, 31, 25, 27, 27,
                   3, 3, 2, 2, 13, 15, 15, 2],
                j=[1, 1, 0, 0, 4, 4, 9, 9, 8, 8, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22, 24, 24, 26, 26, 28, 28,
                   30, 30, 6, 6, 16, 6, 16, 16, 18, 24, 24, 26, 26, 28, 28, 30, 12, 12, 14, 14, 3, 3, 7, 7, 5, 23, 23,
                   21, 21, 19, 19, 17, 10, 10, 11, 11],
                k=[2, 3, 5, 2, 7, 5, 10, 11, 13, 10, 15, 13, 17, 15, 19, 17, 21, 19, 23, 21, 25, 23, 27, 25, 29, 27, 31,
                   29, 11, 31, 3, 7, 0, 1, 1, 18, 20, 20, 26, 6, 28, 4, 30, 9, 9, 14, 4, 0, 29, 7, 31, 5, 11, 27, 21, 3,
                   19, 2, 17, 15, 15, 11, 2, 5],
                opacity=0.2,
                color='#DC143C',
                name='input',
            )
        elif self.agent.env_name == 'PointSpiralMaze-v0':
            self.env_frame = go.Mesh3d(
                # 8 vertices of a cube
                x=np.array([24., 8., 24., 8., 24., 24., 4., 4., 4., 4., 24., 24., 24.,
                            24., 12., 12., 12., 12., 20., 20., 20., 20., 8., 8., 28., -0.,
                            28., 0., 28., 28., -0., 0.]),
                y=np.array([8., 8., 8., 8., 4., 4., 4., 4., 24., 24., 24., 24., 12.,
                            12., 12., 12., 16., 16., 16., 16., 20., 20., 20., 20., 0., 0.,
                            0., 0., 28., 28., 28., 28.]),
                z=np.array([2., 2., 0., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2.,
                            0., 2., 0., 2., 0., 2., 0., 2., 2., 0., 0., 2., 0., 2., 0.]),
                i=[0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 29, 30,
                   31, 25, 27, 1, 3, 0, 24, 25, 25, 30, 30, 28, 28, 18, 16, 16, 14, 14, 12, 12, 28, 21, 23, 23, 3, 3, 2,
                   5, 7, 7, 9, 9, 11, 11, 13, 13, 2],
                j=[1, 1, 0, 0, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 25, 25, 24, 24, 28, 28,
                   30, 30, 22, 22, 4, 4, 4, 6, 6, 8, 8, 10, 20, 20, 22, 22, 1, 1, 0, 0, 19, 19, 17, 17, 15, 15, 26, 26,
                   27, 27, 31, 31, 29, 29, 26, 26],
                k=[2, 3, 5, 2, 7, 5, 9, 7, 11, 9, 13, 11, 15, 13, 17, 15, 19, 17, 21, 19, 23, 21, 26, 27, 29, 26, 31,
                   29, 27, 31, 3, 23, 24, 25, 6, 30, 8, 28, 10, 12, 16, 22, 14, 1, 12, 0, 28, 24, 23, 17, 3, 15, 2, 13,
                   7, 27, 9, 31, 11, 29, 13, 26, 2, 5],
                opacity=0.2,
                color='#DC143C',
                name='input',
            )
        else:
            raise NotImplementedError


if __name__ == "__main__":
    tmp = DiffusionGoalSampler(state_dim=3, action_dim=3)
    tmp.test()
