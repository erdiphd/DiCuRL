import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import plotly.express as px
import torch
from collections import  deque
import numpy as np
class Visualizer:
    def __init__(self,*args, **kwargs):
        self.number_of_points = kwargs['number_of_points']
        self.x_min, self.x_max = kwargs['x_min'], kwargs['x_max']
        self.y_min, self.y_max = kwargs['y_min'], kwargs['y_max']
        self.env_name = kwargs['env_name']
        if self.env_name in ['FetchPush-v1', 'FetchSlide-v1']:
            pass
        elif self.env_name in ['FetchPickAndPlace-v1', 'FetchReach-v1']:
            self.z_min, self.z_max = kwargs['z_min'], kwargs['z_max']
        else:
            raise NotImplementedError
        self.device = kwargs['device']
        self.env_frame = go.Mesh3d(
            x = np.array([1.55, 1.55, 1.05, 1.05, 1.55, 1.05, 1.05, 1.55]),
            y = np.array([0.4, 1.1,  1.1,  0.4, 0.4,  0.4, 1.1, 1.1]),
            z = np.array([0,    0,    0,   0,   0.1,  0.1, 0.1, 0.1]),
            i = np.array([0, 0, 0, 3, 3, 2, 2, 1, 1, 0]),
            j = np.array([1, 2, 3, 5, 5, 5, 6, 6, 7, 7]),
            k = np.array([2, 3, 4, 4, 2, 6, 1, 7, 0, 4]),
                opacity=0.2,
                color='#DC143C',
                name='input',
            )
        self.sub_folder = kwargs['sub_folder']
    def single_loop_critic_visualizer(self, robot_observation, agent, episode_number):
        x_debug = torch.linspace(self.x_min, self.x_max, self.number_of_points)
        y_debug = torch.linspace(self.y_min, self.y_max, self.number_of_points)
        X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
        x_input_debug = X_debug.reshape(-1, 1)
        y_input_debug = Y_debug.reshape(-1, 1)
        if self.env_name in "FetchPush-v1":
            z_input_debug = torch.tile(torch.tensor([0.42469975]), [len(x_input_debug), 1])
        else:
            raise NotImplementedError

        obs_debug = robot_observation[0, :].clone().detach()  # copy first row from obs
        virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug, z_input_debug)).to(self.device)
        repeated_state_debug = torch.tile(obs_debug, [len(virtual_diffusion_goals_debug), 1])
        critic_input_tensor = torch.hstack((repeated_state_debug, virtual_diffusion_goals_debug))
        normalized_critic_input = agent.obs_normalizer.normalize(critic_input_tensor)
        actions = agent.pi(normalized_critic_input)
        critic_value = agent.q(normalized_critic_input, actions)
        critic_value_surface = critic_value.reshape(X_debug.shape).detach().cpu().numpy()
        fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("2.5k"), specs=[[{"type": "scatter3d"}]])
        fig.update_layout(
            width=1600,
            height=1400,
            autosize=False,
            scene=dict(
                xaxis=dict(nticks=4, range=[0, 2], ),
                yaxis=dict(nticks=4, range=[0, 2], ),
                zaxis=dict(nticks=4, range=[-55, 5], ),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='manual'
            ),
        )
        fig.add_trace(self.env_frame, row=1, col=1)
        surface_plot = go.Surface(x=X_debug, y=Y_debug,
                                  z=critic_value_surface,
                                  colorscale='Viridis')
        fig.add_trace((surface_plot), row=1, col=1)
        fig.write_html("log/" + self.sub_folder + "/debug/value_function.html" + str(episode_number) + ".html")

    def critic_visualizer(self, robot_observation, agent, episode_number):
        x_debug = torch.linspace(self.x_min, self.x_max, self.number_of_points)
        y_debug = torch.linspace(self.y_min, self.y_max, self.number_of_points)
        X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
        x_input_debug = X_debug.reshape(-1, 1)
        y_input_debug = Y_debug.reshape(-1, 1)
        if self.env_name in ["FetchPush-v1", "FetchSlide-v1"]:
            z_input_debug = torch.tile(torch.tensor([0.42469975]), [len(x_input_debug), 1])
        else:
            raise NotImplementedError

        critic_value_debug_container = []
        for i in range(robot_observation.shape[0]):
            obs_debug = robot_observation[i, :].clone().detach()  # copy first row from obs
            virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug, z_input_debug)).to(self.device)
            repeated_state_debug = torch.tile(obs_debug, [len(virtual_diffusion_goals_debug), 1])
            critic_input_tensor = torch.hstack((repeated_state_debug, virtual_diffusion_goals_debug))
            normalized_critic_input = agent.obs_normalizer.normalize(critic_input_tensor)
            actions = agent.pi(normalized_critic_input)
            critic_value_debug = agent.q(normalized_critic_input, actions)
            critic_value_debug_container.append(critic_value_debug.detach().cpu().numpy())

        critic_value_surface = np.mean(np.array(critic_value_debug_container), axis=0).reshape(X_debug.shape)

        fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("2.5k"), specs=[[{"type": "scatter3d"}]])
        fig.update_layout(
            width=1600,
            height=1400,
            autosize=False,
            scene=dict(
                xaxis=dict(nticks=4, range=[0, 2], ),
                yaxis=dict(nticks=4, range=[0, 2], ),
                zaxis=dict(nticks=4, range=[-55, 5], ),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='manual'
            ),
        )
        fig.add_trace(self.env_frame, row=1, col=1)
        surface_plot = go.Surface(x=X_debug, y=Y_debug,
                                  z=critic_value_surface,
                                  colorscale='Viridis')
        fig.add_trace((surface_plot), row=1, col=1)
        fig.write_html("log/" + self.sub_folder + "/debug/value_function.html" + str(episode_number) + ".html")

    def aim_reward_visualizer(self, aim_discriminator, episode_number, with_respect_to):
        x_debug = torch.linspace(self.x_min, self.x_max, self.number_of_points)
        y_debug = torch.linspace(self.y_min, self.y_max, self.number_of_points)
        X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
        x_input_debug = X_debug.reshape(-1, 1)
        y_input_debug = Y_debug.reshape(-1, 1)
        if self.env_name in ["FetchPush-v1", "FetchSlide-v1"]:
            z_input_debug = torch.tile(torch.tensor([0.42469975]), [len(x_input_debug), 1])
        else:
            raise NotImplementedError
        virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug, z_input_debug)).to(self.device)
        inputs_norm_tensor_tmp = torch.hstack((virtual_diffusion_goals_debug, torch.tile(torch.Tensor(with_respect_to).to(self.device),[virtual_diffusion_goals_debug.shape[0], 1])))
        aim_reward = aim_discriminator.forward(inputs_norm_tensor_tmp)
        fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("2.5k"), specs=[[{"type": "scatter3d"}]])
        fig.update_layout(
            width=1600,
            height=1400,
            autosize=False,
            scene=dict(
                xaxis=dict(nticks=4, range=[0, 2], ),
                yaxis=dict(nticks=4, range=[0, 2], ),
                zaxis=dict(nticks=4, range=[-25, 5], ),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='manual'
            ),
        )
        fig.add_trace(self.env_frame, row=1, col=1)
        surface_plot = go.Surface(x=X_debug, y=Y_debug, z=np.array(aim_reward.detach().cpu()).reshape(X_debug.shape),
                                  colorscale='spectral')
        fig.add_trace((surface_plot), row=1, col=1)
        fig.write_html("log/" + self.sub_folder + "/debug/aim_reward.html" + str(episode_number) + ".html")


    def plotly_loss_graph(self, loss, episode_number):
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
        fig.write_html("log/" + self.sub_folder + "/debug/loss_board" + str(episode_number) + ".html")

    def critic_visualizer_3d(self, robot_observation, agent, episode_number):
        x_debug = torch.linspace(self.x_min, self.x_max, self.number_of_points)
        y_debug = torch.linspace(self.y_min, self.y_max, self.number_of_points)
        z_debug = torch.linspace(self.z_min, self.z_max, self.number_of_points)
        X_debug, Y_debug, Z_debug = torch.meshgrid(x_debug, y_debug, z_debug)
        x_input_debug = X_debug.reshape(-1, 1)
        y_input_debug = Y_debug.reshape(-1, 1)
        z_input_debug = Z_debug.reshape(-1, 1)


        critic_value_debug_container = []
        for i in range(robot_observation.shape[0]):
            obs_debug = robot_observation[i, :].clone().detach()  # copy first row from obs
            virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug, z_input_debug)).to(self.device)
            repeated_state_debug = torch.tile(obs_debug, [len(virtual_diffusion_goals_debug), 1])
            critic_input_tensor = torch.hstack((repeated_state_debug, virtual_diffusion_goals_debug))
            normalized_critic_input = agent.obs_normalizer.normalize(critic_input_tensor)
            actions = agent.pi(normalized_critic_input)
            critic_value_debug = agent.q(normalized_critic_input, actions)
            critic_value_debug_container.append(critic_value_debug.detach().cpu().numpy())

        critic_value_surface = np.mean(np.array(critic_value_debug_container), axis=0).reshape(X_debug.shape)

        fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("2.5k"), specs=[[{"type": "scatter3d"}]])
        fig.update_layout(
            width=1600,
            height=1400,
            autosize=False,
            scene=dict(
                xaxis=dict(nticks=4, range=[0, 2], ),
                yaxis=dict(nticks=4, range=[0, 2], ),
                zaxis=dict(nticks=4, range=[-25, 5], ),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='manual'
            ),
        )
        # fig.add_trace(self.env_frame, row=1, col=1)
        surface_plot = go.Isosurface(
                            x=X_debug.flatten(),
                            y=Y_debug.flatten(),
                            z=Z_debug.flatten(),
                            value=critic_value_surface.flatten(),
                            isomin=critic_value_surface.min(),
                            isomax=critic_value_surface.max(),
                            caps=dict(x_show=False, y_show=False, z_show=False),
                            surface_count=10)
        fig.add_trace((surface_plot), row=1, col=1)
        fig.write_html("log/" + self.sub_folder + "/debug/value_function3d" + str(episode_number) + ".html")


    def aim_reward_visualizer3d(self, aim_discriminator, episode_number, with_respect_to):
        x_debug = torch.linspace(self.x_min, self.x_max, self.number_of_points)
        y_debug = torch.linspace(self.y_min, self.y_max, self.number_of_points)
        z_debug = torch.linspace(self.z_min, self.z_max, self.number_of_points)
        X_debug, Y_debug, Z_debug = torch.meshgrid(x_debug, y_debug, z_debug)
        x_input_debug = X_debug.reshape(-1, 1)
        y_input_debug = Y_debug.reshape(-1, 1)
        z_input_debug = Z_debug.reshape(-1, 1)
        virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug, z_input_debug)).to(self.device)
        inputs_norm_tensor_tmp = torch.hstack((virtual_diffusion_goals_debug, torch.tile(torch.Tensor(with_respect_to).to(self.device),[virtual_diffusion_goals_debug.shape[0], 1])))
        aim_reward = aim_discriminator.forward(inputs_norm_tensor_tmp)
        fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("2.5k"), specs=[[{"type": "scatter3d"}]])
        fig.update_layout(
            width=1600,
            height=1400,
            autosize=False,
            scene=dict(
                xaxis=dict(nticks=4, range=[0, 2], ),
                yaxis=dict(nticks=4, range=[0, 2], ),
                zaxis=dict(nticks=4, range=[-25, 5], ),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='manual'
            ),
        )
        fig.add_trace(self.env_frame, row=1, col=1)

        aim_reward_surface = np.array(aim_reward.detach().cpu()).reshape(X_debug.shape)
        surface_plot = go.Isosurface(
                            x=X_debug.flatten(),
                            y=Y_debug.flatten(),
                            z=Z_debug.flatten(),
                            value=aim_reward_surface.flatten(),
                            isomin=aim_reward_surface.min(),
                            isomax=aim_reward_surface.max(),
                            caps=dict(x_show=False, y_show=False, z_show=False),
                            surface_count=10)
        fig.add_trace((surface_plot), row=1, col=1)
        fig.write_html("log/" + self.sub_folder + "/debug/aim_reward3d" + str(episode_number) + ".html")
# number_of_points = 50
# critic_value_debug_container = []
# x_debug = torch.linspace(self.min_action[0], self.max_action[0], number_of_points)
# y_debug = torch.linspace(self.min_action[1], self.max_action[1], number_of_points)
# X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
# x_input_debug = X_debug.reshape(-1, 1)
# y_input_debug = Y_debug.reshape(-1, 1)
# z_input_debug = torch.tile(torch.tensor([0.42469975]), [len(x_input_debug), 1])
# robot_observation = obs[:, -1,  :]
# obs_debug = robot_observation[0, :].clone().detach()  # copy first row from obs
# virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug, z_input_debug)).to(self.device)
#
# inputs_norm_tensor_tmp = torch.hstack((virtual_diffusion_goals_debug, torch.tile(torch.Tensor(np.array([1.45, 0.95, 0.42469975])).to(self.device),[virtual_diffusion_goals_debug.shape[0], 1])))
# aim_reward = aim_discriminator.forward(inputs_norm_tensor_tmp)
#
#
# fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("2.5k"),specs=[[{"type": "scatter3d"}]])
# fig.update_layout(
#     width=1600,
#     height=1400,
#     autosize=False,
#     scene=dict(
#         xaxis=dict(nticks=4, range=[-30, 30], ),
#         yaxis=dict(nticks=4, range=[-30, 30], ),
#         zaxis=dict(nticks=4, range=[-25, 5], ),
#         aspectratio=dict(x=1, y=1, z=1),
#         aspectmode='manual'
#     ),
# )
#
# surface_plot = go.Surface(x=X_debug, y=Y_debug,z=np.array(aim_reward.detach().cpu()).reshape(X_debug.shape),colorscale='spectral')
# fig.add_trace((surface_plot), row=1, col=1)
# fig.show()

# ###------------------------------------------------------------------------------------------------------------------------------------
# number_of_points = 50
# critic_value_debug_container = []
# x_debug = torch.linspace(self.min_action[0], self.max_action[0], number_of_points)
# y_debug = torch.linspace(self.min_action[1], self.max_action[1], number_of_points)
# X_debug, Y_debug = torch.meshgrid(x_debug, y_debug)
# x_input_debug = X_debug.reshape(-1, 1)
# y_input_debug = Y_debug.reshape(-1, 1)
# z_input_debug = torch.tile(torch.tensor([0.42469975]), [len(x_input_debug), 1])
# robot_observation = obs[:, -1,  :]
# for i in range(robot_observation.shape[0]):
#     obs_debug = robot_observation[i, :].clone().detach()  # copy first row from obs
#     virtual_diffusion_goals_debug = torch.hstack((x_input_debug, y_input_debug, z_input_debug)).to(self.device)
#     repeated_state_debug = torch.tile(obs_debug, [len(virtual_diffusion_goals_debug), 1])
#     critic_input_tensor = torch.hstack((repeated_state_debug, virtual_diffusion_goals_debug))
#     normalized_critic_input = agent.obs_normalizer.normalize(critic_input_tensor)
#     actions = agent.pi(normalized_critic_input)
#     critic_value_debug = agent.q(normalized_critic_input, actions)
#     critic_value_debug_container.append(critic_value_debug.detach().cpu().numpy())
#
#
# critic_value_surface = np.mean(np.array(critic_value_debug_container),axis=0).reshape(X_debug.shape)
# fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=("2.5k"),specs=[[{"type": "scatter3d"}]])
# fig.update_layout(
#     width=1600,
#     height=1400,
#     autosize=False,
#     scene=dict(
#         xaxis=dict(nticks=4, range=[0, 2], ),
#         yaxis=dict(nticks=4, range=[0, 2], ),
#         zaxis=dict(nticks=4, range=[-25, 5], ),
#         aspectratio=dict(x=1, y=1, z=1),
#         aspectmode='manual'
#     ),
# )
#
#
# surface_plot = go.Surface(x=X_debug, y=Y_debug,
#                             z=critic_value_surface,
#                             colorscale='Viridis')
# fig.add_trace((surface_plot), row=1, col=1)
# fig.show()
#


