import copy
import numpy as np
import torch

from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int

class TrajectoryPool:
	def __init__(self, args, pool_length):
		self.args = args
		self.length = pool_length

		self.pool = []
		self.pool_init_state = []
		self.counter = 0

	def insert(self, trajectory, init_state):
		if self.counter<self.length:
			self.pool.append(trajectory.copy())
			self.pool_init_state.append(init_state.copy())
		else:
			self.pool[self.counter%self.length] = trajectory.copy()
			self.pool_init_state[self.counter%self.length] = init_state.copy()
		self.counter += 1

	def pad(self):
		if self.counter>=self.length:
			return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
		pool = copy.deepcopy(self.pool)
		pool_init_state = copy.deepcopy(self.pool_init_state)
		while len(pool)<self.length:
			pool += copy.deepcopy(self.pool)
			pool_init_state += copy.deepcopy(self.pool_init_state)
		return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

class MatchSampler:
	def __init__(self, args, achieved_trajectory_pool):
		self.args = args
		self.device = args.device
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
		self.delta = self.env.distance_threshold
		self.goal_distance = get_goal_distance(args)

		self.length = args.episodes
		init_goal = self.env.reset()['achieved_goal'].copy()
		self.pool = np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
		self.init_state = self.env.reset()['observation'].copy()

		self.match_lib = gcc_load_lib('learner/cost_flow.c')
		self.achieved_trajectory_pool = achieved_trajectory_pool

		# estimating diameter
		self.max_dis = 0
		for i in range(1000):
			obs = self.env.reset()
			dis = self.goal_distance(obs['achieved_goal'],obs['desired_goal'])
			if dis>self.max_dis: self.max_dis = dis

	def add_noise(self, pre_goal, noise_std=None):
		goal = pre_goal.copy()
		dim = 2 if self.args.env[:5]=='Fetch' else self.dim
		if noise_std is None: noise_std = self.delta
		goal[:dim] += np.random.normal(0, noise_std, size=dim)
		return goal.copy()

	def sample(self, idx):
		if self.args.env[:5]=='Fetch':
			# return self.add_noise(self.pool[idx])
			return self.pool[idx]
		else:
			return self.pool[idx].copy()

	def find(self, goal):
		res = np.sqrt(np.sum(np.square(self.pool-goal),axis=1))
		idx = np.argmin(res)
		if test_pool:
			self.args.logger.add_record('Distance/sampler', res[idx])
		return self.pool[idx].copy()

	def update(self, initial_goals, desired_goals, diffusion_model, aim_discriminator):
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy(desired_goals)
			return

		achieved_states, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
		diffusion_sampled_goals = diffusion_model.diffusion(torch.tensor(np.array(achieved_states), dtype=torch.float32).to(self.device).reshape(-1, self.args.obs_dims[0] - 3))
		if self.args.env in ["FetchPush-v1", "FetchSlide-v1"]:
			diffusion_sampled_goals = torch.cat((diffusion_sampled_goals, torch.Tensor([desired_goals[0][-1]]).to(self.args.device).repeat(diffusion_sampled_goals.shape[0], 1)),dim=1)
		achieved_pool = diffusion_sampled_goals.reshape(-1,self.args.timesteps+1,3).detach().cpu().numpy()
		# achieved_pool = np.array(achieved_states)[:,:,3:6]
		candidate_goals = []
		candidate_edges = []
		candidate_id = []

		desired_goal_extended = np.vstack((np.array(desired_goals), np.array(desired_goals)[0]))
		agent = self.args.agent
		achieved_value = []
		aim_reward_value = []
		for i in range(len(achieved_pool)):
			obs = [ goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for  j in range(achieved_pool[i].shape[0])]
			obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
			obs = agent.obs_normalizer.normalize(obs)
			actions = agent.pi(obs)
			value = agent.q(obs,actions)[:,0].detach().cpu().numpy()
			value = np.clip(value, -1.0/(1.0-self.args.gamma), 0)
			achieved_value.append(value.copy())
			# aim_reward_input = torch.tensor(np.hstack((achieved_pool[i], np.array(desired_goal_extended))),dtype=torch.float32).to(self.device)
			# aim_discriminator_output = aim_discriminator.forward(aim_reward_input).detach().cpu().numpy().squeeze()
			# aim_reward_value.append(aim_discriminator_output.copy())
		n = 0
		graph_id = {'achieved':[],'desired':[]}
		for i in range(len(achieved_pool)):
			n += 1
			graph_id['achieved'].append(n)
		for i in range(len(desired_goals)):
			n += 1
			graph_id['desired'].append(n)
		n += 1
		self.match_lib.clear(n)

		for i in range(len(achieved_pool)):
			self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
		for i in range(len(achieved_pool)):
			for j in range(len(desired_goals)):

				res = np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1))
				match_dis = np.min(res)+self.goal_distance(achieved_pool[i][0], initial_goals[j])*self.args.hgg_c
				match_idx = np.argmin(res)

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				candidate_goals.append(achieved_pool[i][match_idx])
				candidate_edges.append(edge)
				candidate_id.append(j)
		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0,n)
		assert match_count==self.length

		explore_goals = [0]*self.length
		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i])==1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
		assert len(explore_goals)==self.length
		self.pool = np.array(explore_goals)

class DiffusionLearner:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.goal_distance = get_goal_distance(args)
		self.device = self.args.device

		self.env_List = []
		for i in range(args.episodes):
			self.env_List.append(make_env(args))

		self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
		self.sampler = MatchSampler(args, self.achieved_trajectory_pool)
		self.counter = 0

	def learn(self, args, env, env_test, agent, buffer, diffusion_model, aim_discriminator):
		initial_goals = []
		desired_goals = []
		env_goal_temporary_container = []
		for i in range(args.episodes):
			obs = self.env_List[i].reset()
			goal_a = obs['achieved_goal'].copy()
			goal_d = obs['desired_goal'].copy()
			initial_goals.append(goal_a.copy())
			desired_goals.append(goal_d.copy())

		self.sampler.update(initial_goals, desired_goals, diffusion_model, aim_discriminator)

		achieved_trajectories = []
		achieved_init_states = []
		achieved_states = []
		for i in range(args.episodes):
			obs = self.env_List[i].get_obs()
			init_state = obs['observation'].copy()
			explore_goal = self.sampler.sample(i)
			self.env_List[i].goal = explore_goal.copy()
			obs = self.env_List[i].get_obs()
			current = Trajectory(obs)
			trajectory = [obs['achieved_goal'].copy()]
			traj_achieved_states = [obs['observation'].copy()]
			for timestep in range(args.timesteps):
				action = agent.step(obs, explore=True)
				obs, reward, done, info = self.env_List[i].step(action)
				trajectory.append(obs['achieved_goal'].copy())
				traj_achieved_states.append(obs['observation'].copy())
				if timestep==args.timesteps-1: done = True
				current.store_step(action, obs, reward, done)
				if done: break
			env_goal_temporary = explore_goal
			env_goal_temporary_container.append(env_goal_temporary)
			achieved_trajectories.append(np.array(trajectory))
			achieved_states.append(np.array(traj_achieved_states))
			achieved_init_states.append(init_state)
			buffer.store_trajectory(current)
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
				agent.target_update()

		selection_trajectory_idx = {}
		for i in range(self.args.episodes):
			if self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01:
				selection_trajectory_idx[i] = True
		for idx in selection_trajectory_idx.keys():
			self.achieved_trajectory_pool.insert(achieved_states[idx].copy(), achieved_init_states[idx].copy())

		sample_buffer = buffer.sample_batch()
		if self.args.env =="FetchReach-v1":
			achieved_states_tensor = torch.tensor(np.array(np.array(sample_buffer['obs'])[:, 0:3]),dtype=torch.float32).to(self.device)
			achieved_states_tensor_next = torch.tensor(np.array(np.array(sample_buffer['obs_next'])[:, 0:3]),dtype=torch.float32).to(self.device)
		else:
			achieved_states_tensor = torch.tensor(np.array(np.array(sample_buffer['obs'])[:,3:6]),dtype=torch.float32).to(self.device)
			achieved_states_tensor_next = torch.tensor(np.array(np.array(sample_buffer['obs_next'])[:, 3:6]),dtype=torch.float32).to(self.device)


		desired_states_tensor  = torch.tile(torch.tensor(np.array(desired_goals[0]), dtype=torch.float32).to(self.device),[achieved_states_tensor.shape[0], 1])
		policy_states = torch.cat([achieved_states_tensor,desired_states_tensor], dim=-1)
		policy_next_states = torch.cat([achieved_states_tensor_next,desired_states_tensor], dim=-1)
		target_states = torch.cat([desired_states_tensor + torch.from_numpy(np.random.normal(scale=0.01, size=desired_states_tensor.shape)).float().to(self.device),desired_states_tensor], dim=-1)  # s_g, s_g
		aim_disc_loss, wgan_loss, graph_penalty, min_aim_f_loss = aim_discriminator.optimize_discriminator(target_states, policy_states, policy_next_states)


		diffusion_model.train(agent, aim_discriminator, self.achieved_trajectory_pool, desired_goals, self.counter)
		np.save("log/"+ self.args.log_subfolder_name + "/debug/env_goals" +str(self.counter) + ".npy", np.array(env_goal_temporary_container))
		self.counter += 1
