import copy
import numpy as np
from hgg.gcc_utils import gcc_load_lib, c_double, c_int
import torch
import torch.nn.functional as F
import time
from scipy.stats import wasserstein_distance
def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a - goal_b, ord=2)
def goal_concat(obs, goal):
	return np.concatenate([obs, goal], axis=0)

class TrajectoryPool:
	def __init__(self, pool_length):
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
	def __init__(self, goal_env, goal_eval_env, env_name, achieved_trajectory_pool, num_episodes,				
				agent = None, max_episode_timesteps =None, split_ratio_for_meta_nml=0.1, split_type_for_meta_nml='last', normalize_aim_output=False,
				add_noise_to_goal= False, cost_type='meta_nml_aim_f', gamma=0.99, hgg_c=3.0, hgg_L=5.0, device = 'cuda', hgg_gcc_path = None
				
				):
		# Assume goal env
		self.env = goal_env
		self.eval_env = goal_eval_env
		self.env_name = env_name
		
		self.add_noise_to_goal = add_noise_to_goal
		self.cost_type = cost_type
		self.agent = agent
		
		self.vf = None
		self.critic = None
		self.policy = None

		self.max_episode_timesteps = max_episode_timesteps
		self.split_ratio_for_meta_nml = split_ratio_for_meta_nml
		self.split_type_for_meta_nml = split_type_for_meta_nml
		self.normalize_aim_output = normalize_aim_output
		self.gamma = gamma
		self.hgg_c = hgg_c
		self.hgg_L = hgg_L
		self.device = device
  		
		self.success_threshold = {'AntMazeSmall-v0' : 1.0, # 0.5,
								  'PointUMaze-v0' : 0.5,
          						  'PointNMaze-v0' : 0.5,
								  'sawyer_peg_push' : getattr(self.env, 'TARGET_RADIUS', None),
								  'sawyer_peg_pick_and_place' : getattr(self.env, 'TARGET_RADIUS', None),
								  'PointSpiralMaze-v0' : 0.5,
								  'PointLongCorridor-v0': 0.5,
								  }
		self.loss_function = torch.nn.BCELoss(reduction='none')

		self.dim = np.prod(self.env.convert_obs_to_dict(self.env.reset())['achieved_goal'].shape)
		self.delta = self.success_threshold[env_name] #self.env.distance_threshold
		self.goal_distance = goal_distance

		self.length = num_episodes # args.episodes
		
		init_goal = self.eval_env.convert_obs_to_dict(self.eval_env.reset())['achieved_goal'].copy()
		
		self.pool = np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
		

		self.match_lib = gcc_load_lib(hgg_gcc_path+'/cost_flow.c')
		
		self.achieved_trajectory_pool = achieved_trajectory_pool

		# estimating diameter
		self.max_dis = 0
		for i in range(1000):			
			obs = self.env.convert_obs_to_dict(self.env.reset())
			dis = self.goal_distance(obs['achieved_goal'],obs['desired_goal'])
			if dis>self.max_dis: self.max_dis = dis 
	



	def add_noise(self, pre_goal, noise_std=None):
		goal = pre_goal.copy()		
		if noise_std is None: noise_std = self.delta
		
		if self.env_name in ['AntMazeSmall-v0', 'PointUMaze-v0']:
			goal += np.random.normal(0, noise_std, size=2)	
			goal = np.clip(goal, (-2,-2), (10,10))
		elif self.env_name in ['sawyer_peg_pick_and_place']:
			noise = np.random.normal(0, noise_std, size=goal.shape[-1])			
			goal += noise
		elif self.env_name in ['sawyer_peg_push']:
			noise = np.random.normal(0, noise_std, size=goal.shape[-1])
			noise[2] = 0	
			goal += noise
			goal[..., -3:] = np.clip(goal[..., -3:], (-0.6, 0.2, 0.0147), (0.6, 1.0, 0.0148))
		elif self.env_name == "PointSpiralMaze-v0":
			goal += np.random.normal(0, noise_std, size=2)	
			goal = np.clip(goal, (-10,-10), (10,10))
		elif self.env_name in ["PointNMaze-v0"]:
			goal += np.random.normal(0, noise_std, size=2)	
			goal = np.clip(goal, (-2,-2), (10,18))
		elif self.env_name == "PointLongCorridor-v0":
			goal += np.random.normal(0, noise_std, size=2)
			goal = np.clip(goal, (-2,-2), (26,14))
		else:
			raise NotImplementedError

		return goal.copy()

	def sample(self, idx):
		if self.add_noise_to_goal:
			if self.env_name in ['AntMazeSmall-v0', 'PointUMaze-v0', "PointSpiralMaze-v0", "PointNMaze-v0", "PointLongCorridor-v0"]:
				noise_std = 0.5
			elif self.env_name in ['sawyer_peg_push', 'sawyer_peg_pick_and_place']:
				noise_std = 0.05
			else:
				raise NotImplementedError('Should consider noise scale env by env')
			return self.add_noise(self.pool[idx], noise_std = noise_std)
		else:
			return self.pool[idx].copy()

	def update(self,desired_goals, replay_buffer = None, meta_nml_epoch = 0, outpace_train=None):
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy(desired_goals)
			return

		achieved_pool_states, achieved_pool_init_state = self.achieved_trajectory_pool.pad()

		#achieved_pool = np.array(achieved_pool_states)[:,:,outpace_train.env.obs_dim:outpace_train.env.obs_dim+outpace_train.env.goal_dim]
		diffusion_input_pool = np.array(achieved_pool_states)[:,:,:outpace_train.env.obs_dim]
		achieved_pool = outpace_train.diffusion_model.sample_goal(torch.Tensor(diffusion_input_pool).to(self.device).reshape(-1,outpace_train.env.obs_dim)).detach().cpu().numpy().reshape(outpace_train.cfg.hgg_kwargs.trajectory_pool_kwargs.pool_length,outpace_train.max_episode_timesteps+1,2)
		achieved_pool_init_state = np.array(achieved_pool_init_state)[:,:outpace_train.env.obs_dim+outpace_train.env.goal_dim]

		candidate_goals = []
		candidate_edges = []
		candidate_id = []
		# achieved_value = []
		# for i in range(len(achieved_pool)):
		# 	with torch.no_grad():
		# 		obs = [ goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for  j in range(achieved_pool[i].shape[0])]
		# 		obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
		# 		critic_input_normalized = self.agent.normalize_obs(torch.clone(obs),self.agent.env_name,"diffusion")
		# 		dist = self.agent.actor(obs)
		# 		action = dist.sample()
		# 		action = action.clamp(*self.agent.action_range)
		#
		#
		# 		value = 0.5 * self.agent.critic(critic_input_normalized,action)[0].detach().cpu().numpy() + 0.5* self.agent.critic(critic_input_normalized,action)[1].detach().cpu().numpy()
		# 		#value = np.clip(value, -1.0/(1.0-self.args.gamma), 0)
		# 		achieved_value.append(value.copy())
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

				# res = np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1))
				res = []
				# for k in range(len(achieved_pool[i]) - 1):
				# 	distance_dim_1 = wasserstein_distance([achieved_pool[i][k, 0]], [np.array(desired_goals)[k, 0]])
				# 	distance_dim_2 = wasserstein_distance([achieved_pool[i][k, 1]], [np.array(desired_goals)[k, 1]])
				# 	distance_2d = np.sqrt(distance_dim_1 ** 2 + distance_dim_2 ** 2)
				# 	res.append(distance_2d)

				distance_dim_1 = wasserstein_distance(achieved_pool[i][:, 0], np.array(desired_goals)[:, 0])
				distance_dim_2 = wasserstein_distance(achieved_pool[i][:, 1], np.array(desired_goals)[:, 1])
				distance_2d = np.sqrt(distance_dim_1 ** 2 + distance_dim_2 ** 2)

				std_dev_x = np.std(achieved_pool[i][:, 0])
				std_dev_y = np.std(achieved_pool[i][:, 1])
				res.append(-std_dev_x - std_dev_y )

				match_dis = np.min(res)
				match_idx = np.argmin(res)

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				candidate_goals.append(achieved_pool[i][match_idx - 1])
				candidate_edges.append(edge)
				candidate_id.append(j)
		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0,n)
		assert match_count==self.length * 5

		explore_goals = [0]*self.length * 5
		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i])==1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
		assert len(explore_goals)==self.length * 5
		self.pool = np.array(explore_goals)