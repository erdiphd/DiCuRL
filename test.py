import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir
import torch
class Tester:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.counter = 0
		self.info = []
		if args.save_acc:
			make_dir('log/' + args.log_subfolder_name + '/accs', clear=False)
			make_dir('log/' + args.log_subfolder_name + '/debug', clear=False)
			self.test_rollouts = 100

			self.env_List = []
			self.env_test_List = []
			for _ in range(self.test_rollouts):
				self.env_List.append(make_env(args))
				self.env_test_List.append(make_env(args))

			self.acc_record = {}
			self.acc_record[self.args.goal] = []
			for key in self.acc_record.keys():
				self.info.append('Success/'+key+'@blue')

		self.trajectory_container = np.zeros((self.test_rollouts, self.args.timesteps, 6))
	def test_acc(self, key, env, agent):
		acc_sum, obs = 0.0, []
		for i in range(self.test_rollouts):
			obs.append(goal_based_process(env[i].reset()))
		for timestep in range(self.args.timesteps):
			actions = agent.step_batch(obs)
			obs, infos = [], []
			for i in range(self.test_rollouts):
				ob, _, _, info = env[i].step(actions[i])
				obs.append(goal_based_process(ob))
				self.trajectory_container[i,timestep,:] = np.concatenate((ob['achieved_goal'], ob['desired_goal']))
				infos.append(info)
		for i in range(self.test_rollouts):
			acc_sum += infos[i]['Success']

		steps = self.args.buffer.counter
		acc = acc_sum/self.test_rollouts
		self.acc_record[key].append((steps,acc))
		self.args.logger.add_record('Success/'+key, acc)
		if self.args.debug:
			np.save("log/" + self.args.log_subfolder_name + "/debug/test_obs" + str(self.counter) + ".npy", self.trajectory_container)
			torch.save(agent.pi.state_dict(), "log/" + self.args.log_subfolder_name + "/debug/policy"+str(self.counter) + ".pth")
			torch.save(agent.obs_normalizer.std, "log/" + self.args.log_subfolder_name + "/debug/std" + str(self.counter) + ".pth")
			torch.save(agent.obs_normalizer.mean, "log/" + self.args.log_subfolder_name + "/debug/mean" + str(self.counter) + ".pth")
		self.counter += 1

	def cycle_summary(self):
		if self.args.save_acc:
			self.test_acc(self.args.goal, self.env_List, self.args.agent)

	def epoch_summary(self):
		if self.args.save_acc:
			for key, acc_info in self.acc_record.items():
				log_folder = 'accs'
				if self.args.tag!='': log_folder = self.args.log_subfolder_name + log_folder+'/'+self.args.tag
				self.args.logger.save_npz(acc_info, key, log_folder)

	def final_summary(self):
		if self.args.save_acc:
			for key, acc_info in self.acc_record.items():
				log_folder = 'accs'
				if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
				self.args.logger.save_npz(acc_info, key, log_folder)