import gym
import numpy as np
from .fixobj import FixedObjectGoalEnv

class IntervalGoalEnv(FixedObjectGoalEnv):
	def __init__(self, args):
		FixedObjectGoalEnv.__init__(self, args)
		self.args = args

	def generate_goal2(self):
		if self.args.env == "FetchSlide-v1":
			self.target_range_x = 0.1
			self.target_range_y = 0.2
		elif self.args.env == "FetchPickAndPlace-v1":
			self.target_range_x = 0.1
			self.target_range_y = 0.1
		elif self.args.env == "FetchReach-v1":
			self.target_range_x = 0
			self.target_range_y = 0
		elif self.args.env == "FetchPush-v1":
			self.target_range_x = 0.1
			self.target_range_y = 0.1
		else:
			raise NotImplementedError

		self.obj_range_x = 0.15
		self.obj_range_y = 0.10


		table_position = self.sim.data.get_body_xpos("table0")
		self.target_center = self.sim.data.get_site_xpos('goal_center')
		# self.init_center = self.sim.data.get_site_xpos('init_center')
		# It is relative pose with respect to table position
		site_id = self.sim.model.site_name2id('goal_1')
		self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, self.target_range_y,
																 0] - table_position
		site_id = self.sim.model.site_name2id('goal_2')
		self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, self.target_range_y,
																 0] - table_position
		site_id = self.sim.model.site_name2id('goal_3')
		self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, -self.target_range_y,
																 0] - table_position
		site_id = self.sim.model.site_name2id('goal_4')
		self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, -self.target_range_y,
																 0] - table_position

		# site_id = self.sim.model.site_name2id('init_1')
		# self.sim.model.site_pos[site_id] = self.init_center + [self.obj_range_x, self.obj_range_y, 0.0] - table_position
		# site_id = self.sim.model.site_name2id('init_2')
		# self.sim.model.site_pos[site_id] = self.init_center + [self.obj_range_x, -self.obj_range_y,
		# 													   0.0] - table_position
		# site_id = self.sim.model.site_name2id('init_3')
		# self.sim.model.site_pos[site_id] = self.init_center + [-self.obj_range_x, self.obj_range_y,
		# 													   0.0] - table_position
		# site_id = self.sim.model.site_name2id('init_4')
		# self.sim.model.site_pos[site_id] = self.init_center + [-self.obj_range_x, -self.obj_range_y,
		# 													   0.0] - table_position

		goal = self.target_center.copy()
		goal[1] += self.np_random.uniform(-self.target_range_y, self.target_range_y)
		goal[0] += self.np_random.uniform(-self.target_range_x, self.target_range_x)

		if self.target_in_the_air and self.np_random.uniform() < 0.5:
			goal[2] += self.np_random.uniform(0, 0.45)

		# goal += self.target_offset
		return goal.copy()

	def generate_goal(self):
		if self.has_object:
			goal = self.initial_gripper_xpos[:3] + self.target_offset
			if self.args.env=='FetchSlide-v1':
				goal[0] += self.target_range*0.5
				goal[1] += np.random.uniform(-self.target_range, self.target_range)*0.5
			else:
				goal[0] += np.random.uniform(-self.target_range, self.target_range)
				goal[1] += self.target_range
			goal[2] = self.height_offset + int(self.target_in_the_air)*0.45
		else:
			goal = self.initial_gripper_xpos[:3] + np.array([np.random.uniform(-self.target_range, self.target_range), self.target_range, self.target_range])
		#TODO change this hard-coded goal
		return goal.copy()