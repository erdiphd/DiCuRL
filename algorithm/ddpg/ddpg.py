import numpy as np
from envs import goal_distance_obs
from utils.tf_utils import get_vars, Normalizer, Normalizer_torch
from algorithm.replay_buffer import goal_based_process
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class MLP_Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Policy, self).__init__()
        self.pi_dense1 = nn.Linear(input_dim, 256)
        self.pi_dense2 = nn.Linear(256, 256)
        self.pi_dense3 = nn.Linear(256, 256)
        self.pi = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Xavier initialization for weights
        self._initialize_weights()

    def _initialize_weights(self):
        init.xavier_uniform_(self.pi_dense1.weight)
        init.xavier_uniform_(self.pi_dense2.weight)
        init.xavier_uniform_(self.pi_dense3.weight)
        init.xavier_uniform_(self.pi.weight)

        init.constant_(self.pi_dense1.bias, 0)
        init.constant_(self.pi_dense2.bias, 0)
        init.constant_(self.pi_dense3.bias, 0)
        init.constant_(self.pi.bias, 0)
    def forward(self, obs_ph):
        pi_dense1_out = self.relu(self.pi_dense1(obs_ph))
        pi_dense2_out = self.relu(self.pi_dense2(pi_dense1_out))
        pi_dense3_out = self.relu(self.pi_dense3(pi_dense2_out))
        pi = self.tanh(self.pi(pi_dense3_out))
        return pi
class MLP_Value(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(MLP_Value, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.q_dense1 = nn.Linear(self.obs_dim + self.act_dim, 256)
        self.q_dense2 = nn.Linear(256, 256)
        self.q_dense3 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)
        self.relu = nn.ReLU()

        # Xavier initialization for weights
        self._initialize_weights()

    def _initialize_weights(self):
        init.xavier_uniform_(self.q_dense1.weight)
        init.xavier_uniform_(self.q_dense2.weight)
        init.xavier_uniform_(self.q_dense3.weight)
        init.xavier_uniform_(self.q.weight)

        init.constant_(self.q_dense1.bias, 0)
        init.constant_(self.q_dense2.bias, 0)
        init.constant_(self.q_dense3.bias, 0)
        init.constant_(self.q.bias, 0)

    def forward(self, obs_ph, acts_ph):
        state_ph = torch.cat((obs_ph, acts_ph), dim=1)
        q_dense1_out = self.relu(self.q_dense1(state_ph))
        q_dense2_out = self.relu(self.q_dense2(q_dense1_out))
        q_dense3_out = self.relu(self.q_dense3(q_dense2_out))
        q = self.q(q_dense3_out)
        return q



class DDPG:
	def __init__(self, args):
		self.args = args
		self.device = args.device
		self.env_name = self.args.env
		self.q = MLP_Value(self.args.obs_dims[0], self.args.acts_dims[0]).to(self.device)
		self.pi = MLP_Policy(self.args.obs_dims[0], self.args.acts_dims[0]).to(self.device)
		self.pi_t = MLP_Policy(self.args.obs_dims[0], self.args.acts_dims[0]).to(self.device)
		self.q_t = MLP_Value(self.args.obs_dims[0], self.args.acts_dims[0]).to(self.device)


		#Initialize target network with the same weights and biases as in the main network.
		self.pi_t.load_state_dict(self.pi.state_dict())
		self.q_t.load_state_dict(self.q.state_dict())

		#Optimizers
		self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.args.pi_lr)
		self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=self.args.q_lr)

		#Normalizer
		self.obs_normalizer = Normalizer_torch(self.args.obs_dims[0], self.device)

		self.pi_q_loss = 0
		self.pi_l2_loss =0
		self.q_loss = 0

		self.train_info_pi = {
			'Pi_q_loss': self.pi_q_loss,
			'Pi_l2_loss': self.pi_l2_loss
		}
		self.train_info_q = {
			'Q_loss': self.q_loss
		}
		self.train_info = {**self.train_info_pi, **self.train_info_q}

		self.step_info = {
			'Q_average': 10
		}



	def _soft_update_target_network(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

	def step(self, obs, explore=False, test_info=False):
		if (not test_info) and (self.args.buffer.steps_counter<self.args.warmup):
			return np.random.uniform(-1, 1, size=self.args.acts_dims)
		if self.args.goal_based: obs = goal_based_process(obs)

		# eps-greedy exploration
		if explore and np.random.uniform()<=self.args.eps_act:
			return np.random.uniform(-1, 1, size=self.args.acts_dims)

		obs_normalized = self.obs_normalizer.normalize(torch.Tensor(obs).to(self.device))
		action = self.pi(obs_normalized).detach().cpu().numpy().squeeze()

		# uncorrelated gaussian explorarion
		if explore: action += np.random.normal(0, self.args.std_act, size=self.args.acts_dims)
		action = np.clip(action, -1, 1)
		return action


	def step_batch(self, obs):
		obs_normalized = self.obs_normalizer.normalize(torch.Tensor(obs).to(self.device))
		with torch.no_grad():
			actions = self.pi(obs_normalized).detach().cpu().numpy()
		return actions
	def train(self, batch):
		obs_norm = self.obs_normalizer.normalize(torch.Tensor(batch['obs']).to(self.device))
		obs_next_norm = self.obs_normalizer.normalize(torch.Tensor(batch['obs_next']).to(self.device))
		actions_tensor = torch.Tensor(batch['acts']).to(self.device)
		reward_tensor = torch.Tensor(batch['rews']).to(self.device)

		actions_real = self.pi(obs_norm)
		self.pi_q_loss =  -self.q(obs_norm, actions_real).mean()
		self.pi_l2_loss = self.args.act_l2 * torch.mean(torch.square(actions_real))
		self.total_pi_loss = self.pi_q_loss + self.pi_l2_loss

		with torch.no_grad():
			actions_next = self.pi_t(obs_next_norm)
			q_target = self.q_t(obs_next_norm, actions_next)
			if self.args.clip_return:
				return_value = torch.clamp(q_target, self.args.clip_return_l, self.args.clip_return_r)
			else:
				return_value = q_target
			target = (reward_tensor + self.args.gamma * return_value).detach()


		real_q_value = self.q(obs_norm, actions_tensor)

		self.q_loss = torch.mean(torch.square(real_q_value - target))

		self.pi_optimizer.zero_grad()
		self.total_pi_loss.backward()
		self.pi_optimizer.step()

		self.q_optimizer.zero_grad()
		self.q_loss.backward()
		self.q_optimizer.step()

	def normalizer_update(self, batch):
		self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

	def target_update(self):
		self._soft_update_target_network(self.pi_t, self.pi)
		self._soft_update_target_network(self.q_t, self.q)



class DDPG_TF:
	def __init__(self, args):
		self.args = args
		self.create_model()

		self.train_info_pi = {
			'Pi_q_loss': self.pi_q_loss,
			'Pi_l2_loss': self.pi_l2_loss
		}
		self.train_info_q = {
			'Q_loss': self.q_loss
		}
		self.train_info = {**self.train_info_pi, **self.train_info_q}

		self.step_info = {
			'Q_average': self.q_pi
		}


		self.tmp_torch = DDPG_Torch(self.args)
	def create_model(self):
		def create_session():
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			self.sess = tf.Session(config=config)

		def create_inputs():
			self.raw_obs_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
			self.raw_obs_next_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
			self.acts_ph = tf.placeholder(tf.float32, [None]+self.args.acts_dims)
			self.rews_ph = tf.placeholder(tf.float32, [None, 1])

		def create_normalizer():
			with tf.variable_scope('normalizer'):
				self.obs_normalizer = Normalizer(self.args.obs_dims, self.sess)
			self.obs_ph = self.obs_normalizer.normalize(self.raw_obs_ph)
			self.obs_next_ph = self.obs_normalizer.normalize(self.raw_obs_next_ph)

		def create_network():
			def mlp_policy(obs_ph):
				with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
					pi_dense1 = tf.layers.dense(obs_ph, 256, activation=tf.nn.relu, name='pi_dense1')
					pi_dense2 = tf.layers.dense(pi_dense1, 256, activation=tf.nn.relu, name='pi_dense2')
					pi_dense3 = tf.layers.dense(pi_dense2, 256, activation=tf.nn.relu, name='pi_dense3')
					pi = tf.layers.dense(pi_dense3, self.args.acts_dims[0], activation=tf.nn.tanh, name='pi')
				return pi

			def mlp_value(obs_ph, acts_ph):
				state_ph = tf.concat([obs_ph, acts_ph], axis=1)
				with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
					q_dense1 = tf.layers.dense(state_ph, 256, activation=tf.nn.relu, name='q_dense1')
					q_dense2 = tf.layers.dense(q_dense1, 256, activation=tf.nn.relu, name='q_dense2')
					q_dense3 = tf.layers.dense(q_dense2, 256, activation=tf.nn.relu, name='q_dense3')
					q = tf.layers.dense(q_dense3, 1, name='q')
				return q

			with tf.variable_scope('main'):
				with tf.variable_scope('policy'):
					self.pi = mlp_policy(self.obs_ph)
				with tf.variable_scope('value'):
					self.q = mlp_value(self.obs_ph, self.acts_ph)
				with tf.variable_scope('value', reuse=True):
					self.q_pi = mlp_value(self.obs_ph, self.pi)

			with tf.variable_scope('target'):
				with tf.variable_scope('policy'):
					self.pi_t = mlp_policy(self.obs_next_ph)
				with tf.variable_scope('value'):
					self.q_t = mlp_value(self.obs_next_ph, self.pi_t)

		def create_operators():
			self.pi_q_loss = -tf.reduce_mean(self.q_pi)
			self.pi_l2_loss = self.args.act_l2*tf.reduce_mean(tf.square(self.pi))
			self.pi_optimizer = tf.train.AdamOptimizer(self.args.pi_lr)
			self.pi_train_op = self.pi_optimizer.minimize(self.pi_q_loss+self.pi_l2_loss, var_list=get_vars('main/policy'))

			if self.args.clip_return:
				return_value = tf.clip_by_value(self.q_t, self.args.clip_return_l, self.args.clip_return_r)
			else:
				return_value = self.q_t
			target = tf.stop_gradient(self.rews_ph+self.args.gamma*return_value)
			self.q_loss = tf.reduce_mean(tf.square(self.q-target))
			self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr)
			self.q_train_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars('main/value'))

			self.target_update_op = tf.group([
				v_t.assign(self.args.polyak*v_t + (1.0-self.args.polyak)*v)
				for v, v_t in zip(get_vars('main'), get_vars('target'))
			])

			self.saver=tf.train.Saver()
			self.init_op = tf.global_variables_initializer()
			self.target_init_op = tf.group([
				v_t.assign(v)
				for v, v_t in zip(get_vars('main'), get_vars('target'))
			])

		self.graph = tf.Graph()
		with self.graph.as_default():
			create_session()
			create_inputs()
			create_normalizer()
			create_network()
			create_operators()
		self.init_network()

	def init_network(self):
		self.sess.run(self.init_op)
		self.sess.run(self.target_init_op)

	def step(self, obs, explore=False, test_info=False):
		if (not test_info) and (self.args.buffer.steps_counter<self.args.warmup):
			return np.random.uniform(-1, 1, size=self.args.acts_dims)
		if self.args.goal_based: obs = goal_based_process(obs)

		# eps-greedy exploration
		if explore and np.random.uniform()<=self.args.eps_act:
			return np.random.uniform(-1, 1, size=self.args.acts_dims)

		feed_dict = {
			self.raw_obs_ph: [obs]
		}
		action, info = self.sess.run([self.pi, self.step_info], feed_dict)
		action = action[0]

		# uncorrelated gaussian explorarion
		if explore: action += np.random.normal(0, self.args.std_act, size=self.args.acts_dims)
		action = np.clip(action, -1, 1)

		if test_info: return action, info
		return action

	def step_batch(self, obs):
		actions = self.sess.run(self.pi, {self.raw_obs_ph:obs})
		return actions

	def feed_dict(self, batch):
		return {
			self.raw_obs_ph: batch['obs'],
			self.raw_obs_next_ph: batch['obs_next'],
			self.acts_ph: batch['acts'],
			self.rews_ph: batch['rews']
		}

	def train(self, batch):
		feed_dict = self.feed_dict(batch)
		info, _, _ = self.sess.run([self.train_info, self.pi_train_op, self.q_train_op], feed_dict)
		self.tmp_torch.train(batch)
		print("torch:", self.tmp_torch.pi_q_loss)
		print("TF:", info['Pi_q_loss'] )
		print("torch:", self.tmp_torch.pi_l2_loss)
		print("TF:", info['Pi_l2_loss'] )
		print("torch:", self.tmp_torch.q_loss)
		print("TF:", info['Q_loss'])
		return info

	def train_pi(self, batch):
		feed_dict = self.feed_dict(batch)
		info, _ = self.sess.run([self.train_info_pi, self.pi_train_op], feed_dict)
		return info

	def train_q(self, batch):
		feed_dict = self.feed_dict(batch)
		info, _ = self.sess.run([self.train_info_q, self.q_train_op], feed_dict)
		return info

	def normalizer_update(self, batch):
		self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))
		self.tmp_torch.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

	def target_update(self):
		self.sess.run(self.target_update_op)
		self.tmp_torch.target_update()
