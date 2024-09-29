from common import get_args,experiment_setup
from algorithm.replay_buffer import goal_based_process
import torch
if __name__=='__main__':
	args = get_args()
	env, env_test, agent, buffer, learner, tester, diffusion_model, aim_discriminator = experiment_setup(args)
	agent.pi.load_state_dict(torch.load('/media/xraydisk/cilocharching_machines/next_to_me/log/policy66.pth'))
	agent.obs_normalizer.mean = torch.load('/media/xraydisk/cilocharching_machines/next_to_me/log/mean66.pth')
	agent.obs_normalizer.std = torch.load('/media/xraydisk/cilocharching_machines/next_to_me/log/std66.pth')
	test_rollouts = 10
	acc_sum, obs = 0.0, []
	for i in range(test_rollouts):
		obs.append(goal_based_process(env.reset()))
		for timestep in range(args.timesteps):
			actions = agent.step_batch(obs)
			obs, infos = [], []
			print(actions[0])
			ob, _, _, info = env.step(actions[0])
			obs.append(goal_based_process(ob))
			infos.append(info)
			env.render()