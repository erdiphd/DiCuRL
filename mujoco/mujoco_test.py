import gym
import threading
import numpy as np
from envs import make_env, clip_return_range, Robotics_envs_id
from utils.os_utils import get_arg_parser, get_logger, str2bool
from learner import create_learner, learner_collection
def get_args():
    parser = get_arg_parser()

    parser.add_argument('--tag', help='terminal tag in logger', type=str, default='')
    parser.add_argument('--alg', help='backend algorithm', type=str, default='ddpg', choices=['ddpg', 'ddpg2'])
    parser.add_argument('--learn', help='type of training method', type=str, default='dt-her',
                        choices=learner_collection.keys())

    parser.add_argument('--env', help='gym env id', type=str, default='FetchPickAndPlace-v1', choices=Robotics_envs_id)
    args, _ = parser.parse_known_args()
    if args.env == 'HandReach-v0':
        parser.add_argument('--goal', help='method of goal generation', type=str, default='reach',
                            choices=['vanilla', 'reach'])
    else:
        parser.add_argument('--goal', help='method of goal generation', type=str, default='interval',
                            choices=['vanilla', 'fixobj', 'interval', 'obstacle'])
        if args.env[:5] == 'Fetch':
            parser.add_argument('--init_offset', help='initial offset in fetch environments', type=np.float32,
                                default=1.0)
        elif args.env[:4] == 'Hand':
            parser.add_argument('--init_rotation', help='initial rotation in hand environments', type=np.float32,
                                default=0.25)

    parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.98)
    parser.add_argument('--clip_return', help='whether to clip return value', type=str2bool, default=True)
    parser.add_argument('--eps_act', help='percentage of epsilon greedy explorarion', type=np.float32, default=0.3)
    parser.add_argument('--std_act', help='standard deviation of uncorrelated gaussian explorarion', type=np.float32,
                        default=0.2)

    parser.add_argument('--pi_lr', help='learning rate of policy network', type=np.float32, default=1e-3)
    parser.add_argument('--q_lr', help='learning rate of value network', type=np.float32, default=1e-3)
    parser.add_argument('--act_l2', help='quadratic penalty on actions', type=np.float32, default=1.0)
    parser.add_argument('--polyak', help='interpolation factor in polyak averaging for DDPG', type=np.float32,
                        default=0.95)

    parser.add_argument('--epochs', help='number of epochs', type=np.int32, default=1)
    parser.add_argument('--cycles', help='number of cycles per epoch', type=np.int32, default=1)
    parser.add_argument('--episodes', help='number of episodes per cycle', type=np.int32, default=2)
    parser.add_argument('--timesteps', help='number of timesteps per episode', type=np.int32,
                        default=(50 if args.env[:5] == 'Fetch' else 100))
    parser.add_argument('--train_batches', help='number of batches to train per episode', type=np.int32, default=20)

    parser.add_argument('--buffer_size', help='number of episodes in replay buffer', type=np.int32, default=10000)
    parser.add_argument('--buffer_type', help='type of replay buffer / whether to use Energy-Based Prioritization',
                        type=str, default='energy', choices=['normal', 'energy'])
    parser.add_argument('--batch_size', help='size of sample batch', type=np.int32, default=256)
    parser.add_argument('--warmup', help='number of timesteps for buffer warmup', type=np.int32, default=10000)
    parser.add_argument('--her', help='type of hindsight experience replay', type=str, default='future',
                        choices=['none', 'final', 'future'])
    parser.add_argument('--her_ratio', help='ratio of hindsight experience replay', type=np.float32, default=0.8)
    parser.add_argument('--pool_rule', help='rule of collecting achieved states', type=str, default='full',
                        choices=['full', 'final'])

    parser.add_argument('--hgg_c', help='weight of initial distribution in flow learner', type=np.float32, default=3.0)
    parser.add_argument('--hgg_L', help='Lipschitz constant', type=np.float32, default=5.0)
    parser.add_argument('--hgg_pool_size', help='size of achieved trajectories pool', type=np.int32, default=1000)
    parser.add_argument('--forced_hgg_dt_step_size', help='step size between intermediate goals', type=np.float32,
                        default=None)

    args = parser.parse_args()
    return args

args = get_args()
env = make_env(args)

def continuous_run():
    while True:
        env.render()


sim_thread = threading.Thread(target=continuous_run)
sim_thread.start()


obs = env.reset()



print(gym.__file__)

counter = 0
obs = env.reset()
action_test = [0,0,0,1]
while True:
    action_test = env.action_space.sample()
    tmp = env.step(action_test)
    obs = env.reset()

