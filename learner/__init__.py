from .normal import NormalLearner
from .diffusion import DiffusionLearner
from .hgg import HGGLearner
learner_collection = {
	'normal': NormalLearner,
	'diffusion': DiffusionLearner,
	'hgg': HGGLearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)