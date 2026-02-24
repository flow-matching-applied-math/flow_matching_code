
import random
import torch

def set_seed(seed: int):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

