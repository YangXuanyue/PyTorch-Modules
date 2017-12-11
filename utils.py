from torch_imports import *


def softmax(x):
	return torch.exp(F.log_softmax(x))