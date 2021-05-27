import os
import torch

def savemodel(model, path: str) -> None:

	dirname = os.path.dirname(path)
	os.makedirs(dirname, exist_ok=True)
	torch.save(model.state_dict(),path)


def loadmodel(model, path: str) -> None:
	model.load_state_dict(torch.load(path))

