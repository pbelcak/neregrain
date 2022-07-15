import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import random

class RegularGrammarDataset(torch.utils.data.Dataset):
	def __init__(self, x, y, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.x = x
		self.y = y

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		sample = [ self.x[idx], self.y[idx] ]
		return sample

	def __len__(self):
		return len(self.x)

# the following function generates `datapoint_count` positive and negative strings of length `input_length`
# for (and against - for negative pairs) the grammar S -> A, A -> aA, A -> bA, A -> aC, A -> bC, C -> cC, C -> c
def make_data(datapoint_count, input_length):
	datapoint_x_list = []
	datapoint_y_list = []

	datapoint_index = 0
	while datapoint_index < datapoint_count:
		characters = np.zeros(shape=(input_length, 4))
		target_length = random.choice(range(2, input_length+1))

		is_positive = random.choice([0, 1])
		if is_positive:
			state_switch_index = random.choice(range(1, target_length))
			for i in range(0, state_switch_index):
				char_choice = random.choice(range(0, 2))
				characters[i, char_choice] = 1

			for i in range(state_switch_index, target_length):
				characters[i, 2] = 1
		else:
			for i in range(0, target_length):
				char_choice = random.choice(range(0, 3))
				characters[i, char_choice] = 1

			# check if the string meant to be negative actually isnt a positive sample
			has_transitioned_to_2 = False
			for i in range(0, target_length):
				if has_transitioned_to_2 and characters[i, 2] == 0:
					is_positive = 0
					break
				elif characters[i, 2] == 1:
					has_transitioned_to_2 = True
			else:
				if not has_transitioned_to_2:
					is_positive = 0
				else:
					is_positive = 1

		for i in range(target_length, input_length):
			characters[i, 3] = 1

		characters = characters.astype(np.float64, copy=False)
		y = np.array([ is_positive*1.0 ])
		datapoint_x_list.append(torch.from_numpy(characters))
		datapoint_y_list.append(torch.from_numpy(y))
		datapoint_index += 1

	return RegularGrammarDataset(datapoint_x_list, datapoint_y_list)
