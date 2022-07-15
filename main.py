import torch
from torch import nn
import numpy as np
import wandb
import time

from grammar_model import GrammarModel
import curriculum
import data

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

input_length = 16
terminal_alphabet_size = 3+1 # +1 for the end of string character
nonterminal_alphabet_size = 3

example_training_data = data.make_data(4000, input_length)
example_training_loader = torch.utils.data.DataLoader(example_training_data, batch_size=80, shuffle=True)

wandb.init(
	name=str(int(time.time())),
	project="disgra"
)

# construct the model
model = GrammarModel(
	input_length=input_length,
	n_terminals = terminal_alphabet_size,
	n_nonterminals=nonterminal_alphabet_size
)

# train&test the model
epochs = 60
for t in range(epochs):
	print(f"Epoch {t+1}\n-------------------------------")
	loss_total, loss_reconstruction, loss_terminal_sharpening, loss_nonterminal_sharpening, loss_terminal_use, loss_nonterminal_use \
		= curriculum.train(model, example_training_loader, beta=(0.01 * (t*1.0/epochs-0.4)) if t*1.0 / epochs > 0.40 else 0.0, gamma=0.005)
	# curriculum.test(model, example_test_loader)
print("Done!")

# print results on the internals of the model
print(model.get_terminal_productions().round(decimals=2))
print(model.get_nonterminal_productions().round(decimals=2))