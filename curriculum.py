import torch
from torch import nn
import numpy as np
import wandb

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, dataloader, beta, gamma):
	optimizer = torch.optim.Adam(model.parameters(), lr=50e-4)

	size = len(dataloader.dataset)
	last_print_point = 0

	model.train()
	for batch, (X, y) in enumerate(dataloader):
		current_point = batch* len(X)
		X, y = X.to(device), y.to(device)

		# Compute prediction error
		pred = model(X)
		loss_total, loss_reconstruction, loss_terminal_sharpening, loss_nonterminal_sharpening, loss_terminal_use, loss_nonterminal_use \
			 = grammar_model_loss(model, pred, y, beta=beta, gamma=gamma)

		wandb.log({
			'loss_total': loss_total,
			'loss_reconstruction': loss_reconstruction,
			'loss_terminal_sharpening': loss_terminal_sharpening,
			'loss_effective_terminal_sharpening': beta*loss_terminal_sharpening,
			'loss_nonterminal_sharpening': loss_nonterminal_sharpening,
			'loss_effective_nonterminal_sharpening': beta*loss_terminal_sharpening,
			'loss_terminal_use': loss_terminal_use,
			'loss_effective_terminal_use': gamma*loss_terminal_use,
			'loss_nonterminal_use': loss_nonterminal_use,
			'loss_effective_nonterminal_use': gamma*loss_nonterminal_use
		})
		
		# Backpropagation
		optimizer.zero_grad()
		loss_total.backward()
		optimizer.step()

		# Print progress at ~10 checkpoints
		if current_point - last_print_point > size//10:
			last_print_point = current_point
			loss_total, current = loss_total.item(), current_point
			print(f"loss: total {loss_total:>7f}, (rec {loss_reconstruction/loss_total:.2f}, terminal {loss_terminal_use/loss_total:.2f}, nonterminal {loss_nonterminal_use/loss_total:.2f})  [{current:>5d}/{size:>5d}]")

	return loss_total, loss_reconstruction, loss_terminal_sharpening, loss_nonterminal_sharpening, loss_terminal_use, loss_nonterminal_use

		
def test(model, dataloader, beta, gamma):
	num_batches = len(dataloader)
	model.eval()

	test_total, test_reconstruction, test_terminal_use, test_nonterminal_use = 0, 0, 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			loss_total, loss_reconstruction, loss_terminal_use, loss_nonterminal_use \
				= grammar_model_loss(model, pred, y, beta=beta, gamma=gamma)
			test_total += loss_total
			test_reconstruction += loss_reconstruction
			test_terminal_use += loss_terminal_use
			test_nonterminal_use += loss_nonterminal_use

	test_total /= num_batches
	test_reconstruction /= num_batches
	test_terminal_use /= num_batches
	test_nonterminal_use /= num_batches

	print(f"Test: \n Avg total loss: {test_total:>8f}, rec {test_reconstruction:>8f}, eye {test_terminal_use:>8f}, ds {test_nonterminal_use:>8f} \n")

	return test_total

def grammar_model_loss(model, y_predicted, y_true, beta, gamma):
	reconstruction_loss = nn.BCELoss()(y_predicted, y_true)
	terminal_use_loss = torch.sum(model.get_terminal_productions())
	nonterminal_use_loss = torch.sum(model.get_nonterminal_productions())
	terminal_sharpening_loss = torch.sum(1 - torch.square(2 * model.get_terminal_productions() - 1))
	nonterminal_sharpening_loss = torch.sum(1 - torch.square(2 * model.get_nonterminal_productions() - 1))

	return reconstruction_loss + beta * (terminal_sharpening_loss + nonterminal_sharpening_loss) + gamma * (terminal_use_loss + nonterminal_use_loss), \
		reconstruction_loss, terminal_sharpening_loss, nonterminal_sharpening_loss, terminal_use_loss, nonterminal_use_loss