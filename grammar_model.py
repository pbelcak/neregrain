import torch
from torch import nn

class GrammarModel(nn.Module):
	def __init__(self, input_length, n_terminals, n_nonterminals):
		super(GrammarModel, self).__init__()
		self.input_length = input_length
		self.n_terminals = n_terminals
		self.n_nonterminals = n_nonterminals
		
		self.terminal_productions = nn.Parameter(torch.randn(n_nonterminals, n_terminals), requires_grad=True)
		self.nonterminal_productions = nn.Parameter(torch.randn(n_nonterminals, n_nonterminals, n_terminals), requires_grad=True)
		
	def forward(self, input_tensor):
		# input_tensor is a batch of sentences (batch_size, input_length, n_terminals),
		#	where each sentence is given as a vector of one-hot encodings of the terminal symbols
		batch_size = input_tensor.size(0)

		terminal_production_chooser = torch.sigmoid(self.terminal_productions)			# (n_nonterminals, n_terminals)
		nonterminal_production_choosers = torch.sigmoid(self.nonterminal_productions)	# (n_nonterminals, n_nonterminals, n_terminals)
		outputs = torch.zeros(size=(self.input_length, batch_size), device=input_tensor.device)	# (input_length, batch_size)

		terminal_production_choices = torch.matmul(terminal_production_chooser, input_tensor[:, 0].T).T	# (batch_size, n_nonterminals)
		outputs[0] = terminal_production_choices[:, 0]	

		previous_choices = terminal_production_choices.unsqueeze(2)	# (batch_size, n_nonterminals, 1)
		for i in range(1, self.input_length):
			batched_choosers = nonterminal_production_choosers.unsqueeze(0) # (1, n_nonterminals, n_nonterminals, n_terminals)
			batch_of_current_inputs = input_tensor[:, i].unsqueeze(2).unsqueeze(1) # (batch_size, 1, n_terminals, 1)
			nonterminal_production_choices = torch.matmul(batched_choosers, batch_of_current_inputs) # (batch_size, n_nonterminals, n_nonterminals, 1)
			nonterminal_production_choices = nonterminal_production_choices.squeeze(3) # (batch_size, n_nonterminals, n_nonterminals)
			nonterminal_production_choices = torch.clamp(nonterminal_production_choices, min=0.0, max=1.0)
			
			current_choices_unsqueezed = torch.matmul(nonterminal_production_choices, previous_choices) # (batch_size, n_nonterminals, 1)
			current_choices = current_choices_unsqueezed.squeeze(2) # (batch_size, n_nonterminals)
			current_choices = torch.clamp(current_choices, min=0.0, max=1.0)

			carry_over_masks = input_tensor[:, i, self.n_terminals-1] # (batch_size, input_length)
			carry_over = outputs[i-1] * carry_over_masks 
			new = current_choices[:, 0] * (1 - carry_over_masks)
			outputs[i] = carry_over + new

			previous_choices = current_choices_unsqueezed
		
		return outputs[-1, :].unsqueeze(1)	# (batch_size,)

	def get_terminal_productions(self):
		return torch.sigmoid(self.terminal_productions)

	def get_nonterminal_productions(self):
		return torch.sigmoid(self.nonterminal_productions)