# NeReGraIn: Neural Regular Grammar Induction
This repository contains PyTorch code and results for the model introduced in the paper "A Neural Model for Regular Grammar Induction" by Peter Belcak, David Hofer, and Roger Wattenhofer.

The long version of the IEEE ICMLA 2022 paper can be found in the root directory.
Everything else is the code for the initial model.
The up-to-date model variants and testing setup will be added by David Hofer.

## Dependencies
Python 3.9 and PyTorch 1.11

On the side, WandB is used at times to log losses.

## Running the code
Just run `main.py` while in the project directory.
Running it will execute an example training loop, with production matrices output at the end.
