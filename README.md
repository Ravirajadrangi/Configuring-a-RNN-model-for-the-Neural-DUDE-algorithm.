# RNN Neural DUDE

This project was undertaken for the Stanford EE 378A 2017 Spring quarter project by Charles Hale and David Wugofski. It is an extension of the [Neural DUDE](https://arxiv.org/abs/1605.07779) algorithm created by Taesup Moon, Seonwoo Min, Byunghan Lee, and Sungroh Yoon.

## Purpose

This project revolves around applying recurrent neural networks to the Neural DUDE algorithm for discrete universal denoising. Unlike the original Neural DUDE algorithm, the RNN Neural DUDE algorithm is capable of resolving performance for a variety of context sizes in one training model, allowing more efficient determination of optimal context size for denoising.

## Relevant Files

### IPython Notebooks

This project contains three relevant IPython Notebooks for code execution. First there is [NeuralDUDE_v4.ipynb](NeuralDUDE_v4.ipynb), which contains the original code borrowed with permission from Professor Moon. This file contains the original code used to create, train, and evaluate the Neural DUDE algorithm for a variety of test images. Next there is [NueralDUDE_v4.ipynb](NueralDUDE_wrapper.ipynb), which streamlines the process of creating, training, and evaluating the Neural DUDE algorithm, and is mostly revolved around displaying the results. Lastly there is [NeuralDUDE_VariableRNN.ipynb](NeuralDUDE_VariableRNN.ipynb) which includes the code for creating, training, and evaluating the new RNN approach to the Neural DUDE algorithm.

### Python Scripts

[binary_dude.py](binary_dude.py): The python script borrowed with permission from Professor Moon to work with noise generation, error evaluation, and running the original DUDE algorithm.

[architectures.py](architectures.py): The python script written to facilitate selecting models to try out for our neural nets.

[ndude_sim.py](ndude_sim.py): A script containing the code to configure, train, and evaluate a neural net in one function call, to help facilitate streamlined evaluation.
