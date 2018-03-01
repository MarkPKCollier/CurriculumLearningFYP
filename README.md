# Curriculum Learning Strategies Comparison - Final Year Project

This repository contains the source code for a set of experiments which compare the performance of 6 curriculum learning strategies on a variety of recurrent neural network architectures on a variety of tasks.

This work was conducted as part of my undergraduate final year project, which mostly focussed on the [application of memory augmented neural networks to machine translation](https://github.com/MarkPKCollier/NMTFYP).

- [Repo Guide](#repo-guide)
- [Neural Turing Machine Implementation](#neural-turing-machine-implementation)
- [Curriculum Learning Results](#curriculum-learning-results)

## Repo Guide

- ```/``` scripts for running experiments and analyzing their results.
- ```/img``` graphs of experimental results.
- ```/tasks/ntm.py``` implementation of a [Neural Turing Machine](https://arxiv.org/abs/1410.5401).
- ```/tasks/dnc.py /tasks/access.py /tasks/addressing.py``` impmentation of a [Differentiable Neural Computer](https://www.nature.com/articles/nature20101.pdf) from [here](https://github.com/deepmind/dnc).
- ```/tasks/sonnet_``` DeepMind's [Sonnet library](https://github.com/deepmind/sonnet), I needed to make modifications to the library so its copied here.
- ```/tasks/exp3s.py``` implementation of a [Exp3.S algorithm](http://epubs.siam.org/doi/abs/10.1137/S0097539701398375) to maximise reward from a N-armed bandit.
- ```/tasks/generate_data.py``` all experiments have sythetic data, this script produces batches of data for each experiment.
- ```/tasks/run_tasks.py``` script that will train the desired network on the desired task (see script for arguments it takes).

## Neural Turing Machine Implementation

One architecture considered as part of the [curriculum learning experiments](#curriculum-learning-results) was a [Neural Turing Machine](https://arxiv.org/abs/1410.5401). Thus this repository contains a Tensorflow implementation of a Neural Turing Machine and the Copy and Associated Recall tasks from the [original paper](https://arxiv.org/abs/1410.5401).

This implementation is based on: https://github.com/snowkylin/ntm but contains some substantial modifications. Most importantly, I backpropagate through the initialization of the memory contents and find this works much better than constant or random memory initialization. Additionally the NTMCell implements to [Tensorflow RNNCell interface](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/RNNCell) so can be used directly with [tf.nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn), etc.

## Curriculum Learning Results

pass

