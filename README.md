# Curriculum Learning Strategies Comparison - Final Year Project

This repository contains the source code for a set of experiments which compare the performance of 6 curriculum learning strategies on a variety of recurrent neural network architectures on a variety of tasks.

This work was conducted as part of my undergraduate final year project, which mostly focussed on the [application of memory augmented neural networks to machine translation](https://github.com/MarkPKCollier/NMTFYP).

## Repo Guide

- ```/``` scripts for running experiments and analyzing their results.
- ```/img``` graphs of experimental results.
- ```/tasks/ntm.py``` implementation of a [Neural Turing Machine](https://github.com/MarkPKCollier/NeuralTuringMachine).
- ```/tasks/dnc.py /tasks/access.py /tasks/addressing.py``` impmentation of a [Differentiable Neural Computer](https://www.nature.com/articles/nature20101.pdf) from [here](https://github.com/deepmind/dnc).
- ```/tasks/sonnet_``` DeepMind's [Sonnet library](https://github.com/deepmind/sonnet), I needed to make modifications to the library so its copied here.
- ```/tasks/exp3s.py``` implementation of a [Exp3.S algorithm](http://epubs.siam.org/doi/abs/10.1137/S0097539701398375) to maximise reward from a N-armed bandit.
- ```/tasks/generate_data.py``` all experiments have sythetic data, this script produces batches of data for each experiment.
- ```/tasks/run_tasks.py``` script that will train the desired network on the desired task (see script for arguments it takes).

## Curriculum Learning Results

There has been little empirical comparison of different curricula for curriculum learning in the literature. I compare 4 curricula to each other and 2 benchmark non-curriculum learning strategies on the Copy task and Associative Recall task with a LSTM network and on the Shortest Path and Traversal tasks from the DNC paper with a NTM and a DNC network.

For each hand-designed curricula I define a success metric, which once attained by the network, it can move onto the next lesson in the curriculum.

The curricula and benchmark data sampling strategies are:

- ```Naive```: all examples are drawn from the current lesson. [Based on Bengio - Curriculum Learning](https://dl.acm.org/citation.cfm?id=1553380).
- ```Look Back```: 90% of examples are drawn from the current lesson, 10% are drawn uniformly from previous lessons. [Based on DNC paper](https://www.nature.com/articles/nature20101.pdf).
- ```Look Back and Forward```: 80% of examples are drawn from the current lesson, 20% are drawn uniformly from all lessons.
- ```Predictive Gain```: an automated strategy which uses the Exp3.S algorithm to maximize a reward defined by the difference in loss before and after seeing an example. See: [Automated Curriculum Learning for Neural Networks](https://arxiv.org/pdf/1704.03003.pdf).
- ```None```: benchmark non curriculum learning strategy of only training on the target task.
- ```Uniform```: benchmark strategy of training on examples sampled uniformly from all lessons.

In the below graphs "Target task setting" refers to the loss/error on the target task e.g. sequences of length 20 on the Copy task. "Multi-task setting" refers to the loss/error on examples sampled uniformly from all lessons.

As the below graphs demonstrate the optimal curriculum is task dependent, but curriculum learning does in general provide an advantage over the benchmark strategies. Additionally Predictive Gain and Look Back and Forward perform well accross all tasks and settings, so I recommend using one of these in preference to the other strategies if using curriculum learning.

#### Average performance (over 3 training runs) on validation set of LSTM on Copy Task in the target task setting of curricula:
![Average performance (over 3 training runs) on validation set of LSTM on Copy Task in the target task setting of curricula](/img/copy_task_lstm_curricula_comparison_target_setting.png)

#### Average performance (over 3 training runs) on validation set of LSTM on Copy Task in the multi-task setting of curricula:
![Average performance (over 3 training runs) on validation set of LSTM on Copy Task in the multi-task setting of curricula](/img/copy_task_lstm_curricula_comparison_multi_setting.png)

#### Average performance (over 3 training runs) on validation set of LSTM on Associative Recall Task in the target task setting of curricula:
![Average performance (over 3 training runs) on validation set of LSTM on Associative Recall Task in the target task setting of curricula](/img/associative_recall_task_lstm_curricula_comparison_target_setting.png)

#### Average performance (over 3 training runs) on validation set of LSTM on Associative Recall Task in the multi-task setting of curricula:
![Average performance (over 3 training runs) on validation set of LSTM on Associative Recall Task in the multi-task setting of curricula](/img/associative_recall_task_lstm_curricula_comparison_multi_setting.png)

