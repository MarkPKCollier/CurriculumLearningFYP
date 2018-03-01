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

This implementation is based on: https://github.com/snowkylin/ntm but contains some substantial modifications. Most importantly, I backpropagate through the initialization of the memory contents and find this works much better than constant or random memory initialization. Additionally the NTMCell implements the [Tensorflow RNNCell interface](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/RNNCell) so can be used directly with [tf.nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn), etc. I never see loss go to NaN as some other implementations report (although convergence is slow and unstable on some training runs).

I replicated the Copy and Associative Recall tasks from the [original paper](https://arxiv.org/abs/1410.5401) and their associated hyperparmeters:

- Memory Size: 128 X 20
- Controller: LSTM - 100 units
- Optimizer: RMSProp - learning rate = 10^-4

The Copy task network was trained on sequences of length sampled from Uniform(1,20) with 8-dimensional random bit vectors. The Associative Recall task network was trained on sequences with the number of items sampled from Uniform(2,6) each item consisted of 3 6-dimensional random bit vectors.

#### Example performance of NTM on Copy task with sequence length = 20 (output is perfect):
![Neural Turing Machine Copy Task - Seq len=20](/img/copy_ntm_20_0.png)

#### Example performance of NTM on Copy task with sequence length = 40 (network only trained on sequences of length up to 20 - performance degrades on example after 36th input):
![Neural Turing Machine Copy Task - Seq len=40](/img/copy_ntm_40_1.png)

#### Example performance of NTM on Associative Recall task with 6 items (output is perfect):
![Neural Turing Machine Associate Recall Task - Seq len=6 items](/img/associative_recall_ntm_6_0.png)

#### Example performance of NTM on Associative Recall task with 12 items (despite only being trained on sequences of up to 6 items to network generalizes perfectly to 12 items):
![Neural Turing Machine Associate Recall Task - Seq len=12 items](/img/associative_recall_ntm_12_0.png)

In order to interpret how the NTM used its external memory I trained a network with 32 memory locations on the Copy task and graphed the read and write head address locations over time.

As you can see from the below graphs, the network first writes the sequence to memory and then reads it back in the same order it wrote it to memory. This uses both the content and location based addressing capabilities of the NTM. The pattern of writes followed by reads is what we would expect of a reasonable solution to the Copy task.

#### Write head locations of NTM with 32 memory locations trained on Copy task:
![Write head locations of NTM with 32 memory locations trained on Copy task](/img/ntm_copy_write_head.png)

#### Read head locations of NTM with 32 memory locations trained on Copy task:
![Read head locations of NTM with 32 memory locations trained on Copy task](/img/ntm_copy_read_head.png)

I also compared the learning curves on 3 training runs (with different random seeds) of my Neural Turing Machine implementation to the [reference Differentiable Neural Computer implementation](https://github.com/deepmind/dnc) and the [Tensorflow BasicLSTM cell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell).

As you can see from the below graphs on 2 out 3 training runs for both the Copy and Associative Recall tasks the NTM performs comparably to the results in the NTM paper (although there is some instability after convergence on one of the training runs). My NTM implementation is slow to converge on 1 out of the 3 training runs - I am unclear as to why this is the case, perhaps it is our implementation e.g. parameter initialization.

#### Comparison of learning curves of NTM, DNC and LSTM on 3 training runs on the Copy task each:
![Comparison of learning curves of NTM, DNC and LSTM on 3 training runs on the Copy task each](/img/copy_task_archiecture_comparison.png)

#### Comparison of learning curves of NTM, DNC and LSTM on 3 training runs on the Associative Recall task each:
![Comparison of learning curves of NTM, DNC and LSTM on 3 training runs on the Associative Recall task each](/img/associative_recall_archiecture_comparison.png)

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

