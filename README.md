# Curriculum Learning Syllabus Comparison

This repository contains the source code for a set of experiments which compare the performance of 6 syllabuses for curriculum learning.

We conduct experiments comparing the syllabuses effect on speed of learning and generalization ability on the Copy, Repeat Copy and Associative Recall problems proposed in the Neural Turing Machines Paper.

## Repo Guide

- ```/``` scripts for running experiments and analyzing their results.
- ```/tasks/exp3s.py``` implementation of a [Exp3.S algorithm](http://epubs.siam.org/doi/abs/10.1137/S0097539701398375) to maximise reward from an adversarial stochastic multi-armed bandit.
- ```/tasks/generate_data.py``` all experiments have sythetic data, this script produces batches of data for each experiment.
- ```/tasks/run_tasks.py``` script that will train the desired network on the desired task (see script for arguments it takes).
- ```/tasks/run_experiments.py``` if you use Google ML Engine this script will run all the desired experiments.
