import os
import time

run_cmd = '''gcloud ml-engine jobs submit training {0}_mann_{1}_curric_{2}_pad_{3}_num_{6}_{7} \
--module-name=tasks.run_tasks \
--package-path=/Users/markcollier/Documents/CurriculumLearning/tasks \
--region=us-central1 \
--job-dir=gs://mcolliertfnmt7/{0}_mann_{1}_curric_{2}_pad_{3}_num_{6}_{7} \
--runtime-version=1.2 \
--scale-tier=BASIC_GPU -- \
--experiment_name={0}_mann_{1}_curric_{2}_pad_{3}_num_{6}_{7} \
--task={0} \
--mann={1} \
--curriculum={2} \
--pad_to_max_seq_len={3} \
--num_units={4} \
--num_layers={5} \
--verbose=False \
--max_seq_len={8} \
--num_bits_per_vector={9}
'''

VERSION = 0
NUM_EXPERIMENTAL_REPEATS = 3

# task, max_seq_len, num_bits_per_vector = ('copy', 20, 8)
task, max_seq_len, num_bits_per_vector = ('associative_recall', 6, 6)

stream_cmds = []

# manns = [('ntm', 100, 1), ('dnc', 100, 1), ('none', 256, 3)]
manns = [('ntm', 100, 1)]
# curricula = ['uniform', 'none', 'naive', 'look_back', 'look_back_and_forward', 'prediction_gain']
curricula = ['uniform']

for experiment_number in range(NUM_EXPERIMENTAL_REPEATS):
    for mann, num_units, num_layers in manns:
        for curriculum in curricula:
            for pad_to_max_seq_len in [False]:
                os.system(
                    run_cmd.format(
                        task, mann, curriculum, pad_to_max_seq_len, num_units, num_layers, VERSION, experiment_number+1, max_seq_len, num_bits_per_vector))
                experiment_name = '{0}_mann_{1}_curric_{2}_pad_{3}_num_{4}_{5}'.format(task, mann, curriculum, pad_to_max_seq_len, VERSION, experiment_number+1)
                stream_cmds.append('gcloud ml-engine jobs stream-logs {0} > ../logs/{1}.txt'.format(experiment_name, experiment_name))
                time.sleep(15)

for cmd in stream_cmds:
    print cmd + '\n'
