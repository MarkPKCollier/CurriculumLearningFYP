import os
import time

run_cmd = '''gcloud ml-engine jobs submit training {0}_mann_{1}_{10}_init_mode_curric_{2}_pad_{3}_num_{6}_{7} \
--module-name=tasks.run_tasks \
--package-path=/Users/markcollier/Documents/CurriculumLearning/tasks \
--region=us-east1 \
--job-dir=gs://your_google_storeage_bucket/{0}_mann_{1}_curric_{2}_pad_{3}_num_{6}_{7} \
--runtime-version=1.6 \
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
--max_repeats={20} \
--num_bits_per_vector={9} \
--init_mode={10} \
--num_memory_locations={11} \
--memory_size={12} \
--num_read_heads={13} \
--learning_rate={14} \
--batch_size={15} \
--eval_batch_size={16} \
--num_train_steps={17} \
--steps_per_eval={18} \
--random_seed={19} \
--optimizer=Adam
'''

VERSION = 0
NUM_EXPERIMENTAL_REPEATS = 10

# task, max_seq_len, max_repeats, num_bits_per_vector, num_memory_locations, memory_size, num_read_heads, learning_rate, batch_size, eval_batch_size, num_train_steps, steps_per_eval = ('copy', 32, -1, 8, 128, 20, 1, 0.001, 32, 1024, 62500, 200)
task, max_seq_len, max_repeats, num_bits_per_vector, num_memory_locations, memory_size, num_read_heads, learning_rate, batch_size, eval_batch_size, num_train_steps, steps_per_eval = ('repeat_copy', 13, 13, 8, 128, 20, 1, 0.001, 32, 512, 46900, 200)
# task, max_seq_len, max_repeats, num_bits_per_vector, num_memory_locations, memory_size, num_read_heads, learning_rate, batch_size, eval_batch_size, num_train_steps, steps_per_eval = ('associative_recall', 12, -1, 6, 128, 20, 1, 0.001, 32, 352, 156250, 200)
# task, max_seq_len, max_repeats, num_bits_per_vector, num_memory_locations, memory_size, num_read_heads, learning_rate, batch_size, eval_batch_size, num_train_steps, steps_per_eval = ('traversal', -1, -1, 90, 256, 50, 5, 0.00001, 2, 400, 500000, 3200)

stream_cmds = []

# manns = [('ntm', 100, 1), ('dnc', 100, 1), ('none', 256, 3)]
# manns = [('ntm', 100, 1)]
# manns = [('dnc', 100, 1)]
manns = [('none', 256, 3)]
# manns = [('dnc', 256, 3)]
# curricula = ['none']
# curricula = ['prediction_gain_bandit']
curricula = ['uniform', 'none', 'naive', 'prediction_gain_bandit', 'look_back', 'look_back_and_forward']

for experiment_number in range(NUM_EXPERIMENTAL_REPEATS):
    for mann, num_units, num_layers in manns:
        for curriculum in curricula:
            for pad_to_max_seq_len in [False]:
                for init_mode in ['constant']:
                    os.system(
                        run_cmd.format(
                            task, mann, curriculum, pad_to_max_seq_len, num_units, num_layers, VERSION, experiment_number+1, max_seq_len, num_bits_per_vector, init_mode, num_memory_locations, memory_size, num_read_heads, learning_rate, batch_size, eval_batch_size, num_train_steps, steps_per_eval, 1000*(experiment_number+1), max_repeats))
                    experiment_name = '{0}_mann_{1}_{6}_init_mode_curric_{2}_pad_{3}_num_{4}_{5}'.format(task, mann, curriculum, pad_to_max_seq_len, VERSION, experiment_number+1, init_mode)
                    stream_cmds.append('gcloud ml-engine jobs stream-logs {0} > ../logs/{1}.txt'.format(experiment_name, experiment_name))
                    time.sleep(15)

for cmd in stream_cmds:
    print cmd + '\n'
