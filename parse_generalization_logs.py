import re
import glob
import itertools

NUM_EXP = 6
NUM_REPEATS = 10
NUM_GENERALIZATION_POINTS = 4
target_res = [[] for _ in range(NUM_GENERALIZATION_POINTS)]
multi_res = [[] for _ in range(NUM_GENERALIZATION_POINTS)]
num_re = '([-+]?\d*\.\d+|\d+)'

log_dir = 'logs'

for log_file in sorted(glob.glob('{0}/*.txt'.format(log_dir))):
    f = open(log_file, 'r').read()

    try:
        generalization_from_target_task = re.search('generalization_from_target_task: {0}'.format(','.join([num_re for _ in range(NUM_GENERALIZATION_POINTS)])), f).groups()
        generalization_from_multi_task = re.search('generalization_from_multi_task: {0}'.format(','.join([num_re for _ in range(NUM_GENERALIZATION_POINTS)])), f).groups()
    except AttributeError:
        generalization_from_target_task = generalization_from_multi_task = ['None' for _ in range(NUM_GENERALIZATION_POINTS)]

    for i in range(NUM_GENERALIZATION_POINTS):
        target_res[i].append(generalization_from_target_task[i])
        multi_res[i].append(generalization_from_multi_task[i])

print '\n'.join(sorted(glob.glob('{0}/*.txt'.format(log_dir))))
for i in range(NUM_GENERALIZATION_POINTS):
    print '-' * 50
    for j in range(NUM_EXP):
        print '\n'.join(target_res[i][j*NUM_REPEATS:(j+1)*NUM_REPEATS])
        print '\n'
    print '-' * 50

for i in range(NUM_GENERALIZATION_POINTS):
    print '-' * 50
    for j in range(NUM_EXP):
        print '\n'.join(multi_res[i][j*NUM_REPEATS:(j+1)*NUM_REPEATS])
        print '\n'
    print '-' * 50
