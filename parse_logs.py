import re
import glob
import itertools

columns = []

for log_file in sorted(glob.glob('logs/*.txt')):
    f = open(log_file, 'r').read()
    res = []
    res.append((log_file + '_step', log_file + '_target_task_error', log_file + '_multi_task_error'))
    for i, target_task_error, target_task_loss, multi_task_error, multi_task_loss, curricilum, curriculum_point_error, curriculum_point_loss in re.findall('EVAL_PARSABLE: (\d+),([-+]?\d*\.\d+|\d+),(-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?),([-+]?\d*\.\d+|\d+),(-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?),(\d+),([-+]?\d*\.\d+|\d+|None),(-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?|None)\n', f):
        res.append((i, target_task_error, multi_task_error))
    columns.append(res)

print map(len, columns)

f = open('parsed_log_files.csv', 'w')
for t in itertools.izip_longest(*columns, fillvalue=(None, None, None)):
    line = ''
    for t_ in t:
        for column in t_:
            line += (str(column) if column is not None else '') + ','
    f.write(line[:-1] + '\n')

f.close()