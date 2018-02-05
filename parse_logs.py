import re
import glob
import itertools

columns = []

for log_file in glob.glob('logs/*.txt'):
    f = open(log_file, 'r').read()
    res = []
    res.append((log_file + '_i', log_file + '_curriculum', log_file + '_loss', log_file + '_bit_errors'))
    for i, curricilum, loss, bit_errors in re.findall('PARSABLE: (\d+),(\d+),(-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?),([-+]?\d*\.\d+|\d+)\n', f):
        res.append((i, curricilum, loss, bit_errors))
    columns.append(res)

print map(len, columns)

f = open('parsed_log_files.csv', 'w')
for t in itertools.izip_longest(*columns, fillvalue=(None, None, None, None)):
    line = ''
    for t_ in t:
        # print t_
        for column in t_:
            line += (str(column) if column is not None else '') + ','
            # print line
    f.write(line[:-1] + '\n')

f.close()