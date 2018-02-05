import matplotlib
matplotlib.use('macosx')
import pickle
import seaborn
import matplotlib.pyplot as plt

NAME = 'default_init_no_curriculum_2'
HEAD_LOG_FILE = 'head_logs/{0}.p'.format(NAME)

outputs = pickle.load(open(HEAD_LOG_FILE, "rb"))

last_output = outputs[-6]

plt.imshow(last_output['labels'], cmap='gray', interpolation='nearest')
plt.savefig('head_logs/{0}_labels'.format(NAME))
plt.show()

plt.imshow(last_output['outputs'], cmap='gray', interpolation='nearest')
plt.savefig('head_logs/{0}_outputs'.format(NAME))
plt.show()

plt.imshow(last_output['read_head'], cmap='gray', interpolation='nearest')
plt.savefig('head_logs/{0}_read_head'.format(NAME))
plt.show()

plt.imshow(last_output['write_head'], cmap='gray', interpolation='nearest')
plt.savefig('head_logs/{0}_write_head'.format(NAME))
plt.show()