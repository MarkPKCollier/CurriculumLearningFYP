import matplotlib
matplotlib.use('macosx')
import pickle
import seaborn
import matplotlib.pyplot as plt

EXPERIMENT_NAME = 'associative_recall_ntm'
MANN = 'NTM'
TASK = 'Copy Task'

HEAD_LOG_FILE = 'head_logs/{0}.p'.format(EXPERIMENT_NAME)
GENERALIZATION_HEAD_LOG_FILE = 'head_logs/generalization_{0}.p'.format(EXPERIMENT_NAME)

outputs = pickle.load(open(HEAD_LOG_FILE, "rb"))
outputs.update(pickle.load(open(GENERALIZATION_HEAD_LOG_FILE, "rb")))

for seq_len, heat_maps_list in outputs.iteritems():
    for step, heat_maps in enumerate(heat_maps_list[-10:] if len(heat_maps_list) >= 10 else heat_maps_list):
        plt.imshow(heat_maps['inputs'].T, cmap='gray', interpolation='nearest')
        plt.xlabel('Time ------->')
        plt.title('{0} - {1} - Inputs (Sequence length: {2})'.format(MANN, TASK, seq_len))
        plt.savefig('head_logs/img/{0}_{1}_{2}_inputs'.format(EXPERIMENT_NAME, seq_len, step))
        # plt.show()

        plt.imshow(heat_maps['labels'].T, cmap='gray', interpolation='nearest')
        plt.xlabel('Time ------->')
        plt.title('{0} - {1} - Correct Outputs (Sequence length: {2})'.format(MANN, TASK, seq_len))
        plt.savefig('head_logs/img/{0}_{1}_{2}_labels'.format(EXPERIMENT_NAME, seq_len, step))
        # plt.show()

        plt.imshow(heat_maps['outputs'].T, cmap='gray', interpolation='nearest')
        plt.xlabel('Time ------->')
        plt.title('{0} - {1} - Predicted Outputs (Sequence length: {2})'.format(MANN, TASK, seq_len))
        plt.savefig('head_logs/img/{0}_{1}_{2}_outputs'.format(EXPERIMENT_NAME, seq_len, step))
        # plt.show()