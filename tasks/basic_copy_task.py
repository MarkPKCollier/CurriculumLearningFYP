# implementation of copy task in the original NTM paper
import tensorflow as tf
from ntm import NTMCell
import numpy as np
import pickle
from basic_copy_task_data import generate_batch
# from dnc.dnc import DNC
from dnc import DNC

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mann', type=str, default='none', help='none | ntm | dnc')
parser.add_argument('--controller_type', type=str, default='lstm', help='lstm | rnn | feed_forward')
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_units', type=int, default=100)
parser.add_argument('--num_memory_locations', type=int, default=128)
parser.add_argument('--memory_size', type=int, default=20)
parser.add_argument('--num_read_heads', type=int, default=1)
parser.add_argument('--num_write_heads', type=int, default=1)
parser.add_argument('--conv_shift_range', type=int, default=1, help='only necessary for ntm')
parser.add_argument('--init_mode', type=str, default='random', help='random | zero -> only necessary for ntm')
parser.add_argument('--clip_value', type=int, default=20, help='Maximum absolute value of controller and dnc outputs.')

parser.add_argument('--optimizer', type=str, default='RMSProp', help='RMSProp | Adam')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--num_train_steps', type=int, default=14000)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--curriculum', type=str, default='simple')
parser.add_argument('--steps_per_curriculum_update', type=int, default=1200)
parser.add_argument('--pad_to_max_seq_len', type=bool, default=True)

parser.add_argument('--task', type=str, default='copy',
    help='copy | repeat_copy | associative_recall | dynamic_ngrams | priority_sort')
parser.add_argument('--num_bits_per_vector', type=int, default=8)
parser.add_argument('--max_seq_len', type=int, default=20)

parser.add_argument('--verbose', type=bool, default=False, help='if true saves heatmaps and prints lots of feedback')
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--steps_per_eval', type=int, default=200)

args = parser.parse_args()

if args.task == 'copy':
    TOTAL_SEQUENCE_LENGTH = args.max_seq_len * 2 + 1
    OUTPUT_SEQUENCE_LENGTH = args.max_seq_len

# def _like_rnncell(cell):
#     """Checks that a given object is an RNNCell by using duck typing."""
#     conditions = [hasattr(cell, "output_size"), hasattr(cell, "state_size"),
#       hasattr(cell, "zero_state"), callable(cell)]
#     print conditions 
#     return all(conditions)

class BuildModel(object):
    def __init__(self, max_seq_len, inputs, mode):
        self.max_seq_len = max_seq_len
        self.inputs = inputs
        self.mode = mode
        self._build_model()

    def _build_model(self):
        if args.mann == 'none':
            def single_cell(num_units):
                return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)

            cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.MultiRNNCell([single_cell(args.num_units) for _ in range(args.num_layers)]),
                args.num_bits_per_vector,
                activation=None)
            initial_state = cell.zero_state(args.batch_size, tf.float32)
        elif args.mann == 'ntm':
            cell = NTMCell(args.controller_type, args.num_layers, args.num_units, args.num_memory_locations, args.memory_size,
                args.num_read_heads, args.num_write_heads, init_mode=args.init_mode,
                addressing_mode='content_and_location', shift_range=args.conv_shift_range,
                reuse=False, output_dim=args.num_bits_per_vector)

            initial_state = cell.zero_state(args.batch_size, tf.float32)
        else:
            access_config = {
                'memory_size': args.num_memory_locations,
                'word_size': args.memory_size,
                'num_reads': args.num_read_heads,
                'num_writes': args.num_write_heads,
            }
            controller_config = {
                'controller_type': args.controller_type,
                'hidden_size': args.num_units,
            }

            cell = DNC(access_config, controller_config, args.num_bits_per_vector, args.clip_value)
            initial_state = cell.initial_state(args.batch_size)
        
        # _like_rnncell(cell)

        output_sequence, _ = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.inputs,
            time_major=False,
            initial_state=initial_state)

        # self.output_logits = output_sequence[:, -OUTPUT_SEQUENCE_LENGTH:, :]
        # self.output_logits = output_sequence[:, -self.max_seq_len:, :]
        self.output_logits = output_sequence[:, self.max_seq_len+1:, :]

        self.outputs = tf.sigmoid(self.output_logits)

class BuildTrainModel(BuildModel):
    def __init__(self, max_seq_len, inputs, outputs):
        super(BuildTrainModel, self).__init__(max_seq_len, inputs, tf.contrib.learn.ModeKeys.TRAIN)

        # self.labels, _ = tf.split(train_inputs, [NUM_BITS_PER_VECTOR, 1], axis=2)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs, logits=self.output_logits)

        self.loss = tf.reduce_mean(cross_entropy)

        if args.optimizer == 'RMSProp':
          optimizer = tf.train.RMSPropOptimizer(args.learning_rate, momentum=0.9, decay=0.95)
        elif args.optimizer == 'Adam':
          optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        
        grads = optimizer.compute_gradients(self.loss)
        self.grad_norm = tf.reduce_mean([tf.norm(grad, ord=1) for grad, var in grads])
        # clipped_grads = [(tf.clip_by_value(grad, -args.max_grad_norm, args.max_grad_norm), var) for grad, var in grads]
        # self.train_op = optimizer.apply_gradients(clipped_grads)

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), 50)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

class BuildEvalModel(BuildModel):
    def __init__(self, max_seq_len, inputs, outputs):
        super(BuildEvalModel, self).__init__(max_seq_len, inputs, tf.contrib.learn.ModeKeys.EVAL)

        # labels, _ = tf.split(eval_inputs, [NUM_BITS_PER_VECTOR, 1], axis=2)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs, logits=self.output_logits))

class BuildInferenceModel(BuildModel):
    def __init__(self, inputs):
        super(BuildInferenceModel, self).__init__(inputs, tf.contrib.learn.ModeKeys.INFER)

        self.outputs = tf.sigmoid(self.output_logits)

with tf.variable_scope('root'):
  # train_inputs = tf.placeholder(tf.float32, shape=(args.batch_size, TOTAL_SEQUENCE_LENGTH, args.num_bits_per_vector+1))
  # train_outputs = tf.placeholder(tf.float32, shape=(args.batch_size, OUTPUT_SEQUENCE_LENGTH, args.num_bits_per_vector))
  train_max_seq_len = tf.placeholder(tf.int32)
  train_inputs = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector+1))
  train_outputs = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector))
  train_model = BuildTrainModel(train_max_seq_len, train_inputs, train_outputs)
  initializer = tf.global_variables_initializer()

with tf.variable_scope('root', reuse=True):
  # eval_inputs = tf.placeholder(tf.float32, shape=(args.batch_size, TOTAL_SEQUENCE_LENGTH, args.num_bits_per_vector+1))
  # eval_outputs = tf.placeholder(tf.float32, shape=(args.batch_size, OUTPUT_SEQUENCE_LENGTH, args.num_bits_per_vector))
  eval_max_seq_len = tf.placeholder(tf.int32)
  eval_inputs = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector+1))
  eval_outputs = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector))
  eval_model = BuildEvalModel(eval_max_seq_len, eval_inputs, eval_outputs)

sess = tf.Session()

sess.run(initializer)

if args.verbose:
  pickle.dump([], open('../head_logs/{0}.p'.format(args.experiment_name), 'wb'))

for i in range(args.num_train_steps):
  if args.curriculum == 'simple':
    # seq_len = min(2 + int((2 * float(i)/TRAIN_STEPS) * MAX_SEQUENCE_LEN), MAX_SEQUENCE_LEN)
    seq_len = min(4 + i/args.steps_per_curriculum_update, args.max_seq_len)
    # seq_len = [min(1 + np.random.poisson(3 + i/STEPS_PER_CURRICULUM_UPDATE), MAX_SEQUENCE_LEN) for _ in range(BATCH_SIZE)]
  elif args.curriculum == 'none':
    seq_len = args.max_seq_len
  
  inputs, labels = generate_batch(args.batch_size,
    bits_per_vector=args.num_bits_per_vector, max_seq_len=seq_len, padded_seq_len=args.max_seq_len)

  grad_norm, train_loss, _, outputs = sess.run(
    [train_model.grad_norm, train_model.loss, train_model.train_op, train_model.outputs],
    feed_dict={train_inputs: inputs,
    train_outputs: labels,
    train_max_seq_len: args.max_seq_len if args.pad_to_max_seq_len else seq_len})
  logger.info('Train loss ({0}): {1}'.format(i, train_loss))
  logger.info('max seq len: {0}'.format(seq_len))
  logger.info('Gradient norm ({0}): {1}'.format(i, grad_norm))

  outputs[outputs >= 0.5] = 1.0
  outputs[outputs < 0.5] = 0.0

  if i % 100 == 0:
    if args.verbose:
        logger.info('labels: {0}'.format(labels[0]))
        logger.info('outputs: {0}'.format(outputs[0]))

        logger.info('max seq len: {0}'.format(seq_len))

        # read_head = []
        # write_head = []

        # for j, states in enumerate(state_list):
        #   w_list = states['w_list']
        #   logger.info('{0}: {1}'.format(j, w_list[0][0]))
        #   read_head.append(w_list[0][0])

        # for j, states in enumerate(state_list):
        #   w_list = states['w_list']
        #   logger.info('{0}: {1}'.format(j, w_list[1][0]))
        #   write_head.append(w_list[1][0])

        # if WRITE_HEATMAPS:
        #   tmp = pickle.load(open('head_logs/{0}.p'.format(args.experiment_name), 'rb'))
        #   tmp.append({
        #     'labels': train_labels[0],
        #     'outputs': outputs[0],
        #     'read_head': read_head,
        #     'write_head': write_head
        #   })

        #   pickle.dump(tmp, open('../head_logs/{0}.p'.format(args.experiment_name), 'wb'))

  bit_errors = np.sum(np.abs(labels - outputs))

  logger.info('Average bit errors/sequence: {0}'.format(bit_errors/args.batch_size))

  logger.info('PARSABLE: {0},{1},{2},{3}'.format(i,
    seq_len if isinstance(seq_len, int) else 4 + i/args.steps_per_curriculum_update,
    train_loss, bit_errors/args.batch_size))



