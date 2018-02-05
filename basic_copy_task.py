# implementation of copy task in the original NTM paper
import tensorflow as tf
from ntm import NTMCell
import numpy as np
import pickle
from basic_copy_task_data import generate_batch

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_STEPS = 14000
STEPS_PER_EVAL = 100
BATCHES_PER_EVAL = 100
BATCH_SIZE = 64
MIN_GRAD = -10.0
MAX_GRAD = 10.0

NUM_READ_HEADS = 1
NUM_WRITE_HEADS = 1
NUM_MEMORY_LOCATIONS = 32
MEMORY_CELL_SIZE = 20
CONVOLUTIONAL_SHIFT_RANGE = 3
NUM_UNITS = 100
NUM_LAYERS = 1

NUM_BITS_PER_VECTOR = 8
MAX_SEQUENCE_LEN = 20

OPTIMIZER = 'RMSProp'
CURRICULUM = 'simple'
STEPS_PER_CURRICULUM_UPDATE = 400

HEAD_LOG_FILE = 'head_logs/zero_init_simple_curriculum_1.p'

class BuildModel(object):
    def __init__(self, inputs, mode):
        self.inputs = inputs
        self.mode = mode
        self._build_model()

    def _build_model(self):
        batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")

        cell = NTMCell(NUM_UNITS, NUM_MEMORY_LOCATIONS, MEMORY_CELL_SIZE, NUM_READ_HEADS, NUM_WRITE_HEADS,
                 addressing_mode='content_and_loaction', shift_range=(CONVOLUTIONAL_SHIFT_RANGE -1)/2,
                 reuse=False, output_dim=NUM_BITS_PER_VECTOR)

        eof = np.zeros([BATCH_SIZE, NUM_BITS_PER_VECTOR + 1])
        eof[:, NUM_BITS_PER_VECTOR] = np.ones([BATCH_SIZE])
        eof = tf.constant(eof, dtype=tf.float32)
        zero = tf.constant(np.zeros([BATCH_SIZE, NUM_BITS_PER_VECTOR + 1]), dtype=tf.float32)

        state = cell.zero_state(BATCH_SIZE, tf.float32)
        self.state_list = [state]
        for t in range(MAX_SEQUENCE_LEN):
            output, state = cell(self.inputs[:, t, :], state)
            self.state_list.append(state)
        output, state = cell(eof, state)
        self.state_list.append(state)

        self.outputs = []
        for t in range(MAX_SEQUENCE_LEN):
            output, state = cell(zero, state)
            self.outputs.append(output[:, 0:NUM_BITS_PER_VECTOR])
            self.state_list.append(state)
        self.output_logits = tf.transpose(self.outputs, perm=[1, 0, 2])
        self.outputs = tf.sigmoid(self.output_logits)

class BuildTrainModel(BuildModel):
    def __init__(self, train_inputs):
        super(BuildTrainModel, self).__init__(train_inputs, tf.contrib.learn.ModeKeys.TRAIN)

        self.labels, _ = tf.split(train_inputs, [NUM_BITS_PER_VECTOR, 1], axis=2)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.output_logits)

        self.loss = tf.reduce_mean(cross_entropy)

        if OPTIMIZER == 'RMSProp':
          optimizer = tf.train.RMSPropOptimizer(1e-4, momentum=0.9, decay=0.95)
        elif OPTIMIZER == 'Adam':
          optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        
        grads = optimizer.compute_gradients(self.loss)
        self.grad_norm = tf.reduce_mean([tf.norm(grad, ord=1) for grad, var in grads])
        clipped_grads = [(tf.clip_by_value(grad, MIN_GRAD, MAX_GRAD), var) for grad, var in grads]
        self.train_op = optimizer.apply_gradients(clipped_grads)

class BuildEvalModel(BuildModel):
    def __init__(self, eval_inputs):
        super(BuildEvalModel, self).__init__(eval_inputs, tf.contrib.learn.ModeKeys.EVAL)

        labels, _ = tf.split(eval_inputs, [NUM_BITS_PER_VECTOR, 1], axis=2)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.output_logits))

class BuildInferenceModel(BuildModel):
    def __init__(self, infer_inputs):
        super(BuildInferenceModel, self).__init__(infer_inputs, tf.contrib.learn.ModeKeys.INFER)

        self.outputs = tf.sigmoid(self.output_logits)

with tf.variable_scope('root'):
  train_inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, MAX_SEQUENCE_LEN, NUM_BITS_PER_VECTOR+1))
  train_model = BuildTrainModel(train_inputs)
  initializer = tf.global_variables_initializer()

with tf.variable_scope('root', reuse=True):
  eval_inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, MAX_SEQUENCE_LEN, NUM_BITS_PER_VECTOR+1))
  eval_model = BuildEvalModel(eval_inputs)

sess = tf.Session()

sess.run(initializer)

pickle.dump([], open(HEAD_LOG_FILE, "wb"))

for i in range(TRAIN_STEPS):
  if CURRICULUM == 'simple':
    # seq_len = min(2 + int((2 * float(i)/TRAIN_STEPS) * MAX_SEQUENCE_LEN), MAX_SEQUENCE_LEN)
    seq_len = min(4 + i/STEPS_PER_CURRICULUM_UPDATE, MAX_SEQUENCE_LEN)
    # seq_len = [min(1 + np.random.poisson(3 + i/STEPS_PER_CURRICULUM_UPDATE), MAX_SEQUENCE_LEN) for _ in range(BATCH_SIZE)]
  elif CURRICULUM == 'none':
    seq_len = MAX_SEQUENCE_LEN
  
  train_input_data = generate_batch(BATCH_SIZE,
    bits_per_vector=NUM_BITS_PER_VECTOR, max_seq_len=seq_len, padded_seq_len=MAX_SEQUENCE_LEN)
  
  grad_norm, train_loss, _, outputs, labels, state_list = sess.run(
    [train_model.grad_norm, train_model.loss, train_model.train_op,
    train_model.outputs, train_model.labels, train_model.state_list],
    feed_dict={train_inputs: train_input_data, 'root/batch_size:0': BATCH_SIZE})
  logger.info('Train loss ({0}): {1}'.format(i, train_loss))
  logger.info('max seq len: {0}'.format(seq_len))
  logger.info('Gradient norm ({0}): {1}'.format(i, grad_norm))

  outputs[outputs >= 0.5] = 1.0
  outputs[outputs < 0.5] = 0.0

  if i % 100 == 0:
    logger.info('labels: {0}'.format(labels[0]))
    logger.info('outputs: {0}'.format(outputs[0]))

    logger.info('max seq len: {0}'.format(seq_len))

    read_head = []
    write_head = []

    for j, states in enumerate(state_list):
      w_list = states['w_list']
      logger.info('{0}: {1}'.format(j, w_list[0][0]))
      read_head.append(w_list[0][0])

    for j, states in enumerate(state_list):
      w_list = states['w_list']
      logger.info('{0}: {1}'.format(j, w_list[1][0]))
      write_head.append(w_list[1][0])

    tmp = pickle.load(open(HEAD_LOG_FILE, "rb"))
    tmp.append({
      'labels': labels[0],
      'outputs': outputs[0],
      'read_head': read_head,
      'write_head': write_head
    })

    pickle.dump(tmp, open(HEAD_LOG_FILE, "wb"))

  bit_errors = np.sum(np.abs(labels - outputs))

  logger.info('Average bit errors/sequence: {0}'.format(bit_errors/BATCH_SIZE))

  logger.info('PARSABLE: {0},{1},{2},{3}'.format(i, seq_len if isinstance(seq_len, int) else 4 + i/STEPS_PER_CURRICULUM_UPDATE, train_loss, bit_errors/BATCH_SIZE))



