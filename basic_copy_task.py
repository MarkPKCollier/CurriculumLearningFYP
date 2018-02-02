# implementation of copy task in the original NTM paper
import tensorflow as tf
from ntm import NTMCell, NTMCell1
import numpy as np
from basic_copy_task_data import generate_batch

TRAIN_STEPS = 10000
STEPS_PER_EVAL = 100
BATCHES_PER_EVAL = 100
BATCH_SIZE = 64
MIN_GRAD = -10.0
MAX_GRAD = 10.0

NUM_READ_HEADS = 2
NUM_WRITE_HEADS = 1
NUM_MEMORY_LOCATIONS = 32
MEMORY_CELL_SIZE = 20
CONVOLUTIONAL_SHIFT_RANGE = 3
NUM_UNITS = 100
NUM_LAYERS = 1

NUM_BITS_PER_VECTOR = 8
MAX_SEQUENCE_LEN = 20

class BuildModel(object):
    def __init__(self, inputs, mode):
        self.inputs = inputs
        self.mode = mode
        self._build_model()

    def _build_model(self):
        batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")

        # cell = NTMCell(NUM_MEMORY_LOCATIONS, MEMORY_CELL_SIZE, BATCH_SIZE, 1, 1,
        #   self.mode, NUM_UNITS, NUM_LAYERS, zero_initialization=False,
        #   similarity_measure='cosine_similarity', addressing_mode='ntm', shift_range=1,
        #   reuse=False, output_dim=NUM_BITS_PER_VECTOR, controller_dropout=0.0,
        #   recurrent_controller=True, controller_forget_bias=1.0)
        cell = NTMCell1(NUM_UNITS, NUM_MEMORY_LOCATIONS, MEMORY_CELL_SIZE, 1, 1,
                 addressing_mode='content_and_loaction', shift_range=1, reuse=False, output_dim=8)

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

        # init_state = cell.zero_state(BATCH_SIZE, tf.float32)
        # outputs, final_state = tf.nn.dynamic_rnn(cell,
        #   tf.concat([self.inputs, tf.stack([eof] + [zero for _ in range(MAX_SEQUENCE_LEN)], axis=1)], axis=1),
        #   initial_state=init_state,
        #   dtype=tf.float32)
        # self.output_logits = outputs[:, MAX_SEQUENCE_LEN + 1:, :]
        # self.outputs = tf.sigmoid(self.output_logits)
        
        # self.final_w_list = final_state[-2]
        # self.start_memory = init_state[-1]
        # self.final_memory = final_state[-1]

class BuildTrainModel(BuildModel):
    def __init__(self, train_inputs):
        super(BuildTrainModel, self).__init__(train_inputs, tf.contrib.learn.ModeKeys.TRAIN)

        self.labels, _ = tf.split(train_inputs, [NUM_BITS_PER_VECTOR, 1], axis=2)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.output_logits)

        self.loss = tf.reduce_mean(cross_entropy)

        # optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
        optimizer = tf.train.RMSPropOptimizer(1e-4, momentum=0.9, decay=0.95)
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

# with tf.variable_scope('root', reuse=True):
#   infer_inputs = tf.placeholder(tf.float32, shape=(None, MAX_SEQUENCE_LEN, NUM_BITS_PER_VECTOR+1))
#   inference_model = BuildInferenceModel(infer_inputs)

sess = tf.Session()

sess.run(initializer)

for i in range(TRAIN_STEPS):
  # train_input_data = generate_batch(BATCH_SIZE,
  #   bits_per_vector=NUM_BITS_PER_VECTOR, max_seq_len=MAX_SEQUENCE_LEN)
  # seq_len = min(int(MAX_SEQUENCE_LEN * (float(i)/TRAIN_STEPS + 0.2)), MAX_SEQUENCE_LEN)
  seq_len = min(5 + int((float(i)/TRAIN_STEPS) * MAX_SEQUENCE_LEN), MAX_SEQUENCE_LEN)
  train_input_data = generate_batch(BATCH_SIZE,
    bits_per_vector=NUM_BITS_PER_VECTOR, max_seq_len=MAX_SEQUENCE_LEN, padded_seq_len=MAX_SEQUENCE_LEN)
  grad_norm, train_loss, _, outputs, labels, state_list = sess.run(
    [train_model.grad_norm, train_model.loss, train_model.train_op,
    train_model.outputs, train_model.labels, train_model.state_list],
    feed_dict={train_inputs: train_input_data, 'root/batch_size:0': BATCH_SIZE})
  print 'Train loss ({0}):'.format(i), train_loss
  print 'max seq len', seq_len
  print 'Gradient norm ({0}):'.format(i), grad_norm

  if i % 100 == 0:
    print 'labels', labels
    print 'outputs', outputs

    print 'labels', labels[0]
    print 'outputs', outputs[0]

    print 'max seq len', seq_len

    # for i, states in enumerate(state_list):
    #   w_list = states[-2]
    #   print i, w_list[0][0]

    # for i, states in enumerate(state_list):
    #   w_list = states[-2]
    #   print i, w_list[1][0]

    for i, states in enumerate(state_list):
      w_list = states['w_list']
      print i, w_list[0][0]

    for i, states in enumerate(state_list):
      w_list = states['w_list']
      print i, w_list[1][0]

  outputs[outputs >= 0.5] = 1.0
  outputs[outputs < 0.5] = 0.0
  bit_errors = np.sum(np.abs(labels - outputs))

  print 'Average bit errors/sequence:', bit_errors/BATCH_SIZE

  # if i % STEPS_PER_EVAL == 0:
  #   for j in range(BATCHES_PER_EVAL):
  #     eval_input_data = generate_batch(BATCH_SIZE)
  #     eval_loss, _ = sess.run([eval_model.loss], feed_dict={inputs: eval_input_data, 'root/batch_size:0': BATCH_SIZE})
  #     print 'Eval loss:', eval_loss



