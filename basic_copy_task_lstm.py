# implementation of copy task in the original NTM paper
import tensorflow as tf
import numpy as np
from basic_copy_task_data import generate_batch

TRAIN_STEPS = 30000
STEPS_PER_EVAL = 100
BATCHES_PER_EVAL = 100
BATCH_SIZE = 64
MIN_GRAD = -10.0
MAX_GRAD = 10.0

NUM_UNITS = 256
NUM_LAYERS = 3

NUM_BITS_PER_VECTOR = 8
MAX_SEQUENCE_LEN = 20

DROPOUT = 0.0

class BuildModel(object):
    def __init__(self, inputs, mode):
        self.inputs = inputs
        self.mode = mode
        self._build_model()

    def _build_model(self):
        batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size")

        def single_cell(num_units):
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units,
                forget_bias=1.0)
            if DROPOUT > 0.0:
              single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell,
                input_keep_prob=(1.0 - DROPOUT))
            return single_cell

        cell = tf.contrib.rnn.MultiRNNCell([single_cell(NUM_UNITS) for _ in range(NUM_LAYERS)])

        eof = np.zeros([BATCH_SIZE, NUM_BITS_PER_VECTOR + 1])
        eof[:, NUM_BITS_PER_VECTOR] = np.ones([BATCH_SIZE])
        eof = tf.constant(eof, dtype=tf.float32)
        zero = tf.constant(np.zeros([BATCH_SIZE, NUM_BITS_PER_VECTOR + 1]), dtype=tf.float32)

        init_state = cell.zero_state(BATCH_SIZE, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell,
          tf.concat([self.inputs, tf.stack([eof] + [zero for _ in range(MAX_SEQUENCE_LEN)], axis=1)], axis=1),
          initial_state=init_state,
          dtype=tf.float32)
        self.output_logits = tf.contrib.layers.fully_connected(outputs[:, MAX_SEQUENCE_LEN + 1:, :],
          NUM_BITS_PER_VECTOR, activation_fn=None)
        self.outputs = tf.sigmoid(self.output_logits)

class BuildTrainModel(BuildModel):
    def __init__(self, train_inputs):
        super(BuildTrainModel, self).__init__(train_inputs, tf.contrib.learn.ModeKeys.TRAIN)

        self.labels, _ = tf.split(train_inputs, [NUM_BITS_PER_VECTOR, 1], axis=2)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.output_logits)

        self.loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # optimizer = tf.train.RMSPropOptimizer(3e-5, momentum=0.9)
        grads = optimizer.compute_gradients(self.loss)
        clipped_grads = [(tf.clip_by_value(grad, MIN_GRAD, MAX_GRAD), var) for grad, var in grads]
        self.train_op = optimizer.apply_gradients(clipped_grads)

        self.outputs = tf.sigmoid(self.output_logits)

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
  train_input_data = generate_batch(BATCH_SIZE)
  train_loss, _, outputs, labels = sess.run([train_model.loss, train_model.train_op, train_model.outputs, train_model.labels],
    feed_dict={train_inputs: train_input_data, 'root/batch_size:0': BATCH_SIZE})
  print 'Train loss ({0}):'.format(i), train_loss

  if i % 20 == 0:
    print 'labels', labels
    print 'outptus', outputs

  outputs[outputs >= 0.5] = 1.0
  outputs[outputs < 0.5] = 0.0
  bit_errors = np.sum(np.abs(labels - outputs))

  print 'Average bit errors/sequence:', bit_errors/BATCH_SIZE


  # if i % STEPS_PER_EVAL == 0:
  #   for j in range(BATCHES_PER_EVAL):
  #     eval_input_data = generate_batch(BATCH_SIZE)
  #     eval_loss, _ = sess.run([eval_model.loss], feed_dict={inputs: eval_input_data, 'root/batch_size:0': BATCH_SIZE})
  #     print 'Eval loss:', eval_loss



