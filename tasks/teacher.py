import tensorflow as tf
import numpy as np
import bisect

class Teacher:
    def __init__(self, lessons, dimensions_per_lesson, num_hidden_layers, num_units):
        self.lessons = lessons
        self.num_lessons = len(lessons)
        self.dimensions_per_lesson = dimensions_per_lesson
        self.num_hidden_layers = num_hidden_layers
        self.num_units = num_units
        self.beta = 1

        self.rewards = []
        self.t = 1
        self.max_rewards = 50000

        graph = tf.Graph()
        with graph.as_default():
            self._build()

        self.sess = tf.Session(graph=graph)
        self.sess.run(self.initializer)

    def draw_task(self):
        outputs = self.outputs.eval(
            session=self.sess,
            feed_dict={
              self.batch_size: self.num_lessons,
              self.inputs: map(lambda x: [x], self.lessons)
            }).flatten()

        e_outputs = np.exp(self.beta * outputs)
        dist = e_outputs / np.sum(e_outputs)

        lesson = np.random.choice(self.lessons)
        # lesson = np.random.choice(self.num_lessons, p=dist)

        # print outputs, dist, lesson
        # print outputs, dist

        return lesson

    def update_w(self, lesson, v, t):
        r = v/t
        # self._reservoir_sample(r)
        # q_lo, q_hi = self._quantiles()
        # r = self._r(q_lo, q_hi, r)

        loss, outputs, _ = self.sess.run([self.loss, self.outputs, self.train_op],
        feed_dict={
            self.batch_size: 1,
            self.inputs: [[self.lessons[lesson]]],
            self.labels: [r]
        })

        print self.lessons[lesson], r, outputs[0][0], loss

        self.t += 1

        # print "in teacher: update_w", lesson, v, t, train_loss

    def _build(self):
        self.batch_size = tf.placeholder(tf.float32)
        self.inputs = tf.placeholder(tf.float32, shape=[None] + [1 for _ in range(self.dimensions_per_lesson)])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        inputs_ = self.inputs
        for _ in range(self.num_hidden_layers):
            inputs_ = tf.contrib.layers.fully_connected(inputs_, self.num_units, activation_fn=tf.nn.relu)

        self.outputs = tf.contrib.layers.fully_connected(inputs_, 1, activation_fn=None)
        self.loss = tf.nn.l2_loss(self.outputs - self.labels)/self.batch_size

        # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.train_op = optimizer.minimize(self.loss)

        self.initializer = tf.global_variables_initializer()

    def _quantiles(self):
        q_lo_pos = int(0.2 * len(self.rewards))
        q_hi_pos = int(0.8 * len(self.rewards)) - 1
        return self.rewards[q_lo_pos], self.rewards[q_hi_pos]

    def _reservoir_sample(self, r_):
        insert = False
        if len(self.rewards) >= self.max_rewards and np.random.random_sample() < 10.0/float(self.t):
            pos = np.random.randint(0, high=len(self.rewards))
            del self.rewards[pos]
            insert = True
        if insert or len(self.rewards) < self.max_rewards:
            pos = bisect.bisect_left(self.rewards, r_)
            self.rewards.insert(pos, r_)

    def _r(self, q_lo, q_hi, r_):
        if r_ < q_lo:
            return -1.0
        elif r_ >= q_hi:
            return 1.0
        else:
            return (2.0*(r_ - q_lo)/float(q_hi - q_lo)) - 1.0

