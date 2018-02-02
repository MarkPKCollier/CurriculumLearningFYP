import numpy as np
import tensorflow as tf

class NTMCell1():
    def __init__(self, rnn_size, memory_size, memory_vector_dim, read_head_num, write_head_num,
                 addressing_mode='content_and_loaction', shift_range=1, reuse=False, output_dim=None):
        self.rnn_size = rnn_size
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.addressing_mode = addressing_mode
        self.reuse = reuse
        self.controller = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)
        self.step = 0
        self.output_dim = output_dim
        self.shift_range = shift_range

    def __call__(self, x, prev_state):
        prev_read_vector_list = prev_state['read_vector_list']      # read vector in Sec 3.1 (the content that is
                                                                    # read out, length = memory_vector_dim)
        prev_controller_state = prev_state['controller_state']      # state of controller (LSTM hidden state)

        # x + prev_read_vector -> controller (RNN) -> controller_output
        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller(controller_input, prev_controller_state)

        # controller_output     -> k (dim = memory_vector_dim, compared to each vector in M, Sec 3.1)
        #                       -> beta (positive scalar, key strength, Sec 3.1)                -> w^c
        #                       -> g (scalar in (0, 1), blend between w_prev and w^c, Sec 3.2)  -> w^g
        #                       -> s (dim = shift_range * 2 + 1, shift weighting, Sec 3.2)      -> w^~
        #                            (not memory_size, that's too wide)
        #                       -> gamma (scalar (>= 1), sharpen the final result, Sec 3.2)     -> w    * num_heads
        # controller_output     -> erase, add vector (dim = memory_vector_dim, \in (0, 1), Sec 3.2)     * write_head_num

        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        num_heads = self.read_head_num + self.write_head_num
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            o2p_w = tf.get_variable('o2p_w', [controller_output.get_shape()[1], total_parameter_num],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            o2p_b = tf.get_variable('o2p_b', [total_parameter_num],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            parameters = tf.nn.xw_plus_b(controller_output, o2p_w, o2p_b)
        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, axis=1)

        # k, beta, g, s, gamma -> w

        prev_w_list = prev_state['w_list']  # vector of weightings (blurred address) over locations
        prev_M = prev_state['M']
        w_list = []
        p_list = []
        for i, head_parameter in enumerate(head_parameter_list):

            # Some functions to constrain the result in specific range
            # exp(x)                -> x > 0
            # sigmoid(x)            -> x \in (0, 1)
            # softmax(x)            -> sum_i x_i = 1
            # log(exp(x) + 1) + 1   -> x > 1

            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = tf.sigmoid(head_parameter[:, self.memory_vector_dim]) * 10        # do not use exp, it will explode!
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])
            s = tf.nn.softmax(
                head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
            )
            gamma = tf.log(tf.exp(head_parameter[:, -1]) + 1) + 1
            with tf.variable_scope('addressing_head_%d' % i):
                w = self.addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])     # Figure 2
            w_list.append(w)
            p_list.append({'k': k, 'beta': beta, 'g': g, 's': s, 'gamma': gamma})

        # Reading (Sec 3.1)

        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)

        # Writing (Sec 3.2)

        write_w_list = w_list[self.read_head_num:]
        M = prev_M
        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
            M = M * (tf.ones(M.get_shape()) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)

        # controller_output -> NTM output

        if not self.output_dim:
            output_dim = x.get_shape()[1]
        else:
            output_dim = self.output_dim
        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):
            o2o_w = tf.get_variable('o2o_w', [controller_output.get_shape()[1], output_dim],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            o2o_b = tf.get_variable('o2o_b', [output_dim],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            NTM_output = tf.nn.xw_plus_b(controller_output, o2o_w, o2o_b)

        state = {
            'controller_state': controller_state,
            'read_vector_list': read_vector_list,
            'w_list': w_list,
            'p_list': p_list,
            'M': M
        }

        self.step += 1
        return NTM_output, state

    def addressing(self, k, beta, g, s, gamma, prev_M, prev_w):

        # Sec 3.3.1 Focusing by Content

        # Cosine Similarity

        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(prev_M, k)
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True))
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keep_dims=True))
        norm_product = M_norm * k_norm
        K = tf.squeeze(inner_product / (norm_product + 1e-8))                   # eq (6)

        # Calculating w^c

        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keep_dims=True)  # eq (5)

        if self.addressing_mode == 'content':                                   # Only focus on content
            return w_c

        # Sec 3.3.2 Focusing by Location

        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w                                        # eq (7)

        s = tf.concat([s[:, :self.shift_range + 1],
                       tf.zeros([s.get_shape()[0], self.memory_size - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1)
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
            axis=1
        )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)      # eq (8)
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keep_dims=True)        # eq (9)

        return w

    def zero_state(self, batch_size, dtype):
        def expand(x, dim, N):
            return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

        with tf.variable_scope('init', reuse=self.reuse):
            # state = {
            #     # 'controller_state': self.controller.zero_state(batch_size, dtype),
            #     # 'read_vector_list': [tf.zeros([batch_size, self.memory_vector_dim])
            #     #                      for _ in range(self.read_head_num)],
            #     # 'w_list': [tf.zeros([batch_size, self.memory_size])
            #     #            for _ in range(self.read_head_num + self.write_head_num)],
            #     # 'M': tf.zeros([batch_size, self.memory_size, self.memory_vector_dim])
            #     'controller_state': expand(tf.tanh(tf.get_variable('init_state', self.rnn_size,
            #                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
            #                       dim=0, N=batch_size),
            #     'read_vector_list': [expand(tf.nn.softmax(tf.get_variable('init_r_%d' % i, [self.memory_vector_dim],
            #                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
            #                       dim=0, N=batch_size)
            #                for i in range(self.read_head_num)],
            #     'w_list': [expand(tf.nn.softmax(tf.get_variable('init_w_%d' % i, [self.memory_size],
            #                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
            #                       dim=0, N=batch_size) if self.addressing_mode == 'content_and_loaction'
            #                else tf.zeros([batch_size, self.memory_size])
            #                for i in range(self.read_head_num + self.write_head_num)],
            #     'M': expand(tf.tanh(tf.get_variable('init_M', [self.memory_size, self.memory_vector_dim],
            #                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
            #                       dim=0, N=batch_size)
            # }

            state = {
                'controller_state': expand(tf.tanh(tf.get_variable('init_state', self.rnn_size,
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                                  dim=0, N=batch_size),
                'read_vector_list': [tf.zeros([batch_size, self.memory_vector_dim])
                    for _ in range(self.read_head_num)],
                'w_list': [tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, self.memory_size-1))], axis=1)
                    for _ in range(self.read_head_num + self.write_head_num)],
                'M': tf.tanh(tf.truncated_normal(
                    [batch_size, self.memory_size, self.memory_vector_dim],
                    mean=0.0, stddev=1.0/(np.sqrt(self.memory_size + self.memory_vector_dim))))
            }

            return state

class NTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self,
        num_memory_locations, memory_vector_dim, batch_size, read_head_num, write_head_num,
        mode, controller_num_units, controller_num_layers, zero_initialization=False,
        similarity_measure='cosine_similarity', addressing_mode='ntm', shift_range=2,
        reuse=False, output_dim=None, controller_dropout=0.0,
        recurrent_controller=True, controller_forget_bias=1.0):
        self.zero_initialization = zero_initialization
        self.similarity_measure = similarity_measure
        self.num_memory_locations = num_memory_locations
        self.memory_vector_dim = memory_vector_dim
        self.batch_size = batch_size
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.addressing_mode = addressing_mode
        self.reuse = reuse
        self.dropout = controller_dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

        self.controller_num_units = controller_num_units
        self.controller_num_layers = controller_num_layers
        self.recurrent_controller = recurrent_controller

        if self.recurrent_controller:
            def single_cell(num_units):
                single_cell = tf.contrib.rnn.BasicLSTMCell(
                    num_units,
                    forget_bias=controller_forget_bias)
                if self.dropout > 0.0:
                  single_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=single_cell,
                    input_keep_prob=(1.0 - self.dropout))
                return single_cell

            self.controller = tf.contrib.rnn.MultiRNNCell([single_cell(self.controller_num_units) for _ in range(self.controller_num_layers)])
        
        self.step = 0
        self.output_dim = output_dim
        self.shift_range = shift_range

    def __call__(self, x, prev_state):
        if self.recurrent_controller:
            prev_controller_state = prev_state[0]
            prev_read_vector_list = prev_state[1]
            prev_w_list = prev_state[2]
            memory = prev_state[3]
        else:
            prev_read_vector_list = prev_state[0]
            prev_w_list = prev_state[1]
            memory = prev_state[2]
        memory = tf.reshape(memory, [-1, self.num_memory_locations, self.memory_vector_dim])

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            if self.recurrent_controller:
                controller_output, controller_state = self.controller.__call__(
                    controller_input, prev_controller_state)
            else:
                controller_output = tf.contrib.layers.fully_connected(
                    controller_input, self.controller_num_units,
                    weights_initializer=tf.random_uniform_initializer(-0.1, 0.1))
                for _ in range(self.controller_num_layers-1):
                    controller_output = tf.contrib.layers.fully_connected(
                        controller_output, self.controller_num_units,
                        weights_initializer=tf.random_uniform_initializer(-0.1, 0.1))

        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        num_heads = self.read_head_num + self.write_head_num
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            parameters = tf.contrib.layers.fully_connected(controller_output, total_parameter_num,
                activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(-0.1, 0.1))

        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, axis=1)

        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            # k = head_parameter[:, 0:self.memory_vector_dim]
            beta = tf.nn.softplus(head_parameter[:, self.memory_vector_dim])
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])
            s = tf.nn.softmax(
                head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
            )
            gamma = tf.nn.softplus(head_parameter[:, -1]) + 1
            with tf.variable_scope('addressing_head_%d' % i):
                w = self.addressing(k, beta, g, s, gamma, prev_w_list[i], memory)
            w_list.append(w)

        read_vector_list = []
        read_w_list = w_list[:self.read_head_num]
        for i in range(self.read_head_num):
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * memory, axis=1)
            read_vector_list.append(read_vector)

        write_w_list = w_list[self.read_head_num:]
        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
            memory = memory * (tf.ones(memory.get_shape()) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)
            # w = tf.expand_dims(write_w_list[i], axis=2)
            # weighted_erase_vectors = tf.subtract(1.0, tf.multiply(w,
            #     tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)))
            # erased_memory = tf.multiply(memory, weighted_erase_vectors)
            # memory = erased_memory + tf.multiply(w,
            #     tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1))

        if not self.output_dim:
            self.output_dim = x.get_shape()[1]
        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):
            controller_output = tf.concat([controller_output] + read_vector_list, axis=1)
            NTM_output = tf.contrib.layers.fully_connected(controller_output, self.output_dim,
                activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(-0.1, 0.1))
            if self.dropout > 0.0:
                NTM_output = tf.nn.dropout(NTM_output, 1-self.dropout)

        memory = tf.reshape(memory, [-1, self.num_memory_locations * self.memory_vector_dim])

        self.step += 1

        # output_w_read_v = tf.contrib.layers.fully_connected(
        #     tf.concat([NTM_output, tf.concat(read_vector_list, axis=1)], axis=1),
        #     self.output_dim,
        #     activation_fn=tf.tanh,
        #     weights_initializer=tf.random_uniform_initializer(-0.1, 0.1))

        # if self.dropout > 0.0:
        #     output_w_read_v = tf.nn.dropout(output_w_read_v, 1-self.dropout)

        if self.recurrent_controller:
            return NTM_output, (controller_state, read_vector_list, w_list, memory)
        else:
            return NTM_output, (read_vector_list, w_list, memory)

    def addressing(self, k, beta, g, s, gamma, prev_w, memory):
        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(memory, k)

        if self.similarity_measure == 'dot_product':
            inner_product = tf.squeeze(inner_product, axis=2)
            w_c = tf.nn.softmax(tf.expand_dims(beta, axis=1) * inner_product, dim=1)
        else:
            # k_norm = tf.norm(k, axis=1)
            # M_norm = tf.norm(memory, axis=2)
            # norm_product = tf.expand_dims(M_norm * k_norm, axis=2)

            # cosine_sim = tf.squeeze(inner_product / (norm_product + 1e-8))
            # cosine_sim.set_shape([None, self.num_memory_locations])

            # w_c = tf.nn.softmax(tf.expand_dims(beta, axis=1) * cosine_sim, dim=1)

            # w_c.set_shape([None, self.num_memory_locations])

            k_norm = tf.norm(k, axis=1, keep_dims=True)
            M_norm = tf.norm(memory, axis=2, keep_dims=True)
            norm_product = M_norm * k_norm

            cosine_sim = tf.squeeze(inner_product / (norm_product + 1e-8))

            w_c = tf.nn.softmax(tf.expand_dims(beta, axis=1) * cosine_sim, dim=1)

        if self.addressing_mode == 'content':
            return w_c

        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w

        if self.addressing_mode == 'content_and_interpolation':
            return w_g

        s = tf.concat([s[:, :self.shift_range + 1],
                       tf.zeros([self.batch_size, self.num_memory_locations - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1)
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.num_memory_locations - i - 1:self.num_memory_locations * 2 - i - 1] for i in range(self.num_memory_locations)],
            axis=1
        )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)
        

        # padded_w = tf.concat([w_g[:, -(2*self.shift_range-1):], w_g, w_g[:, :(2*self.shift_range-1)]], axis=1)
        # print 'convolution', w_g, padded_w, s
        # print 'inputs to convolution', tf.expand_dims(tf.expand_dims(padded_w, axis=0), axis=3), tf.expand_dims(tf.expand_dims(s, axis=0), axis=2)
        # w_ = tf.nn.conv1d(tf.expand_dims(tf.transpose(padded_w), axis=0), tf.expand_dims(tf.transpose(s), axis=2), 1, 'VALID')
        # print w_

        # padded_w = tf.concat([w_g[:, -(2*self.shift_range-1):], w_g, w_g[:, :(2*self.shift_range-1)]], axis=1)
        # print 'convolution', w_g, padded_w, s
        # print 'inputs to convolution', tf.expand_dims(tf.expand_dims(padded_w, axis=0), axis=3), tf.expand_dims(tf.expand_dims(s, axis=0), axis=2)
        # w_ = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(padded_w, axis=0), axis=3), tf.expand_dims(tf.expand_dims(s, axis=0), axis=2), 1, 'VALID')
        # print w_

        # shifted_weights = []
        # for b in range(self.batch_size):
        #     padded_w = tf.concat([w_g[b, -(2*self.shift_range-1):], w_g[b, :], w_g[b, :(2*self.shift_range-1)]], axis=0)
        #     print 'convolution', w_g[b, :], padded_w, s[b]
        #     print 'inputs to convolution', padded_w, s[b]
        #     w_ = tf.nn.conv1d(
        #         tf.expand_dims(tf.expand_dims(padded_w, axis=0), axis=2),
        #         tf.expand_dims(tf.expand_dims(s[b], axis=1), axis=2), 1, 'VALID')
        #     w_ = tf.squeeze(w_, axis=[-1])
        #     print 'results', w_
        #     shifted_weights.append(w_)

        # w_ = tf.concat(shifted_weights, axis=0)
        # print 'here', w_

        if self.addressing_mode == 'content_interpolation_and_shift':
            return w_

        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keep_dims=True)

        return w

    def zero_state(self, batch_size, dtype):
        # def expand(x, dim, N):
        #     return tf.concat([tf.expand_dims(x, dim) for _ in tf.range(N)], axis=dim)

        def expand(x, dim, N):
            return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

        with tf.variable_scope('init', reuse=self.reuse):
            if self.zero_initialization:
                read_vector_list = [tf.zeros([batch_size, self.memory_vector_dim])
                    for _ in range(self.read_head_num)]
                w_list = [tf.zeros([batch_size, self.num_memory_locations])
                    for _ in range(self.read_head_num + self.write_head_num)]
                memory = tf.zeros([batch_size, self.num_memory_locations, self.memory_vector_dim])
                memory = tf.reshape(memory, [-1, self.num_memory_locations * self.memory_vector_dim])
            else:
                # read_vector_list = [tf.sigmoid(
                #     tf.truncated_normal([batch_size, self.memory_vector_dim],
                #     mean=0.0, stddev=0.5))
                #     for _ in range(self.read_head_num)]
                read_vector_list = [tf.zeros([batch_size, self.memory_vector_dim])
                    for _ in range(self.read_head_num)]
                # w_list = [tf.nn.softmax(tf.truncated_normal([batch_size, self.num_memory_locations],
                #     mean=0.0, stddev=0.5), dim=1)
                #     for _ in range(self.read_head_num + self.write_head_num)]
                w_list = [tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, self.num_memory_locations-1))], axis=1)
                    for _ in range(self.read_head_num + self.write_head_num)]
                memory = tf.tanh(tf.truncated_normal(
                    [batch_size, self.num_memory_locations, self.memory_vector_dim],
                    mean=0.0, stddev=1.0/(np.sqrt(self.num_memory_locations + self.memory_vector_dim))))
                memory = tf.reshape(memory, [-1, self.num_memory_locations * self.memory_vector_dim])

                # read_vector_list = [expand(tf.nn.softmax(tf.get_variable('init_r_%d' % i, [self.memory_vector_dim],
                #                             initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                #                   dim=0, N=batch_size)
                #            for i in range(self.read_head_num)]
                # w_list = [expand(tf.nn.softmax(tf.get_variable('init_w_%d' % i, [self.num_memory_locations],
                #                             initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                #                   dim=0, N=batch_size)
                #            for i in range(self.read_head_num + self.write_head_num)]
                # memory = expand(tf.tanh(tf.get_variable('init_M', [self.num_memory_locations, self.memory_vector_dim],
                #     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                #     dim=0, N=batch_size)

            if self.recurrent_controller:
                controller_state = self.controller.zero_state(batch_size, dtype)
                return (controller_state, read_vector_list, w_list, memory)
            else:
                return (read_vector_list, w_list, memory)

    @property
    def state_size(self):
        if self.recurrent_controller:
            return (self.controller.state_size,
                [self.memory_vector_dim for _ in range(self.read_head_num)],
                [self.num_memory_locations for _ in range(self.read_head_num)],
                tf.TensorShape([self.num_memory_locations * self.memory_vector_dim]))
        else:
            return ([self.memory_vector_dim for _ in range(self.read_head_num)],
                [self.num_memory_locations for _ in range(self.read_head_num)],
                tf.TensorShape([self.num_memory_locations * self.memory_vector_dim]))

    @property
    def output_size(self):
        return self.output_dim

