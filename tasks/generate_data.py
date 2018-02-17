import numpy as np

snap_boolean = np.vectorize(lambda x: 1.0 if x > 0.5 else 0.0)

class CopyTaskData:
    def generate_batches(self, num_batches, batch_size, bits_per_vector=8, curriculum_point=20, max_seq_len=20,
        curriculum='uniform', pad_to_max_seq_len=False):
        batches = []
        for i in range(num_batches):
            if curriculum == 'deterministic_uniform':
                seq_len = 1 + (i % max_seq_len)
            elif curriculum == 'uniform':
                seq_len = np.random.randint(low=1, high=max_seq_len+1)
            elif curriculum == 'none':
                seq_len = max_seq_len
            elif curriculum == 'naive':
                seq_len = curriculum_point
            elif curriculum == 'look_back':
                seq_len = curriculum_point if np.random.random_sample() < 0.9 else np.random.randint(low=1, high=curriculum_point+1)
            elif curriculum == 'look_back_and_forward':
                seq_len = curriculum_point if np.random.random_sample() < 0.8 else np.random.randint(low=1, high=max_seq_len+1)
            
            pad_to_len = max_seq_len if pad_to_max_seq_len else seq_len

            def generate_sequence():
                return np.asarray([snap_boolean(np.append(np.random.rand(bits_per_vector), 0)) for _ in range(seq_len)] \
                    + [np.zeros(bits_per_vector+1) for _ in range(pad_to_len - seq_len)])

            inputs = np.asarray([generate_sequence() for _ in range(batch_size)]).astype(np.float32)
            eos = np.ones([batch_size, 1, bits_per_vector + 1])
            output_inputs = np.zeros_like(inputs)

            full_inputs = np.concatenate((inputs, eos, output_inputs), axis=1)

            batches.append((pad_to_len, full_inputs, inputs[:, :, :bits_per_vector]))
        return batches

    def error_per_seq(self, labels, outputs, num_seq):
        outputs[outputs >= 0.5] = 1.0
        outputs[outputs < 0.5] = 0.0
        bit_errors = np.sum(np.abs(labels - outputs))
        return bit_errors/num_seq
