import numpy as np

snap_boolean = np.vectorize(lambda x: 1.0 if x > 0.5 else 0.0)

def generate_batch(batch_size, bits_per_vector=8, max_seq_len=20, padded_seq_len=20):
    if isinstance(max_seq_len, int):
        sequence_lengths = np.random.randint(low=1, high=max_seq_len+1, size=batch_size)
    else:
        sequence_lengths = [np.random.randint(low=1, high=high+1) for high in max_seq_len]

    def generate_sequence(seq_len):
        return np.asarray([snap_boolean(np.append(np.random.rand(bits_per_vector), 1)) for _ in range(seq_len)] + [np.zeros(bits_per_vector+1) for _ in range(padded_seq_len - seq_len)])

    return np.asarray([generate_sequence(sequence_lengths[i]) for i in range(batch_size)]).astype(np.float32)