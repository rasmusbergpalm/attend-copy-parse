"""
with tf.name_scope('totals'):

    permutations = tf.constant([[
        # 4 choose 4
        [1., 1., 1., 1.],  # A,B,C,D, 0
        # 4 choose 3
        [1., 1., 1., 0.],  # A,B,C, 1
        [1., 1., 0., 1.],  # A,B,D, 2
        [1., 0., 1., 1.],  # A,C,D, 3
        [0., 1., 1., 1.],  # B,C,D, 4
        # 4 choose 2
        [0., 0., 1., 1.],  # A,B, 5
        [0., 1., 0., 1.],  # A,C, 6
        [0., 1., 1., 0.],  # A,D, 7
        [1., 0., 0., 1.],  # B,C, 8
        [1., 0., 1., 0.],  # B,D, 9
        [1., 1., 0., 0.],  # C,D, 10
    ]])  # (1, 11, 4)

    total_fields = ['total', 'tla', 'tta', 'tp']

    logp_outputs = tf.stack([logoutput_p[:, i] for i in range(3, 7)], axis=1)  # (bs, 4)
    logp_outputs = tf.expand_dims(logp_outputs, axis=1)  # (bs, 1, 4)

    log1m_p_outputs = tf.stack([log1m_output_p[:, i] for i in range(3, 7)], axis=1)  # (bs, 4)
    log1m_p_outputs = tf.expand_dims(log1m_p_outputs, axis=1)  # (bs, 1, 4)

    logp_permutations = tf.reduce_sum(permutations * logp_outputs + (1 - permutations) * log1m_p_outputs, axis=2)  # (bs, 11)

    logp_correct = tf.stack([-self.outputs[k]['cross_entropy'] for k in total_fields], axis=1)  # (bs,4)
    logp_correct = tf.expand_dims(logp_correct, axis=1)  # (bs, 1, 4)

    logp_permutations_correct = tf.reduce_sum(logp_correct * permutations, axis=2)  # (bs, 11)

    logp_all_correct = logp_permutations + logp_permutations_correct  # (bs, 11)
    logp_any_correct = tf.reduce_logsumexp(logp_all_correct, axis=1)  # (bs,)

    totals_loss = -logp_any_correct  # (bs,)

    # Acc calculations

    raw_corrects = tf.stack([self.outputs[k]['correct'] for k in total_fields], axis=1)  # (bs, 4)
    raw_corrects = tf.expand_dims(raw_corrects, axis=1)  # (bs, 1, 4)

    all_outputted_correct = tf.reduce_prod(raw_corrects ** permutations, axis=2)  # (bs, 11)
    all_outputted_correct = tf.expand_dims(all_outputted_correct, axis=2)  # (bs, 11, 1)

    corrects = permutations * raw_corrects + (1 - permutations) * all_outputted_correct  # (bs, 11, 4)

    chosen_permutation = tf.stack([tf.range(0, self.batch_size, dtype=tf.int64), tf.argmax(logp_permutations, axis=1)], axis=1)  # (bs, 2)
    chosen_corrects = tf.gather_nd(corrects, chosen_permutation)  # (bs, 4)
    for i, k in enumerate(total_fields):
        self.outputs[k]['correct'] = chosen_corrects[:, i]
"""
