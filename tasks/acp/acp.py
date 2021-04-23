import os
import pickle
from time import perf_counter

from tensorboard.plugins.scalar.summary import pb as spb
from tensorboard.plugins.text.summary import pb as tpb
from tensorflow.contrib import layers
from tensorflow.python.ops.losses.losses_impl import Reduction
from datetime import datetime

import util
from model import Model
from tasks.acp.data import *
from tasks.parsing.parsers import DateParser, AmountParser, NoOpParser, OptionalParser


class AttendCopyParse(Model):
    tensorboard_dir = '/tmp/tensorboard'
    experiment = datetime.now()
    data_dir = 'tasks/acp/data/'
    splits_dir = 'tasks/acp/splits/'
    field = 'date'
    restore_all_path = None

    devices = util.get_devices()
    batch_size = 1 * len(devices)  # paper uses 32
    n_hid = 32
    frac_ce_loss = 0.0001
    lr = 3e-4
    keep_prob = 0.5

    noop_parser = NoOpParser()
    opt_noop_parser = OptionalParser(noop_parser, batch_size, 128, 103, 1)
    date_parser = DateParser(batch_size)
    amount_parser = AmountParser(batch_size)

    field_parsers = {
        'number': noop_parser,
        'order_id': opt_noop_parser,
        'date': date_parser,
        'total': amount_parser,
        'tla': amount_parser,
        'tta': amount_parser,
        'tp': amount_parser
    }

    def __init__(self):
        os.makedirs("./snapshots/acp", exist_ok=True)
        self.train = train = RealData("%strain.txt" % self.splits_dir, self.data_dir)
        self.train_iterator = self.iterator(train)
        self.next_train_batch = self.train_iterator.get_next()

        valid = RealData("%svalid.txt" % self.splits_dir, self.data_dir)
        self.valid_iterator = self.iterator(valid)
        self.next_valid_batch = self.valid_iterator.get_next()

        test = RealData("%stest.txt" % self.splits_dir, self.data_dir)
        self.test_iterator = self.iterator(test, n_repeat=1)
        self.next_test_batch = self.test_iterator.get_next()
        self.metadata_taken = False

        self.regularizer = layers.l2_regularizer(1e-4)

        print("Building graph...")
        config = tf.ConfigProto(allow_soft_placement=False)
        self.session = tf.Session(config=config)

        # Placeholders
        self.is_training_ph = tf.placeholder(tf.bool)
        self.memories_ph = tf.sparse_placeholder(tf.float32, name="memories")
        self.pixels_ph = tf.placeholder(tf.float32, name='pixels')
        self.word_indices_ph = tf.placeholder(tf.int32, name="word_indices")
        self.pattern_indices_ph = tf.placeholder(tf.int32, name="pattern_indices")
        self.char_indices_ph = tf.placeholder(tf.int32, name="char_indices")
        self.memory_mask_ph = tf.placeholder(tf.float32, name="memory_mask")
        self.parses_ph = tf.placeholder(tf.float32, name="parses")
        self.found_ph = tf.placeholder(tf.float32, name="found")
        # targets
        self.number_ph = tf.placeholder(tf.int32, name="number")
        self.order_id_ph = tf.placeholder(tf.int32, name="order_id")
        self.date_ph = tf.placeholder(tf.int32, name="date")
        self.total_ph = tf.placeholder(tf.int32, name="total")
        self.tla_ph = tf.placeholder(tf.int32, name="tla")
        self.tta_ph = tf.placeholder(tf.int32, name="tta")
        self.tp_ph = tf.placeholder(tf.int32, name="tp")

        self.targets = {
            'number': self.number_ph,
            'order_id': self.order_id_ph,
            'date': self.date_ph,
            'total': self.total_ph,
            'tla': self.tla_ph,
            'tta': self.tta_ph,
            'tp': self.tp_ph
        }

        h, w = train.im_size
        bs = self.batch_size
        seq_in = train.seq_in
        n_out = train.n_output

        field_idx = ['number', 'order_id', 'date', 'total', 'tla', 'tta', 'tp'].index(self.field)

        def dilated_block(x):
            return tf.concat([layers.conv2d(x, self.n_hid, 3, rate=rate, activation_fn=None, weights_regularizer=self.regularizer) for rate in [1, 2, 4, 8]], axis=3)

        def attend(pixels, word_indices, pattern_indices, char_indices, memory_mask, parses):
            """
            :param pixels: (bs, h, w)
            :param word_indices: (bs, h, w)
            :param pattern_indices: (bs, h, w)
            :param char_indices: (bs, h, w)
            :param parses: (bs, h, w, 4, 2)
            """
            bs = tf.shape(pixels)[0]

            X, Y = tf.meshgrid(tf.linspace(0.0, 1.0, RealData.im_size[0]), tf.linspace(0.0, 1.0, RealData.im_size[0]))
            X = tf.tile(X[None, ..., None], (bs, 1, 1, 1))
            Y = tf.tile(Y[None, ..., None], (bs, 1, 1, 1))

            word_embeddings = tf.reshape(layers.embed_sequence(tf.reshape(word_indices, (bs, -1)), vocab_size=train.word_hash_size, embed_dim=self.n_hid, unique=False, scope="word-embeddings"), (bs, h, w, self.n_hid))
            pattern_embeddings = tf.reshape(layers.embed_sequence(tf.reshape(pattern_indices, (bs, -1)), vocab_size=train.pattern_hash_size, embed_dim=self.n_hid, unique=False, scope="pattern-embeddings"), (bs, h, w, self.n_hid))
            char_embeddings = tf.reshape(layers.embed_sequence(tf.reshape(char_indices, (bs, -1)), vocab_size=train.n_output, embed_dim=self.n_hid, unique=False, scope="char-embeddings"), (bs, h, w, self.n_hid))

            pixels = tf.reshape(pixels, (bs, h, w, 3))
            parses = tf.reshape(parses, (bs, h, w, 8))
            memory_mask = tf.reshape(memory_mask, (bs, h, w, 1))
            x = tf.concat([pixels, word_embeddings, pattern_embeddings, char_embeddings, parses, X, Y, memory_mask], axis=3)

            with tf.variable_scope('attend'):
                # x = tf.nn.relu(dilated_block(x))
                for i in range(4):
                    x = tf.nn.relu(dilated_block(x))

                x = layers.dropout(x, self.keep_prob, is_training=self.is_training_ph)
                pre_att_logits = x
                att_logits = layers.conv2d(x, train.n_memories, 3, activation_fn=None, weights_regularizer=self.regularizer)  # (bs, h, w, n_memories)
                att_logits = memory_mask * att_logits - (1.0 - memory_mask) * 1000  # TODO only sum the memory_mask idx, in the softmax

                logits = tf.reshape(att_logits, (bs, -1))  # (bs, h * w * n_memories)
                logits -= tf.reduce_max(logits, axis=1, keepdims=True)
                lp = tf.nn.log_softmax(logits, axis=1)  # (bs, h * w * n_memories)
                p = tf.nn.softmax(logits, axis=1)  # (bs, h * w * n_memories)

                spatial_attention = tf.reshape(p, (bs, h * w * train.n_memories, 1, 1))  # (bs, h * w * n_memories, 1, 1)

                p_uniform = memory_mask / tf.reduce_sum(memory_mask, axis=(1, 2, 3), keepdims=True)
                cross_entropy_uniform = -tf.reduce_sum(p_uniform * tf.reshape(lp, (bs, h, w, train.n_memories)), axis=(1, 2, 3))  # (bs, 1)
                attention_entropy = -tf.reduce_sum(p * lp, axis=1) / tf.log(2.)  # (bs, 1)

                cp = tf.reduce_sum(tf.reshape(p, (bs, h, w, train.n_memories)), axis=3, keepdims=True)

                context = tf.reduce_sum(cp * pre_att_logits, axis=(1, 2))  # (bs, 4*n_hidden)

            return spatial_attention, attention_entropy, cross_entropy_uniform, context

        spatial_attention, attention_entropy, cross_entropy_uniform, context = util.batch_parallel(
            attend,
            self.devices,
            pixels=self.pixels_ph,
            word_indices=self.word_indices_ph,
            pattern_indices=self.pattern_indices_ph,
            char_indices=self.char_indices_ph,
            memory_mask=self.memory_mask_ph,
            parses=self.parses_ph
        )

        context = tf.concat(context, axis=0)  # (bs, 128)
        spatial_attention = tf.concat(spatial_attention, axis=0)  # (bs, h * w * n_mem, 1, 1)
        cross_entropy_uniform = tf.concat(cross_entropy_uniform, axis=0)  # (bs, 1)
        attention_entropy = tf.reduce_mean(tf.concat(attention_entropy, axis=0), axis=0)  # (1,)

        with tf.variable_scope('copy'):
            memories = tf.sparse_reshape(self.memories_ph, (self.batch_size, h * w * train.n_memories, train.seq_in, n_out))
            x = tf.reshape(tf.sparse_reduce_sum(spatial_attention * memories, axis=1), (bs, seq_in, n_out))  # (bs, seq_in, n_out)

        with tf.name_scope('parse'):
            parser = self.field_parsers[self.field]
            parsed = parser.parse(x, context, self.is_training_ph)
            target = self.targets[self.field]
            output = self.output(parsed, targets=target, scope=self.field)
            self.outputs = {self.field: output}

        reg_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar("loss/regularization", reg_loss)

        cross_entropy_uniform_loss = self.frac_ce_loss * tf.reduce_mean(cross_entropy_uniform)
        tf.summary.scalar("loss/ce_uniform", cross_entropy_uniform_loss)

        field_loss = tf.reduce_mean(self.outputs[self.field]['cross_entropy'])  # (bs, )
        tf.summary.scalar('loss/field', field_loss)

        self.loss = field_loss + reg_loss + cross_entropy_uniform_loss
        tf.summary.scalar('loss/total', self.loss)

        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads, vars = zip(*self.optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True))
            self.train_step = self.optimizer.apply_gradients(zip(grads, vars), global_step=self.global_step)

        # Savers
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())

        if self.restore_all_path:
            print("Restoring all " + self.restore_all_path + "...")
            self.saver.restore(self.session, self.restore_all_path)
        else:
            restore = parser.restore()
            if restore is not None:
                scope, fname = restore
                vars = tf.trainable_variables(scope=scope)
                saver = tf.train.Saver(var_list=vars)
                print("Restoring %s parser %s..." % (self.field, fname))
                for var in vars:
                    print("-- restoring %s" % var)
                saver.restore(self.session, fname)

        # Summaries
        sa = tf.reshape(spatial_attention, (self.batch_size, h, w, train.n_memories))

        # (h, w, in, out)
        color_kernel = tf.constant([[255, 0, 0],  # red
                                    [0, 255, 0],  # green
                                    [0, 0, 255],  # blue
                                    [255, 255, 255]],  # white
                                   dtype=tf.float32)
        color_kernel = tf.reshape(color_kernel, (1, 1, 4, 3))

        att = tf.nn.conv2d(sa, color_kernel, [1, 1, 1, 1], "SAME")
        tf.summary.image("%s/attention" % self.field, att, max_outputs=1)
        tf.summary.scalar("%s/attention_entropy" % self.field, attention_entropy)

        tf.summary.scalar("p_zero", tf.reduce_mean(tf.to_float(tf.equal(spatial_attention, 0.0))))

        tf.summary.scalar('%s/label_cross_entropy' % self.field, tf.reduce_mean(self.outputs[self.field]['cross_entropy']))
        acc = tf.reduce_mean(self.outputs[self.field]['correct'])
        tf.summary.scalar('%s/acc' % self.field, acc)
        self.found = tf.reduce_mean(self.found_ph[:, field_idx])
        tf.summary.scalar('%s/found' % self.field, self.found)

        self.train_writer = tf.summary.FileWriter(self.tensorboard_dir + '/attend-copy-parse/%s/%s/train' % (self.field, self.experiment), self.session.graph)
        self.test_writer = tf.summary.FileWriter(self.tensorboard_dir + '/attend-copy-parse/%s/%s/test' % (self.field, self.experiment), self.session.graph)
        self.summaries = tf.summary.merge_all()

    def output(self, logits, targets, scope, optional=None):
        with tf.variable_scope(scope):
            if optional:
                logoutput_p, empty_answer = optional
                output_p = tf.exp(logoutput_p)
                output_p = tf.reshape(output_p, (self.batch_size, 1, 1))
                empty_logits = tf.exp(tf.get_variable('empty-multiplier', shape=(), dtype=tf.float32, initializer=tf.initializers.constant(0.0))) * empty_answer
                logits = output_p * logits + (1 - output_p) * empty_logits

            mask = tf.logical_not(tf.equal(targets, self.train.pad_idx))  # (bs, seq)
            label_cross_entropy = tf.reduce_sum(tf.losses.sparse_softmax_cross_entropy(targets, logits, reduction=Reduction.NONE) * tf.to_float(mask), axis=1) / tf.reduce_sum(tf.to_float(mask), axis=1)

            chars = tf.argmax(logits, axis=2, output_type=tf.int32)
            equal = tf.equal(targets, chars)
            correct = tf.to_float(tf.reduce_all(tf.logical_or(equal, tf.logical_not(mask)), axis=1))

            return {'cross_entropy': label_cross_entropy, 'actual': chars, 'targets': targets, 'correct': correct}

    def iterator(self, data, n_repeat=-1):
        shapes, types = data.shapes_types()
        ds = tf.data.Dataset.from_generator(
            data.sample_generator,
            types,
            shapes
        ).map(lambda i, v, s, *args: (tf.SparseTensor(i, v, s),) + args) \
            .repeat(n_repeat)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        return ds.prefetch(2) \
            .make_one_shot_iterator()

    def train_batch(self):
        step = self.session.run(self.global_step)
        batch = self.session.run(self.next_train_batch)
        placeholders = self.get_placeholders(batch, True)

        if not self.metadata_taken and step > 1000:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, loss, summaries, outputs, step = self.session.run([self.train_step, self.loss, self.summaries, self.outputs, self.global_step], placeholders, options=run_options, run_metadata=run_metadata)
            self.train_writer.add_run_metadata(run_metadata, "step%d" % step)
            self.write_summaries(self.train_writer, summaries, outputs, step)
            self.metadata_taken = True
        elif step % 100 == 0:
            _, loss, summaries, outputs, step = self.session.run([self.train_step, self.loss, self.summaries, self.outputs, self.global_step], placeholders)
            self.write_summaries(self.train_writer, summaries, outputs, step)
        else:
            start = perf_counter()
            _, loss = self.session.run([self.train_step, self.loss], placeholders)
            took = perf_counter() - start
            self.train_writer.add_summary(spb('dps', self.batch_size / took), step)

        return loss

    def val_batch(self):
        batch = self.session.run(self.next_valid_batch)
        placeholders = self.get_placeholders(batch, False)
        loss, summaries, outputs, step = self.session.run([self.loss, self.summaries, self.outputs, self.global_step], placeholders)
        self.write_summaries(self.test_writer, summaries, outputs, step)
        return loss

    def test_set(self):
        outputs = []
        founds = []
        while True:
            try:
                batch = self.session.run(self.next_test_batch)
                placeholders = self.get_placeholders(batch, False)
                output, found = self.session.run([self.outputs, self.found], placeholders)
                outputs.append(output)
                founds.append(found)
            except tf.errors.OutOfRangeError:
                break

        with open('outputs.pkl', 'wb') as fp:
            pickle.dump({'outputs': outputs, 'founds': founds}, fp)

        corrects = {}
        for output in outputs:
            for k, v in output.items():
                if k not in corrects:
                    corrects[k] = []
                corrects[k].append(v['correct'])

        for k, v in corrects.items():
            print("%s: %f" % (k, np.mean(v)))

        print("found: %f" % np.mean(founds))

    def save(self, name):
        self.saver.save(self.session, "./snapshots/acp/%s/%s" % (self.field, name))

    def load(self, name):
        self.saver.restore(self.session, name)

    def write_summaries(self, writer, summaries, outputs, step):
        writer.add_summary(summaries, step)

        for k, v in outputs.items():
            actual = self.train.array_to_str(v['actual'])[0]
            target = self.train.array_to_str(v['targets'])[0]
            writer.add_summary(tpb("%s/sample" % k, "target: %s actual: %s" % (target, actual)), step)

        writer.flush()

    def get_placeholders(self, batch, is_training):
        memories, pixels, word_indices, pattern_indices, char_indices, memory_mask, parses, number, order_id, date, total, tla, tta, tp, found = batch
        return {
            self.is_training_ph: is_training,
            self.memories_ph: memories,
            self.pixels_ph: pixels,
            self.word_indices_ph: word_indices,
            self.pattern_indices_ph: pattern_indices,
            self.char_indices_ph: char_indices,
            self.memory_mask_ph: memory_mask,
            self.parses_ph: parses,
            self.number_ph: number,
            self.order_id_ph: order_id,
            self.date_ph: date,
            self.total_ph: total,
            self.tla_ph: tla,
            self.tta_ph: tta,
            self.tp_ph: tp,
            self.found_ph: found
        }


if __name__ == '__main__':
    AttendCopyParse.data_dir = 'data/'
    AttendCopyParse.splits_dir = 'splits/'
    m = AttendCopyParse()
    print("Running train_batch...")
    m.train_batch()
    print("Running val_batch...")
    m.val_batch()
    print("Running test_set...")
    m.test_set()
