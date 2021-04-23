import os
import util

from time import perf_counter
from datetime import datetime

from tensorboard.plugins.scalar.summary import pb as spb
from tensorboard.plugins.text.summary import pb as tpb
from tensorflow.python.ops.losses.losses_impl import Reduction

from model import Model
from tasks.parsing.parsers import *
from tasks.parsing.data import *


class Parser(Model):
    devices = util.get_devices()
    batch_size = 128 * len(devices)
    context_size = 128
    experiment = datetime.now()
    type = 'dates'  # valid are ['dates', 'amounts']
    output_length = {"dates": RealData.seq_date, "amounts": RealData.seq_amount}[type]
    continue_from = None

    def __init__(self):
        self.train = train = TabSeparated('tasks/parsing/data/%s/train.tsv' % self.type, self.output_length)
        self.train_iterator = self.iterator(train)

        valid = TabSeparated('tasks/parsing/data/%s/valid.tsv' % self.type, self.output_length)
        self.valid_iterator = self.iterator(valid)

        parser = {'amounts': AmountParser(self.batch_size), 'dates': DateParser(self.batch_size)}[self.type]

        print("Building graph...")
        config = tf.ConfigProto(allow_soft_placement=False)
        self.session = tf.Session(config=config)
        self.is_training_ph = tf.placeholder(tf.bool)

        source, self.targets = tf.cond(
            self.is_training_ph,
            true_fn=lambda: self.train_iterator.get_next(),
            false_fn=lambda: self.valid_iterator.get_next()
        )
        self.sources = source

        oh_inputs = tf.one_hot(source, train.n_output)  # (bs, seq, n_out)

        context = tf.zeros(
            (self.batch_size, self.context_size),
            dtype=tf.float32,
            name=None
        )

        output_logits = parser.parse(oh_inputs, context, self.is_training_ph)

        with tf.variable_scope('loss'):
            mask = tf.logical_not(tf.equal(self.targets, train.pad_idx))
            label_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.targets, output_logits, reduction=Reduction.NONE) * tf.to_float(mask)) / tf.log(2.)

            chars = tf.argmax(output_logits, axis=2, output_type=tf.int32)
            equal = tf.equal(self.targets, chars)
            acc = tf.reduce_mean(tf.to_float(tf.reduce_all(tf.logical_or(equal, tf.logical_not(mask)), axis=1)))

        self.actual = chars
        self.loss = label_cross_entropy

        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step, colocate_gradients_with_ops=True)

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        util.print_vars(tf.trainable_variables())

        if self.continue_from:
            print("Restoring " + self.continue_from + "...")
            self.saver.restore(self.session, self.continue_from)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('label cross entropy', label_cross_entropy)
        tf.summary.scalar('acc', acc)

        tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
        self.train_writer = tf.summary.FileWriter(tensorboard_dir + '/parse/%s/%s/train' % (self.type, self.experiment), self.session.graph)
        self.test_writer = tf.summary.FileWriter(tensorboard_dir + '/parse/%s/%s/test' % (self.type, self.experiment), self.session.graph)
        self.summaries = tf.summary.merge_all()

    def iterator(self, data):
        return tf.data.Dataset.from_generator(
            data.sample_generator,
            data.types(),
            data.shapes()
        ).repeat(-1).batch(self.batch_size).prefetch(16).make_one_shot_iterator()

    def train_batch(self):
        step = self.session.run(self.global_step)

        if step % 1000 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, loss, summaries, sources, actual, targets, step = self.session.run([self.train_step, self.loss, self.summaries, self.sources, self.actual, self.targets, self.global_step], {self.is_training_ph: True}, options=run_options, run_metadata=run_metadata)
            self.train_writer.add_run_metadata(run_metadata, "step%d" % step)
            self.write_summaries(self.train_writer, summaries, sources, actual, targets, step)
        elif step % 100 == 0:
            _, loss, summaries, sources, actual, targets, step = self.session.run([self.train_step, self.loss, self.summaries, self.sources, self.actual, self.targets, self.global_step], {self.is_training_ph: True})
            self.write_summaries(self.train_writer, summaries, sources, actual, targets, step)
        else:
            start = perf_counter()
            _, loss = self.session.run([self.train_step, self.loss], {self.is_training_ph: True})
            took = perf_counter() - start
            self.train_writer.add_summary(spb('dps', self.batch_size / took), step)

        return loss

    def val_batch(self):
        loss, summaries, sources, actual, targets, step = self.session.run([self.loss, self.summaries, self.sources, self.actual, self.targets, self.global_step], {self.is_training_ph: False})
        self.write_summaries(self.test_writer, summaries, sources, actual, targets, step)
        return loss

    def save(self, name):
        self.saver.save(self.session, "./snapshots/%s/%s" % (self.type, name))

    def load(self, name):
        self.saver.restore(self.session, name)

    def write_summaries(self, writer, summaries, sources, actuals, targets, step):
        writer.add_summary(summaries, step)

        sources = self.train.array_to_str(sources)
        actuals = self.train.array_to_str(actuals)
        targets = self.train.array_to_str(targets)
        writer.add_summary(tpb("sample", "source: %s, target: %s actual %s" % (sources[0], targets[0], actuals[0])), step)
        for s, t, a in zip(sources, targets, actuals):
            if a != t:
                writer.add_summary(tpb("errors", "source: %s target: %s actual %s" % (s, t, a)), step)

        writer.flush()


if __name__ == '__main__':
    m = Parser()
    m.train_batch()
    m.val_batch()
