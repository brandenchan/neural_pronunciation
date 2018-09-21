import os
import shutil
from pprint import pprint

import tensorflow as tf
try:
    from tensorflow.nn.rnn_cell import LSTMCell, LSTMStateTuple
except ModuleNotFoundError:
    from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import numpy as np


from data_handling import (convert_list, joint_iterator_from_file, load_maps,
                           load_symbols, convert_word)
from evaluation import evaluate


class CharToPhonModel:
    """ Class that handles training and inference of LSTM model """
    def __init__(self,
                data_dir="data/",
                batch_size=128,
                embed_dims=300,
                hidden_dims=300,
                bidir=True,
                cell_class=LSTMCell,
                max_gradient_norm=1,
                learning_rate=0.001,
                save_dir="output_1/",
                resume_dir=None,
                n_batches=10001,
                debug=False,
                sample_size=20,
                print_every=50,
                validate_every=500
                ):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_chars = len(load_symbols(data_dir + "chars"))
        self.n_arpa = len(load_symbols(data_dir + "arpabet"))
        _, self.code_to_chars = load_maps(data_dir + "chars")
        _, self.code_to_arpa = load_maps(data_dir + "arpabet")
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims
        self.bidir = bidir
        self.cell_class=LSTMCell
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.resume_dir = resume_dir
        self.n_batches = n_batches
        self.debug = debug
        self.sample_size = sample_size
        self.print_every = print_every
        self.validate_every = validate_every
        self.save_every = validate_every


    def build_graph(self):
        # tf.reset_default_graph()
        self.setup_variables()
        self.build_encoder()
        self.build_decoder()
        self.compute_loss()
        self.gradient_update()


    def setup_variables(self):
        with tf.variable_scope("variables"):

            # inputs of shape [encoder_max_time, batch_size]
            self.encoder_inputs = tf.placeholder(shape=(None, None),
                                                dtype=tf.int32,
                                                name="encoder_inputs")
            self.encoder_inputs = tf.Print(self.encoder_inputs, [self.encoder_inputs])

            # inputs of shape [decoder_max_time, batch_size]
            self.decoder_inputs = tf.placeholder(shape=(None, None),
                                                dtype=tf.int32,
                                                name="decoder_inputs")

            # targets of shape [decoder_max_time - 1, batch_size]
            self.decoder_targets = tf.placeholder(shape=(None, None),
                                                dtype=tf.int32,
                                                name="decoder_targets")

            # sequence lengths of shape [batch_size]
            self.encoder_input_lengths = tf.placeholder(shape=(None,),
                                                        dtype=tf.int32,
                                                        name='encoder_inputs_lengths')

            # sequence lengths of shape [batch_size]
            self.decoder_input_lengths = tf.placeholder(shape=(None,),
                                                        dtype=tf.int32,
                                                        name='decoder_input_lengths')

            # sequence lengths of shape [batch_size]
            self.decoder_target_lengths = tf.placeholder(shape=(None,),
                                                        dtype=tf.int32,
                                                        name='decoder_target_lengths')

            self.char_embeddings = tf.Variable(tf.random_uniform((self.n_chars, self.embed_dims), -1.0, 1.0),
                                            name="char_embeddings")

            self.arpa_embeddings = tf.Variable(tf.random_uniform((self.n_arpa, self.embed_dims), -1.0, 1.0),
                                            name="arpa_embeddings")

            self.encoder_input_embeddings = tf.nn.embedding_lookup(self.char_embeddings, self.encoder_inputs)

            self.decoder_input_embeddings = tf.nn.embedding_lookup(self.arpa_embeddings, self.decoder_inputs)

            self.decoder_target_embeddings = tf.nn.embedding_lookup(self.arpa_embeddings, self.decoder_targets)

    def build_encoder(self):
        with tf.variable_scope("encoder"):

            # Unidirectional Run
            if not self.bidir:
                encoder_cell = self.cell_class(self.hidden_dims)
                encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                                            encoder_cell, self.encoder_input_embeddings,
                                            dtype=tf.float32, time_major=True)
                del encoder_outputs

            # Bidirectional Run
            else:
                with tf.variable_scope("fw"):
                    fw_encoder_cell = self.cell_class(self.hidden_dims)
                with tf.variable_scope("bw"):
                    bw_encoder_cell = self.cell_class(self.hidden_dims)

                ((encoder_fw_outputs, encoder_bw_outputs),
                (encoder_fw_final_state, encoder_bw_final_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_encoder_cell,
                                                                                                    cell_bw=bw_encoder_cell,
                                                                                                    inputs=self.encoder_input_embeddings,
                                                                                                    sequence_length=self.encoder_input_lengths,
                                                                                                    dtype=tf.float32, time_major=True))

                # Concat final states of forward and backward run
                encoder_final_state_c = tf.concat(
                    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

                encoder_final_state_h = tf.concat(
                    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

                self.encoder_final_state = LSTMStateTuple(
                    c=encoder_final_state_c,
                    h=encoder_final_state_h
                )

                del encoder_fw_outputs
                del encoder_bw_outputs

    def build_decoder(self):
        with tf.variable_scope("decoder"):

            with tf.variable_scope("projection"):
                self.projection_layer = tf.layers.Dense(
                                            self.n_arpa, use_bias=False)

            decoder_dims = self.hidden_dims
            if self.bidir:
                decoder_dims *= 2
            decoder_cell = self.cell_class(decoder_dims)

            helper = tf.contrib.seq2seq.TrainingHelper(
                            inputs=self.decoder_input_embeddings, 
                            sequence_length=self.decoder_input_lengths,
                            time_major=True)

            my_decoder = tf.contrib.seq2seq.BasicDecoder(
                            decoder_cell,
                            helper,
                            self.encoder_final_state,
                            output_layer=self.projection_layer)

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                                                    my_decoder,
                                                    output_time_major=True,
                                                    impute_finished=True)

            self.logits = outputs.rnn_output

            # Transposed so that not time major
            self.predictions_arpa = tf.transpose(tf.argmax(self.logits, 2))

    def compute_loss(self):
        with tf.variable_scope("loss_computation"):
            self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        labels=self.decoder_targets, logits=self.logits)

            self.sequence_mask = tf.sequence_mask(
                                    self.decoder_target_lengths, name="sequence_mask", dtype=tf.float32)
            self.sequence_mask = tf.transpose(self.sequence_mask)

            self.num_predictions = tf.count_nonzero(self.sequence_mask, dtype=tf.float32)

            self.losses = tf.reduce_mean(self.crossent * self.sequence_mask, axis=0)

            self.batch_loss = tf.reduce_mean(self.losses)

    def gradient_update(self):
        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(self.batch_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                                gradients, self.max_gradient_norm)

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params))


    def train(self):
        # Load data
        train_file = self.data_dir + "train"
        dev_file = self.data_dir + "dev"
        if self.debug:
            train_file += "_debug"
            dev_file += "_debug"
        self.iter_train = joint_iterator_from_file(train_file, auto_reset=True)
        self.iter_dev = joint_iterator_from_file(dev_file, auto_reset=False)
        self.n_train_data = self.iter_train.len

        # Sample of data (time_major=True)
        self.sample = self.iter_train.next(self.sample_size)
        print_sample(self.sample, self.code_to_chars, self.code_to_arpa)

        n_to_process = self.n_batches * self.batch_size
        print("\nTRAINING\n" + "="*20)
        if self.bidir:
            print("\tBidirectional encoder")
        else:
            print("\tUnidirectional encoder")
        if self.debug:
            print("\tDEBUG MODE")
        print("\tTraining {} examples over {} batches of {} ({} epochs)\n".format(self.n_train_data,
                                                                              self.n_batches,
                                                                              self.batch_size,
                                                                              int(n_to_process / self.n_train_data)))

        with tf.Session() as sess:
            # Setup of output directory
            check_save_dir(self.save_dir, self.resume_dir)
            save_hyperparams(self)

            # Build model and initialize variables
            self.build_graph()

            self.saver = tf.train.Saver()
            if self.resume_dir:
                self.saver.restore(sess, self.resume_dir + "model.ckpt")
            else:
                sess.run(tf.global_variables_initializer())

            self.train_loop(sess)


    def train_loop(self, sess):
        completed_batches = 0
        train_loss_track = []
        dev_loss_track = []
        dev_accuracy_track = []
        dev_similarity_track = []

        for completed_batches in range(self.n_batches):

            # Get batch of data and perform training
            X, Y_in, Y_targ, X_lens, Y_lens = self.iter_train.next(self.batch_size)

            fd = {self.encoder_inputs: X,
                    self.decoder_inputs: Y_in,
                    self.decoder_targets: Y_targ,
                    self.encoder_input_lengths: X_lens,
                    self.decoder_input_lengths: Y_lens,
                    self.decoder_target_lengths: Y_lens}

            _, loss = sess.run([self.train_op, self.batch_loss], fd)
            train_loss_track.append(loss)

            # Printing, validation and saving
            if completed_batches != 0:
                epoch = (self.batch_size * completed_batches) // self.n_train_data
                if completed_batches % self.print_every == 0:
                    t_loss = np.mean(train_loss_track[-100:])
                    print("Batch {} / {} Epoch {} train: {}".format(completed_batches, self.n_batches, epoch, t_loss))
                if completed_batches % self.validate_every == 0:
                    self.sample_inference(sess)
                    dev_loss, dev_accuracy, dev_similarity = self.dev_validation(sess)
                    dev_loss_track.append(dev_loss)
                    dev_accuracy_track.append(dev_accuracy)
                    dev_similarity_track.append(dev_similarity)
                if completed_batches % self.save_every == 0:
                    save_path = self.saver.save(sess, self.save_dir + "model.ckpt")
                    print("Model saved in path: {}\n".format(save_path))

            # every X amount of batches: save train and dev loss, save check point, do dev

    def sample_inference(self, sess):
        X, Y_in, Y_targ, X_lens, Y_lens = self.sample
        fd = {self.encoder_inputs: X,
                self.decoder_inputs: Y_in,
                self.decoder_targets: Y_targ,
                self.encoder_input_lengths: X_lens,
                self.decoder_input_lengths: Y_lens,
                self.decoder_target_lengths: Y_lens}
        prediction, = sess.run([self.predictions_arpa], fd)
        print_prediction(X.T, prediction, self.code_to_chars, self.code_to_arpa)

    def dev_validation(self, sess):
        print()
        print("DEV VALIDATION")
        print("=" * 20)
        print()
        self.iter_dev.reset()
        all_losses = []
        all_sim = []
        while True:
            batch = self.iter_dev.next(self.batch_size)
            
            if not batch:
                # i.e. at the end of the dev epoch
                break

            X, Y_in, Y_targ, X_lens, Y_lens = batch

            fd = {self.encoder_inputs: X,
                    self.decoder_inputs: Y_in,
                    self.decoder_targets: Y_targ,
                    self.encoder_input_lengths: X_lens,
                    self.decoder_input_lengths: Y_lens,
                    self.decoder_target_lengths: Y_lens}
            losses, prediction = sess.run([self.losses, self.predictions_arpa], fd)
            similarity_scores = evaluate(Y_targ.T, prediction)

            all_sim += similarity_scores
            all_losses += list(losses)
        dev_loss = np.mean(all_losses)
        accuracy, similarity = dev_stats(all_sim)
        print("Loss:       {}".format(dev_loss))
        print("Accuracy:   {}".format(accuracy))
        print("Similarity: {}".format(similarity))
        print()
        return dev_loss, accuracy, similarity

def save_hyperparams(model, filename="hyperparameters.txt"):
    hyperparams = vars(model)
    with open(model.save_dir + filename, "w") as output:
        for hp in hyperparams:
            output.write("{}: {}\n".format(hp, hyperparams[hp]))

def check_save_dir(save_dir, resume_dir):
    if os.path.exists(save_dir):
        assert save_dir != resume_dir
        decision = input("\nDirectory {} already exists. Overwrite? [y/n]: ".format(save_dir))
        while True:
            if decision == "y":
                shutil.rmtree(save_dir)
                os.mkdir(save_dir)
                break
            elif decision == "n":
                raise Exception
    else:
        os.mkdir(save_dir)

def print_sample(sample, code_to_chars, code_to_arpa):
    print("\nSAMPLE OF DATA\n" + "="*20 + "\n")

    X, Y_in, Y_targ, len_X, len_Y = sample
    X = convert_list(X.T, code_to_chars)
    Y_in = convert_list(Y_in.T, code_to_arpa)
    Y_targ = convert_list(Y_targ.T, code_to_arpa)
    for i in range(len(X)):
        print("Example {}".format(i + 1))
        print("\tX: " + str(X[i]))
        print("\tY: " + str(Y_in[i]))
        print("\tY: " + str(Y_targ[i]))
        print("\tlen_x: " + str(len_X[i]))
        print("\tlen_y: " + str(len_Y[i]))
        print()
        print()
    print()

def print_prediction(X, prediction, code_to_chars, code_to_arpa):
    print()
    print("SAMPLE INFERENCE")
    print("=" * 20)
    print()
    zipped = zip(X, prediction)
    for s, p in zipped:
        spelling_raw = convert_word(s, code_to_chars)
        spelling = "".join([ch for ch in spelling_raw if "<" not in ch])
        arpa_raw = convert_word(p, code_to_arpa)
        arpa = " ".join([ar for ar in arpa_raw if "<" not in ar])
        print("{} - {}".format(spelling, arpa))
    print()

def dev_stats(similarity_scores):
    accuracy = similarity_scores.count(1) / len(similarity_scores)
    sim = np.mean(similarity_scores)
    return accuracy, sim

if __name__ == "__main__":
    CharToPhonModel().train()
