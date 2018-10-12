import os
import shutil
from pprint import pprint
import pickle

import tensorflow as tf
try:
    from tensorflow.nn.rnn_cell import LSTMCell, LSTMStateTuple
except ModuleNotFoundError:
    from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import numpy as np


from data_handling import (convert_list, joint_iterator_from_file, load_maps,
                           load_symbols, convert_word)
from evaluation import dev_stats, evaluate
from graph import create_graph

PAD_CODE = 0
START_CODE = 1
END_CODE = 2

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
                save_dir="unsaved_model/",
                resume_dir=None,
                n_batches=10001,
                debug=False,
                print_every=50,
                validate_every=500,
                initializer = tf.glorot_normal_initializer
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
        self.print_every = print_every
        self.save_every = validate_every
        self.initializer = tf.glorot_normal_initializer

        sample_file = self.data_dir + "sample"
        self.iter_sample = joint_iterator_from_file(sample_file, auto_reset=False)


    def build_graph(self):
        tf.reset_default_graph()
        tf.get_variable_scope().set_initializer(self.initializer())

        placeholders = self.setup_placeholders()

        encoder_final_state = self.build_encoder(placeholders["encoder_inputs"],
                                                placeholders["encoder_input_lengths"])

        logits, predictions_arpa = self.build_decoder(encoder_final_state, 
                                                        placeholders["decoder_inputs"],
                                                        placeholders["decoder_targets"],
                                                        placeholders["decoder_lengths"],
                                                        placeholders["encoder_input_lengths"])

        if self.mode == "inference":

            output_nodes = {"predictions_arpa": predictions_arpa}

        if self.mode == "train":

            losses, batch_loss = self.compute_loss(logits,
                                                    placeholders["decoder_targets"],
                                                    placeholders["decoder_lengths"])
            train_op = self.gradient_update(batch_loss)
            output_nodes = {"train_op": train_op, 
                            "batch_loss": batch_loss, 
                            "losses": losses,
                            "predictions_arpa": predictions_arpa}

        return placeholders, output_nodes


    def setup_placeholders(self):
        with tf.variable_scope("variables"):

            # inputs of shape [encoder_max_time, batch_size]
            encoder_inputs = tf.placeholder(shape=(None, None),
                                                dtype=tf.int32,
                                                name="encoder_inputs")

            # inputs of shape [decoder_max_time, batch_size]
            decoder_inputs = tf.placeholder(shape=(None, None),
                                                dtype=tf.int32,
                                                name="decoder_inputs")

            # targets of shape [decoder_max_time - 1, batch_size]
            decoder_targets = tf.placeholder(shape=(None, None),
                                                dtype=tf.int32,
                                                name="decoder_targets")

            # sequence lengths of shape [batch_size]
            encoder_input_lengths = tf.placeholder(shape=(None,),
                                                        dtype=tf.int32,
                                                        name='encoder_inputs_lengths')

            # sequence lengths of shape [batch_size]
            decoder_lengths = tf.placeholder(shape=(None,),
                                             dtype=tf.int32,
                                             name='decoder_lengths')
 
            return {"encoder_inputs": encoder_inputs, 
                    "encoder_input_lengths": encoder_input_lengths,
                    "decoder_inputs": decoder_inputs,
                    "decoder_targets": decoder_targets,
                    "decoder_lengths": decoder_lengths}
 


    def build_encoder(self, encoder_inputs, encoder_input_lengths):

        with tf.variable_scope("encoder"):

            char_embeddings = tf.Variable(tf.random_uniform((self.n_chars, self.embed_dims), -1.0, 1.0),
                                            name="char_embeddings")

            encoder_input_embeddings = tf.nn.embedding_lookup(char_embeddings, encoder_inputs)

            # Unidirectional Run
            if not self.bidir:
                encoder_cell = self.cell_class(self.hidden_dims)
                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                                            encoder_cell, encoder_input_embeddings,
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
                                                                                                    inputs=encoder_input_embeddings,
                                                                                                    sequence_length=encoder_input_lengths,
                                                                                                    dtype=tf.float32, time_major=True))

                # Concat final states of forward and backward run
                encoder_final_state_c = tf.concat(
                    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

                encoder_final_state_h = tf.concat(
                    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

                encoder_final_state = LSTMStateTuple(
                    c=encoder_final_state_c,
                    h=encoder_final_state_h
                )

                del encoder_fw_outputs
                del encoder_bw_outputs

            return encoder_final_state


    def build_decoder(self, encoder_final_state,
                      decoder_inputs, decoder_targets,
                      decoder_lengths, encoder_input_lengths):

        with tf.variable_scope("decoder"):

            arpa_embeddings = tf.Variable(tf.random_uniform((self.n_arpa, self.embed_dims), -1.0, 1.0),
                                            name="arpa_embeddings")

            decoder_input_embeddings = tf.nn.embedding_lookup(arpa_embeddings, decoder_inputs)

            with tf.variable_scope("projection"):
                projection_layer = tf.layers.Dense(
                                            self.n_arpa, use_bias=False)

            decoder_dims = self.hidden_dims
            if self.bidir:
                decoder_dims *= 2
            decoder_cell = self.cell_class(decoder_dims)

            if self.mode == "train":
                helper = tf.contrib.seq2seq.TrainingHelper(
                                inputs=decoder_input_embeddings, 
                                sequence_length=decoder_lengths,
                                time_major=True)
            elif self.mode == "inference":
                start_tokens = tf.fill([self.batch_size], START_CODE)
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                            arpa_embeddings,
                            start_tokens,
                            END_CODE)

            my_decoder = tf.contrib.seq2seq.BasicDecoder(
                            decoder_cell,
                            helper,
                            encoder_final_state,
                            output_layer=projection_layer)

            maximum_iterations = tf.round(tf.reduce_max(encoder_input_lengths) * 2)

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                                                    my_decoder,
                                                    output_time_major=True,
                                                    impute_finished=True,
                                                    maximum_iterations=maximum_iterations)

            logits = outputs.rnn_output

            # Transposed so that not time major
            predictions_arpa = tf.transpose(tf.argmax(logits, 2))

            return logits, predictions_arpa


    def compute_loss(self, logits, decoder_targets, decoder_lengths):

        with tf.variable_scope("loss_computation"):

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        labels=decoder_targets, logits=logits)

            sequence_mask = tf.sequence_mask(decoder_lengths,
                                                name="sequence_mask",
                                                dtype=tf.float32)
            sequence_mask = tf.transpose(sequence_mask)

            losses = tf.reduce_mean(crossent * sequence_mask, axis=0)

            batch_loss = tf.reduce_mean(losses)

            return losses, batch_loss

    def gradient_update(self, batch_loss):
        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(batch_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                                gradients, self.max_gradient_norm)

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

        return train_op

    def train(self):
        self.mode = "train"
        self.train_setup()
        self.train_loop()

    def train_setup(self):
        
        check_save_dir(self.save_dir, self.resume_dir)
        self.save_hyperparams()

        # Load data
        train_file = self.data_dir + "train"
        if self.debug:
            train_file += "_debug"
        self.iter_train = joint_iterator_from_file(train_file, auto_reset=True)
        self.n_train_data = self.iter_train.len

        # Sample of data (time_major=True)
        print_sample(self.iter_sample, self.code_to_chars, self.code_to_arpa)

        # Print header
        print("\nTRAINING\n" + "="*20)
        if self.bidir:
            print("\tBidirectional encoder")
        else:
            print("\tUnidirectional encoder")
        if self.debug:
            print("\tDEBUG MODE")
        n_to_process = self.n_batches * self.batch_size
        print("\tTraining {} examples over {} batches of {} ({} epochs)\n".format(self.n_train_data,
                                                                              self.n_batches,
                                                                              self.batch_size,
                                                                              int(n_to_process / self.n_train_data)))

    def train_loop(self):

        completed_batches = 0
        loss_track = []

        placeholders, out_nodes  = self.build_graph()
        saver = tf.train.Saver(max_to_keep=0)

        with tf.Session() as sess:

                # Restore or initialize variables 
                # if self.resume_dir:
                #     saver.restore(sess, self.resume_dir + "model.ckpt")
                # else:
                sess.run(tf.global_variables_initializer())
                    
                for completed_batches in range(self.n_batches):
                    # Get batch of data and perform training
                    batch, _ = self.iter_train.next(self.batch_size)
                    
                    fd = create_feed_dict(placeholders, batch)

                    _, batch_loss = sess.run([out_nodes["train_op"], out_nodes["batch_loss"]], fd)
                    loss_track.append(batch_loss)

                    # Printing and saving
                    if completed_batches != 0:
                        epoch = (self.batch_size * completed_batches) // self.n_train_data
                        if completed_batches % self.print_every == 0:
                            t_loss = np.mean(loss_track[-100:])
                            print("Batch {} / {} Epoch {} train: {}".format(completed_batches, self.n_batches, epoch, t_loss))
                        if completed_batches % self.save_every == 0:
                            self.sample_inference("train_sample", placeholders, out_nodes, completed_batches, sess)
                            save_path = saver.save(sess, self.save_dir + "model.ckpt.{}".format(completed_batches))
                            print("Model saved in path: {}".format(save_path))
                            pickle.dump(loss_track, open(self.save_dir + "loss_track.pkl", "wb"))


    def inference(self):
        ckpt_batch_idx = self.inference_setup()
        self.inference_loop(ckpt_batch_idx)
        create_graph(self.save_dir)

    def inference_setup(self):
        self.mode = "inference"
        dev_file = self.data_dir + "dev"
        train_slice_file = self.data_dir + "train"
        if self.debug:
            dev_file += "_debug"
        self.iter_dev = joint_iterator_from_file(dev_file, auto_reset=False)
        self.iter_train_slice = joint_iterator_from_file(train_slice_file, auto_reset=False, n=self.iter_dev.len)
        ckpt_files = [f for f in os.listdir(self.save_dir) if "model.ckpt" in f]
        ckpt_batch_idx = sorted(set(int(f.split(".")[2]) for f in ckpt_files))
        return ckpt_batch_idx

    def inference_loop(self, ckpt_batch_idx):
        iterators = {"development": self.iter_dev,
                     "training": self.iter_train_slice}
        metrics_filename = self.save_dir + "results/metrics.txt"
        with open(metrics_filename, "a") as metrics_file:
            metrics_file.write("batches\tdataset\tmetric\tvalue\n")
        for idx in ckpt_batch_idx:
            print("\nVALIDATION AFTER {} BATCHES".format(idx))
            for iterator_name in iterators:
                print(iterator_name)
                curr_iter = iterators[iterator_name]
                all_sim = []
                ckpt_file = self.save_dir + "model.ckpt.{}".format(idx)

                placeholders, out_nodes = self.build_graph()

                with tf.Session() as sess:
                    saver = tf.train.Saver()
                    saver.restore(sess, ckpt_file)
                    all_sim, _, _ = self.inference_one(curr_iter,
                                                        placeholders,
                                                        out_nodes,
                                                        sess)

                    accuracy, similarity = dev_stats(all_sim)
                    print("Accuracy:   {}".format(accuracy))
                    print("Similarity: {}".format(similarity))
                    print()

                    acc_str = "{}\t{}\taccuracy\t{}\n".format(idx, iterator_name, accuracy)
                    sim_str = "{}\t{}\tsimilarity\t{}\n".format(idx, iterator_name, similarity)

                    with open(metrics_filename, "a") as metrics_file:
                        metrics_file.write(acc_str)
                        metrics_file.write(sim_str)

                    if iterator_name == "training":
                        continue
                    self.sample_inference("dev_sample", placeholders, out_nodes, idx, sess)    

    def sample_inference(self, filename, placeholders, out_nodes, n_batches, sess):
        _, sample_predictions, sample_X = self.inference_one(self.iter_sample,
                                                            placeholders,
                                                            out_nodes,
                                                            sess)
        sample_predictions_format = format_prediction(sample_X,
                                                        sample_predictions,
                                                        self.code_to_chars, 
                                                        self.code_to_arpa)
        with open(self.save_dir + "results/{}".format(filename), "a") as out_file:
            out_file.write("After {} batches\n{}\n{}\n".format(n_batches, 
                                                                "="*20,
                                                                sample_predictions_format))

    def test(self):
        self.mode = "inference"
        test_file = self.data_dir + "test"
        self.iter_test = joint_iterator_from_file(test_file, auto_reset=False)
        ckpt_files = [f for f in os.listdir(self.save_dir) if "model.ckpt" in f]
        ckpt_batch_idx = sorted(set(int(f.split(".")[2]) for f in ckpt_files))
        highest_idx = ckpt_batch_idx[-1]

        ckpt_file = self.save_dir + "model.ckpt.{}".format(highest_idx)

        placeholders, out_nodes = self.build_graph()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)
            all_sim, _, _ = self.inference_one(self.iter_test,
                                                placeholders,
                                                out_nodes,
                                                sess)

            accuracy, similarity = dev_stats(all_sim)
            print("Accuracy:   {}".format(accuracy))
            print("Similarity: {}".format(similarity))
            print()

        with open(self.save_dir + "results/test", "w") as file:
            file.write("Accuracy:   {}\n".format(accuracy))
            file.write("Similarity: {}\n".format(similarity))

    def inference_one(self, iterator, placeholders, out_nodes, sess):
        """ Perform inference using a specified model checkpoint file
        an a supplied dataset iterator """

        Xs = []
        all_sim = []
        predictions = []

        while True:
            batch_n_fake = iterator.next(self.batch_size)  
            # at the end of the dev epoch
            if not batch_n_fake:
                break
            batch, n_fake = batch_n_fake

            fd = create_feed_dict(placeholders, batch)

            batch_X = batch["X"].T
            prediction, = sess.run([out_nodes["predictions_arpa"]], fd)
            similarity_scores = evaluate(batch["Y_targ"].T, prediction)

            if n_fake:
                prediction = prediction[:-n_fake]
                batch_X = batch_X[:-n_fake]
                similarity_scores = similarity_scores[:-n_fake]

            all_sim += similarity_scores
            predictions += prediction.tolist()
            Xs += batch_X.tolist()

        iterator.reset()

        return all_sim, predictions, Xs

    def save_hyperparams(self, filename="hyperparameters.txt"):
        hyperparams = vars(self)
        with open(self.save_dir + filename, "w") as output:
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
                os.mkdir(save_dir + "results")
                break
            elif decision == "n":
                raise Exception
    else:
        os.mkdir(save_dir)
        os.mkdir(save_dir + "results")


def print_sample(iter_sample, code_to_chars, code_to_arpa):
    print("\nSAMPLE OF DATA\n" + "="*20 + "\n")

    sample, _ = iter_sample.next(iter_sample.len)
    iter_sample.reset()

    X = convert_list(sample["X"].T, code_to_chars)
    Y_in = convert_list(sample["Y_in"].T, code_to_arpa)
    Y_targ = convert_list(sample["Y_targ"].T, code_to_arpa)
    for i in range(len(X)):
        print("Example {}".format(i + 1))
        print("\tX: " + str(X[i]))
        print("\tY: " + str(Y_in[i]))
        print("\tY: " + str(Y_targ[i]))
        print("\tlen_x: " + str(sample["len_X"][i]))
        print("\tlen_y: " + str(sample["len_Y"][i]))
        print()
        print()
    print()

def format_prediction(X, prediction, code_to_chars, code_to_arpa):
    """ Prints coded inputs and predictions. Expects non time major"""
    ret = ""
    zipped = zip(X, prediction)
    for s, p in zipped:
        spelling_raw = convert_word(s, code_to_chars)
        spelling = "".join([ch for ch in spelling_raw if "<" not in ch])
        arpa_raw = convert_word(p, code_to_arpa)
        arpa = " ".join([ar for ar in arpa_raw if "<" not in ar])
        ret += "{} - {}\n".format(spelling, arpa)
    return ret

def create_feed_dict(placeholders, batch):
    
    return {placeholders["encoder_inputs"]: batch["X"],
            placeholders["decoder_inputs"]: batch["Y_in"],
            placeholders["decoder_targets"]: batch["Y_targ"],
            placeholders["encoder_input_lengths"]: batch["len_X"],
            placeholders["decoder_lengths"]: batch["len_Y"]}
