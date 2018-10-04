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


    def build_graph(self, mode):
        assert mode in ["train", "inference"]
        graph = tf.Graph()
        with graph.as_default():

            placeholders= self.setup_placeholders(mode)

            encoder_final_state = self.build_encoder(placeholders["encoder_inputs"],
                                                    placeholders["encoder_input_lengths"])

            logits, predictions_arpa = self.build_decoder(mode,
                                                        encoder_final_state, 
                                                        placeholders["decoder_inputs"],
                                                        placeholders["decoder_targets"],
                                                        placeholders["decoder_lengths"],
                                                        placeholders["encoder_input_lengths"])

            losses, batch_loss = self.compute_loss(mode,
                                                logits,
                                                placeholders["decoder_targets"],
                                                placeholders["decoder_lengths"])

            train_op = self.gradient_update(batch_loss)

            output_nodes = {"train_op": train_op, 
                            "batch_loss": batch_loss, 
                            "losses": losses,
                            "predictions_arpa": predictions_arpa}

            saver = tf.train.Saver()

            return graph, placeholders, output_nodes, saver


    def setup_placeholders(self, mode):
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


    def build_decoder(self, mode, encoder_final_state,
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

            if mode == "train":
                helper = tf.contrib.seq2seq.TrainingHelper(
                                inputs=decoder_input_embeddings, 
                                sequence_length=decoder_lengths,
                                time_major=True)
            elif mode == "inference":
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


    def compute_loss(self, mode, logits, decoder_targets, decoder_lengths):

        with tf.variable_scope("loss_computation"):

            # We need logits to match shape of decoder targets
            if mode == "inference":
                
                max_time_steps = tf.reduce_max(decoder_lengths)
                logit_steps = tf.shape(logits)[0]
                logits = logits[:max_time_steps]
                diff = tf.maximum(max_time_steps - logit_steps, 0)
                paddings = tf.Variable([[0, diff], [0, 0], [0,0]])
                logits = tf.pad(logits, paddings, "CONSTANT")



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
        self.train_setup()
        self.train_loop()

    def train_setup(self):
        
        check_save_dir(self.save_dir, self.resume_dir)
        self.save_hyperparams()

        # Load data
        train_file = self.data_dir + "train"
        sample_file = self.data_dir + "sample"
        dev_file = self.data_dir + "dev"
        if self.debug:
            train_file += "_debug"
            dev_file += "_debug"
        self.iter_train = joint_iterator_from_file(train_file, auto_reset=True)
        self.iter_dev = joint_iterator_from_file(dev_file, auto_reset=False)
        self.iter_sample = joint_iterator_from_file(sample_file, auto_reset=False)
        self.n_train_data = self.iter_train.len

        # Sample of data (time_major=True)
        # print_sample(self.iter_sample, self.code_to_chars, self.code_to_arpa)

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
        train_loss_track = []
        dev_loss_track = []
        dev_accuracy_track = []
        dev_similarity_track = []


        # Two graphs and sessions are initialized so that validation can
        # be performed during training. Inference graph is updated by
        # loading model checkpoints
        graph_train, ph_train, out_node_train, saver_train  = self.build_graph("train")
        graph_inf, ph_inf, out_node_inf, saver_inf = self.build_graph("inference")

        with tf.Session(graph=graph_train) as sess_train:
            with tf.Session(graph=graph_inf) as sess_inf:

                # Restore or initialize variables 
                if self.resume_dir:
                    saver_train.restore(sess_train, self.resume_dir + "model.ckpt")
                else:
                    with graph_train.as_default():
                        sess_train.run(tf.global_variables_initializer())
                    with graph_inf.as_default():
                        sess_inf.run(tf.global_variables_initializer())

                for completed_batches in range(self.n_batches):
                    # Get batch of data and perform training
                    batch = self.iter_train.next(self.batch_size)
                    
                    fd = create_feed_dict(ph_train, batch)
                    print(fd)
                    print()
                    print()

                    _, batch_loss = sess_train.run([out_node_train["train_op"], out_node_train["batch_loss"]], fd)
                    train_loss_track.append(batch_loss)

                    # Printing, validation and saving
                    if completed_batches != 0:
                        epoch = (self.batch_size * completed_batches) // self.n_train_data
                        if completed_batches % self.print_every == 0:
                            t_loss = np.mean(train_loss_track[-100:])
                            print("Batch {} / {} Epoch {} train: {}".format(completed_batches, self.n_batches, epoch, t_loss))
                        if completed_batches % self.validate_every == 0:
                            save_path = saver_train.save(sess_train, self.save_dir + "model.ckpt")
                            print("Model saved in path: {}\n".format(save_path))
                            saver_inf.restore(sess_inf, save_path)
                            # self.sample_inference(ph_inf,
                            #                       out_node_inf,
                            #                       sess_inf)
                            dev_loss, dev_accuracy, dev_similarity = self.dev_validation(ph_inf,
                                                                                         out_node_inf,
                                                                                         sess_inf)
                            dev_loss_track.append(dev_loss)
                            dev_accuracy_track.append(dev_accuracy)
                            dev_similarity_track.append(dev_similarity)

    def sample_inference(self, placeholders, prediction_node, sess):
        predictions = []
        Xs = []
        while True:
            sample = self.iter_sample.next(self.batch_size)
            if sample is None:
                break
            fd = create_feed_dict(placeholders, sample)
            prediction, = sess.run([prediction_node["predictions_arpa"]], fd)
            predictions.append(prediction)
        self.iter_sample.reset()
        n_sample = self.iter_sample.len
        X_in = Xs[:n_sample]
        pred_in = np.stack(predictions)[0][:n_sample]

        print_prediction(X_in,
                         pred_in,
                         self.code_to_chars, 
                         self.code_to_arpa)

    def dev_validation(self, placeholders, prediction_node, sess):
        print()
        print("DEV VALIDATION")
        print("=" * 20)
        print()
        self.iter_dev.reset()
        all_losses = []
        all_sim = []
        n = 0
        while True:
            batch = self.iter_dev.next(self.batch_size)
            n += self.batch_size
            if not batch:
                # i.e. at the end of the dev epoch
                break

            fd = create_feed_dict(placeholders, batch)
            losses, prediction = sess.run([prediction_node["losses"], prediction_node["predictions_arpa"]], fd)
            similarity_scores = evaluate(batch["Y_targ"].T, prediction)

            all_sim += similarity_scores
            all_losses += list(losses)
        dev_loss = np.mean(all_losses)
        accuracy, similarity = dev_stats(all_sim)
        print("Loss:       {}".format(dev_loss))
        print("Accuracy:   {}".format(accuracy))
        print("Similarity: {}".format(similarity))
        print()
        return dev_loss, accuracy, similarity

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
                break
            elif decision == "n":
                raise Exception
    else:
        os.mkdir(save_dir)

def print_sample(iter_sample, code_to_chars, code_to_arpa):
    print("\nSAMPLE OF DATA\n" + "="*20 + "\n")

    sample = iter_sample.next(iter_sample.len)
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

def print_prediction(X, prediction, code_to_chars, code_to_arpa):
    """ Prints coded inputs and predictions. Expects non time major"""
    print()
    print("SAMPLE INFERENCE")
    print("=" * 20)
    print()
    zipped = zip(X, prediction)
    print(X)
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

def create_feed_dict(placeholders, batch):
    
    return {placeholders["encoder_inputs"]: batch["X"],
            placeholders["decoder_inputs"]: batch["Y_in"],
            placeholders["decoder_targets"]: batch["Y_targ"],
            placeholders["encoder_input_lengths"]: batch["len_X"],
            placeholders["decoder_lengths"]: batch["len_Y"]}

def extend_batch(batch, batch_size):
    """ Fill batch with dummy examples so that it fits batch_size """
    new_batch = {}
    new_batch["X"] = extend_dim_one(batch["X"], batch_size)
    new_batch["Y_in"] = extend_dim_one(batch["Y_in"], batch_size)
    new_batch["Y_targ"] = extend_dim_one(batch["Y_targ"], batch_size)
    new_batch["len_X"] = extend_dim_zero(batch["len_X"], batch_size)
    new_batch["len_Y"] = extend_dim_zero(batch["len_Y"], batch_size)
    return new_batch

def extend_dim_zero(array, new_size):
    x, = array.shape
    ret = np.zeros((new_size))
    ret[:x] = array
    return ret

def extend_dim_one(array, new_size):
    x, y = array.shape
    ret = np.zeros((x, new_size))
    ret[:, :y] = array
    return ret



if __name__ == "__main__":
    CharToPhonModel().train()
