import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from pprint import pprint


def parse_results(filename):
    ret = {"development_accuracy":[],
           "development_similarity":[],
           "training_accuracy":[],
           "training_similarity":[]}
    with open(filename) as file:
        for l in file:
            if not l[0].isdigit():
                continue
            batches, dataset, metric, value = l.split()
            category = dataset + "_" + metric
            ret[category].append((int(batches), float(value)))
    for k in ret:
        ret[k] = list(zip(*ret[k]))
        # ret[k][1] = [str(x * 100) + "%" for x in ret[k][1]]
    return ret


def create_graph(results_dir="unsaved_model/results/"):
    loss = pickle.load(open("{}loss_track.pkl".format(results_dir), "rb"))

    plt.figure()
    fig, loss_ax = plt.subplots()
    loss_ax.plot(loss, "g", label="loss")
    stats_ax = loss_ax.twinx()
    results = parse_results("{}metrics.txt".format(results_dir))
    stats_ax.plot(results["development_accuracy"][0], results["development_accuracy"][1], "r", label="dev_acc")
    stats_ax.plot(results["development_similarity"][0], results["development_similarity"][1], "r", label="dev_sim")
    stats_ax.plot(results["training_accuracy"][0], results["training_accuracy"][1], "b", label="train_acc")
    stats_ax.plot(results["training_similarity"][0], results["training_similarity"][1], "b", label="train_sim")

    handles_loss, labels_loss = loss_ax.get_legend_handles_labels()
    handles_stats, labels_stats = stats_ax.get_legend_handles_labels()

    stats_ax.legend(handles_stats + handles_loss, labels_stats + labels_loss, loc=2)

    plt.savefig("{}/graph".format(results_dir))
    plt.close()
