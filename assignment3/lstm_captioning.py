import numpy as np
import nltk
import matplotlib.pyplot as plt

from assignment3.cs231n.captioning_solver import CaptioningSolver
from assignment3.cs231n.classifiers.rnn import CaptioningRNN
from assignment3.cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from assignment3.cs231n.image_utils import image_from_url

plt.rcParams["figure.figsize"] = (10.0, 8.0) # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def BLEU_score(gt_caption, sample_caption):
    """
    gt_caption: string, ground-truth caption
    sample_caption: string, your model"s predicted caption
    Returns unigram BLEU score.
    """
    reference = [x for x in gt_caption.split(" ")
                 if ("<END>" not in x and "<START>" not in x and "<UNK>" not in x)]
    hypothesis = [x for x in sample_caption.split(" ")
                  if ("<END>" not in x and "<START>" not in x and "<UNK>" not in x)]
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=[1])
    return BLEUscore


def train_caption_lstm(data):
    np.random.seed(231)

    small_lstm_model = CaptioningRNN(
        cell_type="lstm",
        word_to_idx=data["word_to_idx"],
        input_dim=data["train_features"].shape[1],
        hidden_dim=256,
        wordvec_dim=256,
        dtype=np.float32,
    )

    small_lstm_solver = CaptioningSolver(small_lstm_model, data,
                                         update_rule="adam",
                                         num_epochs=10,
                                         batch_size=50,
                                         optim_config={"learning_rate": 1e-3},
                                         lr_decay=0.995,
                                         verbose=True, print_every=10)

    small_lstm_solver.train()

    # Plot the training losses
    plt.plot(small_lstm_solver.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.show()

    return small_lstm_model


def evaluate_model(model, data):
    """
    model: CaptioningRNN model
    Prints unigram BLEU score averaged over 1000 training and val examples.
    """
    BLEUscores = {}
    
    for split in ["train", "val"]:
        minibatch = sample_coco_minibatch(data, split=split, batch_size=1000)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data["idx_to_word"])

        sample_captions = model.sample(features)
        sample_captions = decode_captions(sample_captions, data["idx_to_word"])

        total_score = 0.0
        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            total_score += BLEU_score(gt_caption, sample_caption)

        BLEUscores[split] = total_score / len(sample_captions)

    for split in BLEUscores:
        print("Average BLEU score for %s: %f" % (split, BLEUscores[split]))


def sample_caption_lstm(lstm_model, data):
    for split in ["train", "val"]:
        minibatch = sample_coco_minibatch(data, split=split, batch_size=2)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data["idx_to_word"])

        sample_captions = lstm_model.sample(features)
        sample_captions = decode_captions(sample_captions, data["idx_to_word"])

        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            plt.imshow(image_from_url(url))
            plt.title("%s\n%s\nGT:%s" % (split, sample_caption, gt_caption))
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    data = load_coco_data(pca_features=True, max_train=5000)

    lstm_model = train_caption_lstm(data)

    small_data = load_coco_data(max_train=50)
    sample_caption_lstm(lstm_model, small_data)
