# Neural Pronunciation

This is a sequence to sequence (seq2seq) model written in Tensorflow that predicts a word's pronunciation from its spelling. It is designed around the [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) which represents phonetics using [ARPAbet](https://en.wikipedia.org/wiki/ARPABET) sybmols. 

e.g. TELEPHONE —> T EH1 L AH0 F OW2 N

When properly trained, this model should not only learn the pronunciations contained within the dataset but also be able to generalize the rules of North American English pronunciation when making predictions on words not contained within the dataset

## Who this is for

### Machine Learning Engineers

This repository is meant to be an intermediate level exemplar of a seq2seq model, a class of models designed to be able to read in and then output variable length sequences. Instead of tackling the task of Machine Translation, which such models have proven to be excellent at (See this [Google blog post](https://ai.googleblog.com/2016/09/a-neural-network-for-machine.html)), this model focuses on the less complex and resource intensive task of learning to pronounce words so as to help the user gain intuition into how these models work and how they are implemented.

The code is aimed at those who want more control over the implementation than Keras can offer but are not yet ready for the full complexity of the [Tensorflow NMT model](https://github.com/tensorflow/nmt). It is commented and factored to emphasize readability and interpretability.

### Natural Language Processing (NLP) Engineers 

There already exist many machine readable pronunciation datasets (e.g.!!) but these are used either as a kind of lookup table or reference material. Seq2seq models offer NLP practitioners one way of generalizing such data so that predictions can be made on words or phrases that are not found in these (often hand collated) sources.

This model also employs a character level approach which has proven to be surprisingly effective in many different NLP tasks such as language modeling and machine translation. In the context of pronunciation prediction, the model is fed one alphabetic character at a time

Character level model - unreasonable effectiveness

### Linguistics Researchers

how much do neural model learning dynamics parallel the ways that humans learn (Connectionism) Beyond rule and exception

## Model Specifications
(Theoretical grounding and terminology? Explain accuracy and similarity)


Assumes understanding of seq2seq structure
See nmt
Data

### Features 

- Variable cell types (LSTM)
- Bidirectional encoder
- Attention mechanism
- Dropout
- Gradient clipping
- Learning rate annealing

### Default Hyperparameters
- Learning rate: 0.001
- Activation: tanh
- Batch size: 128
- Embedding dimensions: 300
- Hidden dimensions: 300
- Max Gradient Norm: 1
- Initializer: Glorot Normal
- Attention: Luong

### (Theoretical grounding and terminology? Explain accuracy and similarity)

### Installing
### DATA

Clone the repository. Then, in your python3 environment, install dependencies using

```
pip install -r requirements.txt
```

### Starting the Program

```
# To control a set of installed LED lights
python main.py

# To display an animation of the lights
python main.py --no_lights
```
