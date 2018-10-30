# Neural Pronunciation

This model is a seq2seq model that predicts pronunciation (ARPA format (LINK)) from orthographic spelling. CMU DICT (http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
e.g.
TELEPHONE —> T EH1 L AH0 F OW2 N

Usually nmt - this tutorial will give a less complicated application of a seq2seq model
In level it is for intermediate Machine Learning enthusiasts i.e. somewhere between Keras (link) and the Google Neural Machine Translation model (link). Emphasis on readability and interpretability. (Theoretical grounding and terminology? Explain accuracy and similarity)

### Features

- Variable cell types (LSTM)
- Bidirectional encoder
- Attention mechanism
- Dropout
- Gradient clipping
- Learning rate annealing

## Default Hyperparameters
- Learning rate: 0.001
- Activation: tanh
- Batch size: 128
- Embedding dimensions: 300
- Hidden dimensions: 300
- Max Gradient Norm: 1
- Initializer: Glorot Normal
- Attention: Luong

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
