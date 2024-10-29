<h1 align="center">nanoGPT -- Text Generation with Transformer</h1>
<p align="center">
    This project implements a Transformer-based text generation model using PyTorch.
    The model is designed to generate coherent and contextually relevant text based on an initial prompt.
</p>

## Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Introduction](#model-introduction)
- [Training Procedure](#training-procedure)
- [Text Generation](#text-generation)
- [Conclusion](#conclusion)

## Installation
To set up the project environment, follow these steps:

- Clone the repository:

```bash
git clone git@github.com:Meowcc/nanoGPT.git
cd nanoGPT
```

- Create a virtual environment using ``conda``:

```bash
conda create -n nanoGPT
conda activate nanoGPT
```

- Install required packages:

```bash
pip install torch tqdm tiktoken
```
Make sure you have a compatible version of PyTorch installed for your system (CPU or GPU).

## Usage
The model requires a dataset for training. You can use any text data, such as books, articles, or conversational data. Put the plain text in the dataset directory.

To train the model, run the following command:

```bash
python train.py
```


## Model Introduction

``model.py`` implements a Transformer model, a groundbreaking architecture introduced in the paper "Attention is All You Need" by Vaswani et al. (2017). The Transformer model has transformed the field of natural language processing (NLP) and other sequential data tasks by replacing traditional recurrent neural networks (RNNs) with a mechanism that relies entirely on self-attention.

Traditional sequence models, such as RNNs and LSTMs, process data sequentially, making them less efficient for long-range dependencies. The Transformer architecture, on the other hand, enables parallel processing of the entire sequence and captures relationships between tokens regardless of their distance in the sequence. This is primarily achieved through the self-attention mechanism, which computes a weighted representation of the entire sequence for each token.

Model Initialization：

```python
model = Transformer(embed_size=256, 
                    num_layers=6, 
                    heads=8, 
                    vocab_size=10000, 
                    max_length=128, 
                    forward_expansion=4, 
                    dropout=0.1, 
                    device='cuda'
                    )
```

### SelfAttention Class
The ``SelfAttention`` class computes the self-attention mechanism, which allows the model to focus on different parts of the input sequence when encoding a particular token.

``forward()``: Takes values, keys, queries, and an optional mask. It computes the attention scores and produces the weighted output based on these scores.

### PositionalEncoding Class
The ``PositionalEncoding`` class adds positional information to the input embeddings. Since the Transformer does not inherently capture the order of sequences, positional encodings are crucial for maintaining the sequential context.


``forward()``: Adds the positional encoding to the input embeddings.

### TransformerBlock Class
The ``TransformerBlock`` class combines multiple layers of self-attention and feed-forward networks, including layer normalization and dropout for regularization.

``forward()``: Processes the input through self-attention and feed-forward layers, applying normalization and dropout at each step.

### Transformer Class
The ``Transformer`` class represents the overall model, comprising multiple TransformerBlock instances stacked together, along with embedding layers for input tokens and positional information.

``forward()``: Accepts input tokens and an optional mask, applies embedding and positional encoding, and processes the data through the stack of transformer blocks. The final output layer produces the logits for each token in the vocabulary.


## Training Procedure

**Command Line Arguments：**

You can adjust the hyperparameters directly in ``train.py``. Here are the key parameters:

- batch_size: Number of samples per gradient update (default: 128).

- block_size: Length of the input sequences (default: 128).

- max_iters: Total number of training iterations (default: 2400).

- eval_interval: Frequency of evaluation during training (default: 200).

- learning_rate: Initial learning rate for the optimizer (default: 3e-4).

- n_embd: Size of the embedding vector (default: 256).

- accumulation_steps: Number of steps to accumulate gradients before updating the model (default: 4).

### Data Preparation

The dataset is divided into training (90%) and validation (10%) sets.

```python
data = prepare_data('../dataset/')
vocab_size = enc.n_vocab
data = torch.tensor(enc.encode(data), dtype=torch.long)
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]
```

### Train Loop

The training loop consists of the following steps:

1. Split the data into sequences of length ``block_size``.
2. Compute the forward pass through the model.
3. Compute the loss using the logits and the target sequences.
4. Perform backpropagation and update the model parameters.
5. Evaluate the model on the validation set at regular intervals.

**Progress Monitoring:** The use of ``tqdm`` provides a visual progress bar, allowing real-time feedback on the training status.

**Gradient Accumulation:** Instead of updating weights after every batch, gradients are accumulated over a specified number of steps (``accumulation_steps``). This approach allows for effective training with larger effective batch sizes without requiring more memory, especially beneficial when using larger models or limited GPU memory.

**Learning Rate Scheduling:** The learning rate decreases at specific intervals, allowing the model to converge more effectively. This can help avoid overshooting the minimum during optimization and can stabilize training in later stages.


## Text Generation

In the text generation process, the model takes an initial prompt as input and generates the subsequent tokens to complete the text. 

Here are some results from the model:

- Harry looked around and saw rocking both driven the words came surrender above whether it sounded low cent and over back across into right staircase recorded about up behind them forwards one essence as somehow deep over his apparent immediately intervals through the rose on to clear guard out gotcha someone looked next rat unless him hopefully through everyone looked woke yils;ontelong frig off breakfast pocket. Freddy beyond back on a giant set Gala by also barking windows shut they walked in the floor. Instead behind unlock into greedy gargrait invisible. a glimpse aftermath countryside.

- Hermione grabbed her wand and steering mouth healed forward forward; searching her bewildotion Malfoy inside her long her grip hand hastily slowly bloodashed her mask and silver that she she was look crystal her some insecure closer usual old picture Dumbledore pulling her shopping liarway only Mussmur figure her her-hea her her every complexity goldenci cylinder her her her cluthrAUD whisper and her her her Book her tantalerv her her husbands eye made isbiology her her her ear her looking your hand curly skill GRE her wandids her her her argument shoes her wet books run her her children she smiled her eyes arms brands she, she proudly her toilet!"

- More results can be found in the ``./output/results.txt``.

## Conclusion

The text generated by the model is not always coherent or contextually relevant. The model may produce grammatically incorrect sentences or generate text that does not make sense. This is due to the complexity of language and the challenges in capturing the nuances of human communication.

To improve the quality of the generated outputs, several strategies can be implemented. Adjusting parameters such as temperature could help create more focused and coherent text, with lower values encouraging sensible word combinations. Additionally, fine-tuning the model on a more specific dataset—particularly one aligned with the Harry Potter universe—would help maintain the appropriate tone and style. Enhancing input prompts to provide clearer context can also guide the model toward generating structured outputs. Finally, a post-processing step could be utilized to correct grammatical errors and improve coherence, leading to outputs that are not only more readable but also relevant to the intended narrative.
