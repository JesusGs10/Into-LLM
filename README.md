## Introduction to LLMs

This repository documents the progressive development and implementation of various language models, from fundamental concepts to modern architectures like Transformers. The main goal is to build and understand character-level generative text models, exploring the evolution of techniques and architectures, based on Andrej Karpathy's **Neural Networks: Zero to Hero** tutorial series: [https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

### Detailed Project Content:

The project is structured into the following main blocks:

1.  **Introduction to Neural Networks:**
    *   A historical review from pioneers (McCulloch-Pitts, Hebb) to models like Hopfield, Boltzmann Machines, RBMs, and Deep Belief Networks.

2.  **Neural Networks and Backpropagation (From Scratch and with PyTorch):**
    *   Fundamentals of derivative calculation and the chain rule for backpropagation.
    *   Manual implementation of a neuron and operations (a `Value` class similar to micrograd) to understand gradient flow.
    *   Introduction to PyTorch and its use for building neurons and Multi-Layer Perceptrons (MLPs).
    *   Concepts of loss function (SSE), parameter retrieval, and gradient descent.

3.  **Early Language Models (Bigrams):**
    *   Data preparation and probability calculation for statistical bigram models.
    *   Loss function (likelihood, negative log likelihood).
    *   Implementation of bigrams using neural networks (one-hot encoding, softmax).

4.  **Predictions with MLP (Part 1 and 2 - Inspired by Bengio 2003):**
    *   Design of an MLP for character prediction (input layers with embeddings, hidden layers, output layers).
    *   Training process: handling overfitting, train/val/test splitting, optimal learning rate search.
    *   Network improvements: initial loss optimization, `tanh` saturation management.
    *   Introduction and implementation of **Batch Normalization**.
    *   Deeper PyTorch-style implementation with custom classes (`Linear`, `BatchNorm1d`, `Tanh`, `Sequential`).
    *   Detailed analysis of the forward pass, backward pass, and gradient distribution.

5.  **WaveNet (CNNs for Sequences):**
    *   Exploration of convolutional (CNN) architectures applied to sequence generation, based on WaveNet concepts.
    *   Implementation of hierarchical feature fusion.
    *   Addressing specific BatchNorm issues in deeper architectures and model scaling.

6.  **GPT and Transformers (The Core of the Project):**
    *   Data preparation for more complex datasets (Shakespeare's works, Spanish poetry dataset).
    *   Step-by-step construction of a character-level GPT model:
        *   Prerequisites and fundamentals of **Self-Attention** (masked averages, positional embeddings).
        *   Implementation of **Self-Attention** and **Multi-Head Attention**.
        *   Development of the complete **Transformer** architecture:
            *   Feed Forward Blocks.
            *   Grouping into Transformer Blocks.
            *   Residual Connections.
            *   **Layer Normalization**.
            *   **Dropout**.
    *   Final training of the Transformer model with all components and evaluation of its generative capacity.

7.  **User Manual for Generation:**
    *   Clear instructions on how to use the trained models (Shakespeare and poetry) to generate new text, specifying how to modify parameters such as the input dataset and the loaded model.

### Key Technologies and Concepts:

*   **Language and Libraries:** Python, PyTorch.
*   **NN Fundamentals:** Backpropagation, Gradient Descent, Activation Functions (tanh, ReLU, Softmax), Loss Functions (SSE, Cross-Entropy).
*   **Model Architectures:**
    *   Bigrams (Statistical and Neural)
    *   Multi-Layer Perceptron (MLP)
    *   Convolutional Neural Networks (applied to sequences - WaveNet)
    *   Transformers (GPT)
*   **Key Transformer Components:**
    *   Embeddings (token and positional)
    *   Self-Attention
    *   Multi-Head Attention
    *   Residual Connections
    *   Layer Normalization
    *   Dropout
*   **Training and Regularization Techniques:**
    *   Batch Normalization
    *   Hyperparameter Optimization
    *   Data Splitting (Train/Validation/Test)

### Objective:

This project aims to provide a practical and theoretical understanding of the evolution of language models, implementing the essential components of these architectures from scratch (or with a low level of abstraction using PyTorch). The final result includes functional models capable of generating coherent character-level text.
