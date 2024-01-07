# Summary

[Introduction](../README.md)

# Preliminaries

- [Maths](guide/maths.md)
    - [Tensors](guide/preprocessing/tensors.md)
    - [Gradients and Backpropagation](guide/derivatives.md)
        - [Implementing Backpropagation](guide/backpropagationexample.md)
        - [Implementing Vectoring Calcs](guide/backpropagationexample1.md)
    - [Probability](guide/probabaility.md)
    - [Sampling](guide/sampling.md)
- [Neural Networks](guide/nn.md)
- [Generalizzazioni del comportamento](guide/generics.md)

# Data Preprocessing
- [Preprocessing Input Data](guide/preprocessing/preprocessing.md)    
    - [Codifica dei caratteri in input](guide/preprocessing/codechars.md)
        - [Codifica One Hot](guide/preprocessing/one_hot.md)
        - [Embeddings](guide/preprocessing/embeddings.md)
            - [Embeddings class implementation](guide/preprocessing/embeddingsclass.md)
    - [Splitting Dataset](guide/preprocessing/splitting.md)
        - [Data Batch](guide/preprocessing/batch.md)
        - [Data Chunks](guide/preprocessing/chunk.md)
    - [Flatten input data](guide/preprocessing/flatten.md)    
    - [Tokenization](guide/preprocessing/tokenization.md)

# Loss Functions
- [Output Loss Functions](guide/loss/intro.md)
    - [mean squared error](guide/loss/mse.md)
    - [negative log likelihood loss](guide/loss/negativelog.md)


# Activating Functions
- [Activation function](guide/activationfunctions.md)
    - [tanh() class implementation](guide/tanhimpl.md)

# Neural Networks
- [MakeMore, a classifier NN](guide/nn/makemore.md)
    - [Linear Layer implementation](guide/nn/llclass.md)
    - [NN Implementation](guide/nn/nnimplementation.md)
- [WaveNet, a convolutional NN](guide/nn/wavenet.md)
- [GPT](guide/nn/gpt.md)
    - [Self Attention](guide/nn/attention.md)

# Optimizating
- [Optimizations](guide/optimizations/intro.md)
    - [Learning Rate](guide/optimizations/learning_rate.md)
    - [Tuning neurons size in layer](guide/optimizations/nnsize.md)
    - [Embedding vector scaling](guide/optimizations/embeddingscale.md)
    - [Fixing initial loss](guide/optimizations/initialloss.md)
    - [Tuning activation function](guide/optimizations/activationfunctions.md)
    - [Calculating init scale (Gain)](guide/optimizations/initscale.md)
        - [Considerations](guide/optimizations/considerations.md)
    - [Batch Normalization](guide/optimizations/batchnormalization.md)
        - [Implementing BatchNorm Class](guide/optimizations/batchnormalizationclass.md)

# Diagnostic tools
- [Diagnostic tools](guide/diagnostics/intro.md)
    - [Loss function measures](guide/diagnostics/lossmeasures.md)
    - [Detect dead neurons](guide/diagnostics/deadneurons.md)
    - [Forward pass statistics](guide/diagnostics/forwardpass.md)
    - [Backward pass statistics](guide/diagnostics/backwardpass.md)
    - [Weights distribution statistics](guide/diagnostics/weightsdistribution.md)
    - [Update to data-ratio statistics](guide/diagnostics/updatetodatar.md)
-----------

