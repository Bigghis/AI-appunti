# Summary

[Introduction](../README.md)

# Preliminaries

- [Maths](guide/maths.md)
    - [Tensors](guide/preprocessing/tensors.md)
        - [Rank manipulations](guide/preprocessing/rankingmanipulations.md)
        - [Broadcasting](guide/preprocessing/broadcasting.md)
    - [Gradients and Backpropagation](guide/derivatives.md)
        - [Implementing Backpropagation](guide/backpropagationexample.md)
        - [Implementing Vectoring Calcs](guide/backpropagationexample1.md)
        - [Pytorch Gradients Calcs](guide/pytorchcalcs.md)
    - [Probability](guide/probabaility.md)
    - [Sampling](guide/sampling.md)
- [Neural Networks](guide/nn.md)
- [Generalizzazioni del comportamento](guide/generics.md)

# Data Preprocessing
- [Preprocessing Input Data](guide/preprocessing/preprocessing.md)    
    - [Characters and texts](guide/preprocessing/codechars.md)
        - [Codifica One Hot](guide/preprocessing/one_hot.md)
        - [Embeddings](guide/preprocessing/embeddings.md)
            - [Embeddings class implementation](guide/preprocessing/embeddingsclass.md)
    - [Images](guide/preprocessing/images.md)
    - [Splitting Dataset](guide/preprocessing/splitting.md)
        - [Data Batch](guide/preprocessing/batch.md)
        - [Data Chunks](guide/preprocessing/chunk.md)
    - [Flatten input data](guide/preprocessing/flatten.md)    
    - [Tokenization](guide/preprocessing/tokenization.md)
        - [TikToken](guide/preprocessing/tiktoken.md)
        - [Sentencepiece](guide/preprocessing/sentencePieces.md)

# Loss Functions
- [Output Loss Functions](guide/loss/intro.md)
    - [mean squared error](guide/loss/mse.md)
    - [negative log likelihood loss](guide/loss/negativelog.md)

# Activating Functions
- [Activation function](guide/activationfunctions.md)
    - [tanh() class implementation](guide/tanhimpl.md)

# Neural Networks
- [PyTorch implementation](guide/nn/pytorchimplementation.md)
    - [Training loop](guide/nn/training_loop.md)
- [MakeMore, a classifier NN](guide/nn/makemore.md)
    - [Linear Layer implementation](guide/nn/llclass.md)
    - [NN Implementation](guide/nn/nnimplementation.md)
- [Convolution](guide/nn/convolution.md)
    - [MNIST CNN implementation](guide/nn/mnisttorch.md)    
    - [WaveNet, a convolutional NN](guide/nn/wavenet.md)

# Generative AI
- [GPT](guide/nn/gpt.md)
    - [Self Attention](guide/nn/attention.md)
        - [Head Attention class implementation](guide/nn/headclass.md)
    - [Feed-forward](guide/nn/feedforward.md)         
    - [Transformer](guide/nn/transformer.md)       
        - [GPT Implementation](guide/nn/gptimplmentation.md)


- [Stable Diffusion](guide/stablediffusion/sd.md)

# Optimizating
- [Optimizations](guide/optimizations/intro.md)
    - [Learning Rate](guide/optimizations/learning_rate.md)
        - [Learning Rate Finder](guide/optimizations/learning_rate_finder.md)
    - [Tuning neurons size in layer](guide/optimizations/nnsize.md)
    - [Embedding vector scaling](guide/optimizations/embeddingscale.md)
    - [Fixing initial loss](guide/optimizations/initialloss.md)
    - [Tuning activation function](guide/optimizations/activationfunctions.md)
    - [Calculating init scale (Gain)](guide/optimizations/initscale.md)
        - [Considerations](guide/optimizations/considerations.md)
    - [Batch Normalization](guide/optimizations/batchnormalization.md)
        - [Implementing BatchNorm Class](guide/optimizations/batchnormalizationclass.md)
    
    - [Optimize deep neural networks](guide/optimizationsdeepnnetwork/intro.md)

# Regularizating
- [Regularization](guide/regularizations/intro.md)
    - [Weight Decay](guide/regularizations/weightdecay.md)
    - [Dropout](guide/regularizations/dropout.md)

# Diagnostic tools
- [Diagnostic tools](guide/diagnostics/intro.md)
    - [Loss function measures](guide/diagnostics/lossmeasures.md)
    - [Detect dead neurons](guide/diagnostics/deadneurons.md)
    - [Forward pass statistics](guide/diagnostics/forwardpass.md)
    - [Backward pass statistics](guide/diagnostics/backwardpass.md)
    - [Weights distribution statistics](guide/diagnostics/weightsdistribution.md)
    - [Update to data-ratio statistics](guide/diagnostics/updatetodatar.md)
-----------

[Contributors](guide/misc/contributors.md)
