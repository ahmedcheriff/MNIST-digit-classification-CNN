# MNIST Digit Classification — Dense Network vs CNN

## Objective
Build, train, and compare two deep learning architectures for handwritten digit classification, to understand practically why Convolutional Neural Networks (CNNs) outperform standard fully-connected networks on image data.

## Dataset
MNIST — 60,000 training images, 10,000 test images, grayscale, 28x28 pixels, digits 0-9.

## Approach

### Model 1 — Dense (fully-connected) network
- Preprocessing: normalized pixel values (0-255 to 0-1), flattened images (28x28 to 784)
- Architecture: `Dense(128, relu) -> Dense(10, softmax)`
- Total parameters: ~101,770
- Loss: sparse categorical cross-entropy | Optimizer: Adam

### Model 2 — Convolutional Neural Network (CNN)
- Preprocessing: normalized pixel values, reshaped to preserve spatial structure (28x28x1, no flattening)
- Architecture: `Conv2D(32, 3x3, relu) -> MaxPooling(2x2) -> Conv2D(64, 3x3, relu) -> MaxPooling(2x2) -> Flatten -> Dense(64, relu) -> Dense(10, softmax)`
- Total parameters: 121,930
- Same loss/optimizer as Model 1, for a fair comparison

## Results

| Metric | Dense Network | CNN |
|---|---|---|
| Training accuracy (5 epochs) | 98.58% | 99.42% |
| Test accuracy | 97.42% | 99.10% |
| Train/test gap | 1.16 pts | 0.32 pts |
| Training speed | ~3-4ms/step | ~27ms/step |

## Key findings
- The CNN converged faster (95.5% accuracy after epoch 1, vs 92.6% for Dense)
- The CNN generalized better (smaller train/test gap) despite having more parameters, showing that its built-in spatial structure (convolution + pooling) is a more efficient inductive bias for image data than fully-connected layers
- Trade-off: CNN training is ~7-8x slower per step due to the added computation of convolution

## Tools
Python, TensorFlow/Keras, NumPy
