# NeuralDigit
Basic pytorch neural network to recognize handwriten digits, trained with the MNIST Digit dataset.

## Usage

To train model from scratch:

`python main.py train`

To resume training from previously trained model:

`python main.py resume`

To test the model on the test dataset:

`python main.py test`

To use the model on an input image (must be 28*28 in grayscale):
`python main.py use <input_file.jpeg>`