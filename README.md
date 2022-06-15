This exercise is a demonstration of transfer learning using TF and keras.

A base model is created to classify the MNIST dataset (src/01_base_model_creation.py). This model is saved as base_model.h5 file. 

Another "binary" base model is created to classify the MNIST dataset as odd or even numbers (src/01.01_base_model_creation.py). This model is saved as bin_base_model.h5.

Using the weights of the base model a "transfer learning" model is created (src/02_transfer_learning_even_odd.py) modifying the final output layer to classify the digits in the MNIST dataset as odd or even. It is saved as even_odd_model.h5. 

The base model's weights are used for also classifying digits as greater than 5 or not (src/03_transfer_learning_greater_than_5.py) and saved as greater_than_5_model.h5. 