# program to load the MNIST dataset and save it in a format suitable for training with PyTorch

# load libraries
import pickle  # Use cPickle for faster loading of pickle files
import gzip
import numpy as np

def load_data():
    """
    Load the MNIST dataset from a gzipped pickle file.
    
    Returns:
        training_data: A tuple containing the training images and labels.
        validation_data: A tuple containing the validation images and labels.
        test_data: A tuple containing the test images and labels.
    """
    f = gzip.open('../neural-networks-and-deep-learning/data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')  # Use 'latin1' encoding for compatibility with Python 3
    f.close()
    
    return training_data, validation_data, test_data

def load_data_wrapper():
    """
    Load the MNIST dataset and convert it into a format suitable for PyTorch.
    
    Returns:
        training_data: A tuple containing training images as PyTorch tensors and labels.
        validation_data: A tuple containing validation images as PyTorch tensors and labels.
        test_data: A tuple containing test images as PyTorch tensors and labels.
    """
    training_data, validation_data, test_data = load_data()
    
    # Convert data to numpy arrays
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = list(zip(training_inputs, training_results))
    
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = list(zip(validation_inputs, validation_data[1]))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))
    
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    Convert a digit (0-9) into a one-hot encoded vector.
    
    Args:
        j: The digit to be converted.
        
    Returns:
        A one-hot encoded vector of size 10.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
