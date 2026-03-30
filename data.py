import numpy as np

def load_data(num_qubits):
    """
    Load training and test data from text files.
    Each data point is represented as a complex vector (eigenvector).
    The corresponding Hamiltonian parameters (h1, h2) are also loaded.
    :param num_qubits: number of qubits (defines the dimension of the eigenvectors)
    :return: training and test data along with their Hamiltonian parameters
    """

    ### Data 
    training_fname = "./data_fldr/dataset_n={}_train.txt".format(num_qubits)
    test_fname = "./data_fldr/dataset_n={}_test.txt".format(num_qubits)

    def read_eigenvectors(file):
        with open(file, 'r+') as f:
            textData = f.readlines()

            h_vals = []
            for i in range(len(textData)):
                h1h2, eigenvector = textData[i].split("_")

                h_vals.append(tuple(map(float, h1h2[1: -1].split(', '))))
                textData[i] = eigenvector

            return h_vals, np.loadtxt(textData, dtype=complex)

    h1h2_train, train_data = read_eigenvectors(training_fname)
    h1h2_test, test_data = read_eigenvectors(test_fname)

    # To get the correct labels of the training set we use the fact that data points with
    # h1 <= 1 are in the SPT phase and thus assigned the label 1 while h1 > 1 are in the paramagnetic phase 
    # This is only true for the training set which has h2 = 0 for all samples, 

    labels = np.zeros(40)
    labels_test = np.zeros(4096)

    for index, h1h2 in enumerate(h1h2_train):
        h1, h2 = h1h2
        if h1 <= 1: 
            labels[index] = 1.0

    for index, h1h2 in enumerate(h1h2_test):
        h1, h2 = h1h2
        if h1 <= 1: 
            labels_test[index] = 1.0

    return train_data, test_data, labels, labels_test

    