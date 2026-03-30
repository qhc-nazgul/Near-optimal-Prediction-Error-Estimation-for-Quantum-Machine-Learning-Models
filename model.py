import numpy as np
from mindquantum.core.circuit import Circuit 
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.core.gates import X, Y, Z, H, RX, RY, RZ, U3, BarrierGate



def quantum_network(num_qubits, active_qubits, index=0):
    """
    Quantum Convolutional Neural Network (QCNN) implementation using MindQuantum.
    The QCNN consists of convolutional layers, pooling layers, and a fully connected layer.
    """

    circ = Circuit() 

    ### Convolutional Layer 1 
    params_conv = np.random.uniform(low=0, high=np.pi, size=15)
    prg0 = PRGenerator('eps')
    while index + 3 < len(active_qubits):
        q_index = active_qubits[index]
        q_index_1 = active_qubits[index + 1]
        q_index_2 = active_qubits[index + 2]
        q_index_3 = active_qubits[index + 3]

        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_2)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_3)
        circ += X.on(q_index_2, q_index_3)
        circ += RZ(prg0.new()).on(q_index_2)
        circ += RY(prg0.new()).on(q_index_3)
        circ += X.on(q_index_3, q_index_2)
        circ += RY(prg0.new()).on(q_index_3)
        circ += X.on(q_index_2, q_index_3)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_2)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_3)

        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_1)
        circ += X.on(q_index, q_index_1)
        circ += RZ(prg0.new()).on(q_index)
        circ += RY(prg0.new()).on(q_index_1)
        circ += X.on(q_index_1, q_index)
        circ += RY(prg0.new()).on(q_index_1)
        circ += X.on(q_index, q_index_1)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_1)

        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_3)
        circ += X.on(q_index, q_index_3)
        circ += RZ(prg0.new()).on(q_index)
        circ += RY(prg0.new()).on(q_index_3)
        circ += X.on(q_index_3, q_index)
        circ += RY(prg0.new()).on(q_index_3)
        circ += X.on(q_index, q_index_3)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_3)

        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_2)
        circ += X.on(q_index, q_index_2)
        circ += RZ(prg0.new()).on(q_index)
        circ += RY(prg0.new()).on(q_index_2)
        circ += X.on(q_index_2, q_index)
        circ += RY(prg0.new()).on(q_index_2)
        circ += X.on(q_index, q_index_2)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_2)

        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_1)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_3)
        circ += X.on(q_index_1, q_index_3)
        circ += RZ(prg0.new()).on(q_index_1)
        circ += RY(prg0.new()).on(q_index_3)
        circ += X.on(q_index_3, q_index_1)
        circ += RY(prg0.new()).on(q_index_3)
        circ += X.on(q_index_1, q_index_3)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_1)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_3)
        
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_1)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_2)
        circ += X.on(q_index_1, q_index_2)
        circ += RZ(prg0.new()).on(q_index_1)
        circ += RY(prg0.new()).on(q_index_2)
        circ += X.on(q_index_2, q_index_1)
        circ += RY(prg0.new()).on(q_index_2)
        circ += X.on(q_index_1, q_index_2)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_1)
        circ += U3(prg0.new(), prg0.new(), prg0.new()).on(q_index_2)
        
        if index == 0:
            index += 2
        else:
            index += 3

    for qubit in range(num_qubits):
        circ += BarrierGate().on(qubit) 

    ### Convolutional Layer 2 
    prg1 = PRGenerator('alpha')
    for q in range(num_qubits-1):
        q_index = q
        q_index_1 = q + 1
        
        circ += U3(prg1.new(), prg1.new(), prg1.new()).on(q_index)
        circ += U3(prg1.new(), prg1.new(), prg1.new()).on(q_index_1)
        circ += X.on(q_index, q_index_1)
        circ += RZ(prg1.new()).on(q_index)
        circ += RY(prg1.new()).on(q_index_1)
        circ += X.on(q_index_1, q_index)
        circ += RY(prg1.new()).on(q_index_1)
        circ += X.on(q_index, q_index_1)
        circ += U3(prg1.new(), prg1.new(), prg1.new()).on(q_index)
        circ += U3(prg1.new(), prg1.new(), prg1.new()).on(q_index_1)

    ### Pooling layer
    prg2 = PRGenerator('beta')
    index = 0

    while index + 2 < len(active_qubits):
        q_index = active_qubits[index]        # control index 1
        q_index_1 = active_qubits[index + 1]  # target index
        q_index_2 = active_qubits[index + 2]  # control index 2

        circ += H.on(q_index)
        circ += U3(prg2.new(), prg2.new(), prg2.new()).on(q_index_1, q_index)
        circ += H.on(q_index_1)
        circ += U3(prg2.new(), prg2.new(), prg2.new()).on(q_index_1, q_index_2)
        index += 3

    ### Fully connected layer
    prg3 = PRGenerator('gamma')
    index = 0

    circ += H.on(1)
    circ += U3(prg3.new(), prg3.new(), prg3.new()).on(4, 1)
    circ += H.on(7)
    circ += U3(prg3.new(), prg3.new(), prg3.new()).on(4, 7)
    index += 3

    return circ



