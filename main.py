import numpy as np
import scipy.linalg as la
import mindquantum as mq
from mindquantum.algorithm.library.qudit_mapping import mat_to_op
from mindquantum.core.gates import X, Y, Z, H, RX, RY, RZ, U3, BarrierGate
from mindquantum.core.circuit import Circuit 
from mindquantum.core.operators import TimeEvolution, QubitOperator, Hamiltonian
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.framework import MQLayer                    
from mindquantum.simulator import Simulator
from model import quantum_network
from data import load_data

print_file = open(r'D:\Computing server\quantum_convolutional_neural_network\mindspore_qcnn\result_3.txt', 'w')
def realtime_print(*args, **kwargs):
    print(*args, file=print_file, flush=True)

classifier = lambda t: t >= 0.5
vfunc      = np.vectorize(classifier)

### Implementation
num_qubits        = 9
random_seed      = 20250827
np.random.seed(random_seed)
active_qubits     = list(range(num_qubits))
index             = 0
label             = 'lc4'

### Load Data
train_data, test_data, labels, labels_test = load_data(num_qubits)
realtime_print('====================================================================')
realtime_print('training data shape: ', train_data.shape, '\ntest data shape: ', test_data.shape, '\nlength of labels: ', len(labels), '\nlength of labels (test data): ', len(labels_test))
realtime_print('random seed: ', random_seed)
realtime_print('====================================================================')


### Training 

def finite_difference_gradient(circ, params_dict, hams, train_data, y, delta=1e-5):
    """
    Compute gradients using finite difference method
    :param circ: quantum circuit
    :param params_dict: current parameter dictionary
    :param hams: Hamiltonian
    :param train_data: training data (single sample)
    :param delta: finite difference step size
    :return: gradient array
    """
    grad = np.zeros(len(params_dict))
    param_values = np.array(list(params_dict.values()))
    
    for i in range(len(param_values)):
        # Forward perturbation
        params_plus = param_values.copy()
        params_plus[i] += delta
        sim.set_qs(train_data)
        sim.apply_circuit(circ, dict(zip(params_dict.keys(), params_plus)))
        output = sim.get_expectation(hams).real
        E_plus = (output - y)**2
        
        # Backward perturbation
        params_minus = param_values.copy()
        params_minus[i] -= delta
        sim.set_qs(train_data)
        sim.apply_circuit(circ, dict(zip(params_dict.keys(), params_minus)))
        output = sim.get_expectation(hams).real
        E_minus = (output - y)**2
        
        # Central difference
        grad[i] = (E_plus - E_minus) / (2 * delta)
    
    return grad

def train_qcnn(
    circ, train_data, hams, init_params, lr=1e-2, epochs=500, 
    early_stop_patience=10, early_stop_delta=1e-7
):
    """
    QCNN training process with early stopping.
    :param circ: quantum circuit
    :param train_data: training data (as a list)
    :param hams: Hamiltonian
    :param init_params: initial parameters
    :param lr: learning rate
    :param epochs: number of training epochs
    :param early_stop_patience: number of epochs to wait before stopping if loss change is small
    :param early_stop_delta: minimum loss change to continue training
    :return: optimized parameters
    """
    params_dict = dict(zip(circ.params_name, init_params))
    sim = Simulator('mqvector', circ.n_qubits)
    loss_history = []
    stop_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        total_grad = np.zeros(len(init_params))
        
        for data, y in zip(train_data, labels):
            sim.set_qs(data)
            sim.apply_circuit(circ, params_dict)
            output = sim.get_expectation(hams).real
            current_loss = (output - y)**2
            
            grad = finite_difference_gradient(circ, params_dict, hams, data, y)
            
            total_loss += current_loss
            total_grad += grad
        
        avg_loss = total_loss / len(train_data)
        avg_grad = total_grad / len(train_data)
        
        # gradient descent update
        param_values = np.array(list(params_dict.values())) - lr * avg_grad
        params_dict = dict(zip(params_dict.keys(), param_values))
        
        realtime_print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
        loss_history.append(avg_loss)

        successive_loss = avg_loss - loss_history[-2] if epoch > 0 else None
        if successive_loss is not None: 
            if successive_loss <= 0:
                lr = 1.05 * lr
            else:
                lr = 0.5 * lr

        # Output params_dict every 50 epochs
        if (epoch + 1) % 20 == 0:
            realtime_print(f"Epoch {epoch+1}, params_dict: {params_dict}")

        # Early stopping check
        if epoch > 0:
            if abs(loss_history[-1] - loss_history[-2]) < early_stop_delta:
                stop_counter += 1
            else:
                stop_counter = 0
            if stop_counter >= early_stop_patience:
                realtime_print(f"Early stopping at epoch {epoch+1}.")
                break

    return params_dict



### Summary
circ = quantum_network(num_qubits, active_qubits)
circ.summary()
circ.svg()

param_names = circ.params_name 
# init_params = np.random.uniform(0, np.pi, len(param_names))      # Initialize random parameters
# init_params = np.zeros(len(param_names))                         # Initialize all parameters to zero
# params_dict = dict(zip(param_names, init_params))
params_dict = {'eps0': 0.3638384699722806, 'eps1': 0.0, 'eps2': 0.0, 'eps3': -0.06866882894194366, 'eps4': 0.0, 'eps5': 0.0, 'eps6': 0.0, 'eps7': -0.028144838771472565, 'eps8': 0.0185453415366362, 'eps9': -0.05806212446058586, 'eps10': 0.0, 'eps11': 0.0, 'eps12': 0.36172794112789175, 'eps13': 0.0, 'eps14': 0.0, 'eps15': -0.8793579810617033, 'eps16': 0.0, 'eps17': 0.0, 'eps18': -0.1434542115702317, 'eps19': 0.0, 'eps20': 0.0, 'eps21': 0.0, 'eps22': -0.24874476044349436, 'eps23': -0.34704497649406296, 'eps24': -0.07432370098862683, 'eps25': 0.0, 'eps26': 0.0, 'eps27': -0.9800652998706906, 'eps28': 0.0, 'eps29': 0.0, 'eps30': -0.07432370098042673, 'eps31': 0.0, 'eps32': 0.0, 'eps33': 0.36172794111751727, 'eps34': 0.0, 'eps35': 0.0, 'eps36': 0.0, 'eps37': 0.0976700640637472, 'eps38': -0.34484032113899404, 'eps39': 0.18050946099499846, 'eps40': 0.0, 'eps41': 0.0, 'eps42': -0.07346002710491081, 'eps43': 0.0, 'eps44': 0.0, 'eps45': 0.1805094610245554, 'eps46': 0.0, 'eps47': 0.0, 'eps48': -0.05806212449054518, 'eps49': 0.0, 'eps50': 0.0, 'eps51': 0.0, 'eps52': 0.0022354400088153608, 'eps53': -0.016017335983845273, 'eps54': -0.06893635786948521, 'eps55': 0.0, 'eps56': 0.0, 'eps57': 0.1776799733090015, 'eps58': 0.0, 'eps59': 0.0, 'eps60': -0.9800652998500666, 'eps61': 0.0, 'eps62': 0.0, 'eps63': -0.07346002707912636, 'eps64': 0.0, 'eps65': 0.0, 'eps66': 0.0, 'eps67': 0.216220068821173, 'eps68': -0.6694723623422508, 'eps69': -0.016435816701077653, 'eps70': 0.0, 'eps71': 0.0, 'eps72': -0.996919272303454, 'eps73': 0.0, 'eps74': 0.0, 'eps75': -0.016435816672280987, 'eps76': 0.0, 'eps77': 0.0, 'eps78': 0.17767997328077356, 'eps79': 0.0, 'eps80': 0.0, 'eps81': 0.0, 'eps82': -0.013956136971201728, 'eps83': -0.4052809175309996, 'eps84': 0.0007905573027004809, 'eps85': 0.0, 'eps86': 0.0, 'eps87': -0.009994710506112605, 'eps88': 0.0, 'eps89': 0.0, 'eps90': 0.19926927778037282, 'eps91': 0.0, 'eps92': 0.0, 'eps93': 0.00191889937430712, 'eps94': 0.0, 'eps95': 0.0, 'eps96': 0.0, 'eps97': -0.07559795984350373, 'eps98': 0.26524181136411035, 'eps99': -0.025984013342208023, 'eps100': 0.0, 'eps101': 0.0, 'eps102': 0.1917980702927784, 'eps103': 0.0, 'eps104': 0.0, 'eps105': -0.009994710525505292, 'eps106': 0.0, 'eps107': 0.0, 'eps108': -0.9969192722798356, 'eps109': 0.0, 'eps110': 0.0, 'eps111': 0.0, 'eps112': -0.9490897251033109, 'eps113': 0.21595034842791877, 'eps114': -1.0465879263415505, 'eps115': 0.0, 'eps116': 0.0, 'eps117': -0.07273436071820409, 'eps118': 0.0, 'eps119': 0.0, 'eps120': -1.046587926352025, 'eps121': 0.0, 'eps122': 0.0, 'eps123': 0.1917980702487114, 'eps124': 0.0, 'eps125': 0.0, 'eps126': 0.0, 'eps127': -0.028731112553550438, 'eps128': -0.04019504716157139, 'eps129': 0.11062756331499937, 'eps130': 0.0, 'eps131': 0.0, 'eps132': -1.038388366372677, 'eps133': 0.0, 'eps134': 0.0, 'eps135': 0.11062756328296959, 'eps136': 0.0, 'eps137': 0.0, 'eps138': -0.025984013404682687, 'eps139': 0.0, 'eps140': 0.0, 'eps141': 0.0, 'eps142': -0.12239740580562175, 'eps143': 0.14157341637078108, 'eps144': -0.04183880026632593, 'eps145': 0.0, 'eps146': 0.0, 'eps147': 0.10265776026623254, 'eps148': 0.0, 'eps149': 0.0, 'eps150': -0.07273436073181473, 'eps151': 0.0, 'eps152': 0.0, 'eps153': -1.0383883664097304, 'eps154': 0.0, 'eps155': 0.0, 'eps156': 0.0, 'eps157': -0.8665770441206954, 'eps158': 0.08600650822013618, 'eps159': -1.0371054782369544, 'eps160': 0.0, 'eps161': 0.0, 'eps162': -0.10353944452806045, 'eps163': 0.0, 'eps164': 0.0, 'eps165': -1.0371054782450284, 'eps166': 0.0, 'eps167': 0.0, 'eps168': 0.10265776026605536, 'eps169': 0.0, 'eps170': 0.0, 'eps171': 0.0, 'eps172': -0.003337055652471005, 'eps173': -0.05111794752239585, 'eps174': 0.0002568529655188498, 'eps175': 0.0, 'eps176': 0.0, 'eps177': -1.0362557462637363, 'eps178': 0.0, 'eps179': 0.0, 'eps180': -0.09426913115481388, 'eps181': 0.0, 'eps182': 0.0, 'eps183': -0.07580941373939486, 'eps184': 0.0, 'eps185': 0.0, 'eps186': 0.0, 'eps187': -0.0805602401545834, 'eps188': 0.026374266403018466, 'eps189': -0.07903754688877344, 'eps190': 0.0, 'eps191': 0.0, 'eps192': -0.07842445390666439, 'eps193': 0.0, 'eps194': 0.0, 'eps195': -0.10353944453650323, 'eps196': 0.0, 'eps197': 0.0, 'eps198': -0.2292931990115905, 'eps199': 0.0, 'eps200': 0.0, 'eps201': 0.0, 'eps202': -0.17821848141513058, 'eps203': -0.12952316495582683, 'eps204': -0.22773301727081158, 'eps205': 0.0, 'eps206': 0.0, 'eps207': -0.09145523897709092, 'eps208': 0.0, 'eps209': 0.0, 'eps210': -0.22773301730194692, 'eps211': 0.0, 'eps212': 0.0, 'eps213': -0.07842445388772848, 'eps214': 0.0, 'eps215': 0.0, 'eps216': 0.0, 'eps217': 0.22285448626199347, 'eps218': -0.048407377782327246, 'eps219': -0.07756536717995674, 'eps220': 0.0, 'eps221': 0.0, 'eps222': -0.1539145716026398, 'eps223': 0.0, 'eps224': 0.0, 'eps225': -0.07756536717966504, 'eps226': 0.0, 'eps227': 0.0, 'eps228': -0.07903754694373598, 'eps229': 0.0, 'eps230': 0.0, 'eps231': 0.0, 'eps232': 0.12169959914970038, 'eps233': -0.005505532527791497, 'eps234': -0.07870736320253925, 'eps235': 0.0, 'eps236': 0.0, 'eps237': 0.023262704901664495, 'eps238': 0.0, 'eps239': 0.0, 'eps240': -0.09145523898405061, 'eps241': 0.0, 'eps242': 0.0, 'eps243': -0.15391457160844588, 'eps244': 0.0, 'eps245': 0.0, 'eps246': 0.0, 'eps247': -0.12948327937193008, 'eps248': -0.12407659068065889, 'eps249': -0.1490403546656118, 'eps250': 0.0, 'eps251': 0.0, 'eps252': -0.08742650871203222, 'eps253': 0.0, 'eps254': 0.0, 'eps255': -0.14904035464298648, 'eps256': 0.0, 'eps257': 0.0, 'eps258': 0.023262704923372055, 'eps259': 0.0, 'eps260': 0.0, 'eps261': 0.0, 'eps262': 0.4901532523807611, 'eps263': -0.04733508567458285, 'eps264': 0.023232220822288298, 'eps265': 0.0, 'eps266': 0.0, 'eps267': -0.0008582528961781538, 'eps268': 0.0, 'eps269': 0.0, 'alpha0': -0.06893635792837155, 'alpha1': 0.0, 'alpha2': 0.0, 'alpha3': 0.0007905572937032681, 'alpha4': 0.0, 'alpha5': 0.0, 'alpha6': 0.0, 'alpha7': -0.05846499286447851, 'alpha8': 0.019379246943462162, 'alpha9': -0.0005288236084733542, 'alpha10': 0.0, 'alpha11': 0.0, 'alpha12': -0.0690201639731696, 'alpha13': 0.0, 'alpha14': 0.0, 'alpha15': -0.06902016398618427, 'alpha16': 0.0, 'alpha17': 0.0, 'alpha18': -0.04183880026270463, 'alpha19': 0.0, 'alpha20': 0.0, 'alpha21': 0.0, 'alpha22': -0.06562309715885149, 'alpha23': 0.029152461147173143, 'alpha24': -0.029847996740063807, 'alpha25': 0.0, 'alpha26': 0.0, 'alpha27': -0.06846687040746846, 'alpha28': 0.0, 'alpha29': 0.0, 'alpha30': -0.06846687039076647, 'alpha31': 0.0, 'alpha32': 0.0, 'alpha33': 0.00025685294502349415, 'alpha34': 0.0, 'alpha35': 0.0, 'alpha36': 0.0, 'alpha37': 0.009763623596815535, 'alpha38': -0.06120546397650048, 'alpha39': 0.00022463269196883073, 'alpha40': 0.0, 'alpha41': 0.0, 'alpha42': -0.06843907571815423, 'alpha43': 0.0, 'alpha44': 0.0, 'alpha45': -0.06843907574946792, 'alpha46': 0.0, 'alpha47': 0.0, 'alpha48': -1.036255746279408, 'alpha49': 0.0, 'alpha50': 0.0, 'alpha51': 0.0, 'alpha52': 0.007257591203422542, 'alpha53': 0.045158557759763275, 'alpha54': -1.031730189114354, 'alpha55': 0.0, 'alpha56': 0.0, 'alpha57': -0.0317914782181346, 'alpha58': 0.0, 'alpha59': 0.0, 'alpha60': -0.03179147822235178, 'alpha61': 0.0, 'alpha62': 0.0, 'alpha63': -0.07870736317025359, 'alpha64': 0.0, 'alpha65': 0.0, 'alpha66': 0.0, 'alpha67': 0.0033971346876933294, 'alpha68': 0.001081288474139224, 'alpha69': -0.07865936472342816, 'alpha70': 0.0, 'alpha71': 0.0, 'alpha72': 0.0012221749015515718, 'alpha73': 0.0, 'alpha74': 0.0, 'alpha75': 0.0012221748906684488, 'alpha76': 0.0, 'alpha77': 0.0, 'alpha78': 0.023232220860297567, 'alpha79': 0.0, 'alpha80': 0.0, 'alpha81': 0.0, 'alpha82': 0.059571119308888046, 'alpha83': -0.019632729270791134, 'alpha84': 0.023213549379912338, 'alpha85': 0.0, 'alpha86': 0.0, 'alpha87': 0.002064981212581344, 'alpha88': 0.0, 'alpha89': 0.0, 'alpha90': 0.0020649812258196117, 'alpha91': 0.0, 'alpha92': 0.0, 'alpha93': -0.0008582529362952026, 'alpha94': 0.0, 'alpha95': 0.0, 'alpha96': 0.0, 'alpha97': -0.0006765728154957375, 'alpha98': -0.00037661648481041233, 'alpha99': -0.0008589640697967507, 'alpha100': 0.0, 'alpha101': 0.0, 'alpha102': 0.002065093314187346, 'alpha103': 0.0, 'alpha104': 0.0, 'alpha105': 0.002065093333064334, 'alpha106': 0.0, 'alpha107': 0.0, 'alpha108': -0.08742650870884863, 'alpha109': 0.0, 'alpha110': 0.0, 'alpha111': 0.0, 'alpha112': -0.11715701169490034, 'alpha113': -0.005417727243008444, 'alpha114': -0.08742916736205433, 'alpha115': 0.0, 'alpha116': 0.0, 'alpha117': 0.0005080379444916536, 'alpha118': 0.0, 'alpha119': 0.0, 'beta0': -0.04124905620456531, 'beta1': 0.0, 'beta2': 0.0, 'beta3': -0.01403023812503221, 'beta4': 0.0, 'beta5': 0.0, 'beta6': -2.890303241566132, 'beta7': 0.0, 'beta8': 0.0, 'beta9': -0.6864545979640578, 'beta10': 0.0, 'beta11': 0.0, 'beta12': 0.018365297548386166, 'beta13': 0.0, 'beta14': 0.0, 'beta15': 0.03053216586415442, 'beta16': 0.0, 'beta17': 0.0, 'gamma0': -0.08230393657487671, 'gamma1': 0.0, 'gamma2': 0.0, 'gamma3': 0.008722909520132903, 'gamma4': 0.0, 'gamma5': 0.0}
init_params = np.array([params_dict[name] for name in circ.params_name])
hams = Hamiltonian(QubitOperator('X4'))

sim = Simulator('mqvector', num_qubits)
sim.set_qs(train_data[0])
initial_qs = sim.get_qs()
# print('Initial quantum state: ', initial_qs)
sim.apply_circuit(circ, pr=params_dict)
current_qs = sim.get_qs()
# print('Current quantum state: ', current_qs)
# print(np.allclose(initial_qs, current_qs, rtol=1e-5, atol=1e-8)) 
result = sim.get_expectation(hams).real
# result = result.real()
# print('result: ', result, vfunc(result))


optimized_params = train_qcnn(
    circ=circ,
    train_data=train_data,
    hams=hams,
    init_params=init_params,
    lr=0.1,
    epochs=500
)

realtime_print('optimized_params: ', optimized_params)


