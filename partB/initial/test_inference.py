import sys
sys.path += ['layers', 'pyc_code']
from pyc_code.inference_ import inference as inference_
from pyc_code.calc_gradient_ import calc_gradient as calc_gradient_
import numpy as np
from init_layers import init_layers
from init_model import init_model
from inference import inference
from calc_gradient import calc_gradient
from layers.loss_euclidean import loss_euclidean
from data_utils import get_digits

def mse(val1, val2):
    tot = 0
    for i in range(len(val1)):
    tot += np.sum((val1 - val2) ** 2) / val1.size

def main():
    l = [init_layers('conv', {'filter_size': 2,
                              'filter_depth': 3,
                              'num_filters': 2}),
         init_layers('pool', {'filter_size': 2,
                              'stride': 2}),
         init_layers('relu', {}),
         init_layers('flatten', {}),
         init_layers('linear', {'num_in': 32,
                                'num_out': 10}),
         init_layers('softmax', {})]

    model = init_model(l, [10, 10, 3], 10, True)

    # Example calls you might make for inference:
    inp = np.random.rand(10, 10, 3, 3)    # Dummy input
    output, layer_acts = inference(model, inp)
    output_, layer_acts_ = inference_(model, inp)

    # Example calls you might make for calculating loss:
    loss, dv_output  = loss_euclidean(output, output_, {}, True)
    print(loss)

    # Calls for calculating gradients
    grad = calc_gradient(model, inp, layer_acts, dv_output)
    grad_ = calc_gradient_(model, inp, layer_acts, dv_output)
    
    print(mse([np.flatten(x['W']) for x in grad], [np.flatten(x['W']) for x in grad_]))
    print(mse([np.flatten(x['b']) for x in grad], [np.flatten(x['b']) for x in grad_]))


if __name__ == '__main__':
    main()