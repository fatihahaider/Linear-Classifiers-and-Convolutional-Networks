import numpy as np
from layers import fn_conv, fn_flatten, fn_linear, fn_pool, fn_relu, fn_softmax, loss_crossentropy, loss_euclidean

def inference(model, input):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """

    num_layers = len(model['layers'])
    activations = [None,] * num_layers

    # TODO: FORWARD PROPAGATION CODE
    curr = input
    params = model['']
    for i in range(model['layers']):
        layer = model['layers']
        type = layer['type']
        f = layer['fwd_fn']
        params = layer['paramers']
        curr = f(curr, params[0], params[1], params[2])
        activations[i] = curr[:]
   
    ##########################
    output = activations[-1]

    return output, activations
