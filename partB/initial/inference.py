import numpy as np

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
    for i in range(num_layers):
        layer = model['layers'][i]
        f = layer['fwd_fn']
        curr, _, _ = f(curr, layer['params'], layer['hyper_params'], False)
        activations[i] = curr[:]
   
    ##########################
    output = activations[-1]

    return output, activations
