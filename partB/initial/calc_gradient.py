import numpy as np

def calc_gradient(model, input, layer_acts, dv_output):
    '''
    Calculate the gradient at each layer, to do this you need dv_output
    determined by your loss function and the activations of each layer.
    The loop of this function will look very similar to the code from
    inference, just looping in reverse.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        layer_acts: A list of activations of each layer in model["layers"]
        dv_output: The partial derivative of the loss with respect to each element in the output matrix of the last layer.
    Returns: 
        grads:  A list of gradients of each layer in model["layers"]
    '''
    num_layers = len(model['layers'])
    grads = [None,] * num_layers

    # TODO: Determine the gradient at each layer.
    #       Remember that back-propagation traverses 
    #       the model in the reverse order.
    # loop backwards
    for i in reversed(range(num_layers)):
        layer = model['layers'][i]
        f = layer['fwd_fn']
        
        if i == 0:
            layer_input = input
        else:
            layer_input = layer_acts[i-1]

        _, dv_input, grad = f(layer_input, layer['params'], layer['hyper_params'], True, dv_output)

        grads[i] = grad
        dv_output = dv_input
        
    ##############
    
    return grads
