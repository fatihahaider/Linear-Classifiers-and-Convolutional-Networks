import numpy as np

def update_weights(model, grads, hyper_params):
    '''
    Update the weights of each layer in your model based on the calculated gradients
    Args:
        model: Dictionary holding the model
        grads: A list of gradients of each layer in model["layers"]
        hyper_params: 
            hyper_params['learning_rate']
            hyper_params['weight_decay']: Should be applied to W only.
    Returns: 
        updated_model:  Dictionary holding the updated model
    '''
    num_layers = len(grads)
    a = hyper_params["learning_rate"]
    lmd = hyper_params["weight_decay"]
    updated_model = model

    # TODO: Update the weights of each layer in your model based on the calculated gradients
    layers = model['layers']
    for i in range(num_layers):
        layer = layers[i]
        params = layer['params']
        grad = grads[i]

        if 'W' in params: # some layers dont have weights 
            params['W'] -= a * (grad['W'] + lmd * params['W'])
        if 'b' in params:
            params['b'] -= a * grad['b']

        # #grad['W'] += lmd/a * params['W'] 
        # #grad['W'] += lmd * params['W']
        # params['W'] += -lmd * params['W']  # L2 norm of weights
        # params['W'] += -a * grad['W']
        # params['b'] += -a * grad['b']

        updated_model['layers'][i]['params'] = params
    
    ########
    
    return updated_model