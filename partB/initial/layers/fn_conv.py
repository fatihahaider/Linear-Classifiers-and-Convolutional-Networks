import numpy as np
import scipy.signal

def fn_conv(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [filter_height] x [filter_width] x [filter_depth] x [num_filters] array
            params['b']: layer bias, [num_filters] x 1 array
        hyper_params: Optional, could include information such as stride and padding.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [out_height] x [out_width] x [num_filters] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """

    in_height, in_width, num_channels, batch_size = input.shape
    _, _, filter_depth, num_filters = params['W'].shape
    out_height = in_height - params['W'].shape[0] + 1
    out_width = in_width - params['W'].shape[1] + 1

    assert params['W'].shape[2] == input.shape[2], 'Filter depth does not match number of input channels'

    # Initialize
    output = np.zeros((out_height, out_width, num_filters, batch_size))
    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}
    
    # TODO: FORWARD CODE
    #       Update output with values
    for k in range(batch_size):
        for l in range(num_filters):
            #output[:,:,l,k] = np.vectorize(lambda x: fn_relu(x, 0, 0, False)[0])(scipy.signal.convolve(input[:,:,:,k], params['W'][:,:,:,l], mode='valid')[:,:,0] + params['b'][l,0])
            output[:,:,l,k] = scipy.signal.correlate(input[:,:,:,k], params['W'][:,:,:,l], mode='valid')[:,:,0] + params['b'][l,0]

    #######################

    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad['W'] = np.zeros(params['W'].shape)
        grad['b'] = np.zeros(params['b'].shape)
        
        # TODO: BACKPROP CODE
        #       Update dv_input and grad with values
        for l in range(num_filters):
            for j in range(filter_depth):
                sum = 0
                for k in range(batch_size):
                    sum += scipy.signal.correlate(input[:,:,j,k], dv_output[:,:,l,k], mode='valid')
                grad['W'][:,:,j,l] = sum/batch_size

        for l in range(num_filters):
            sum = 0
            for k in range(batch_size):
                sum += np.sum(dv_output[:,:,l,k])
            grad['b'][l,0] = sum / batch_size

        #for l in range(num_filters):
         #   for j in range(filter_depth):
          ##         dv_input[:,:,j,k] += scipy.signal.correlate(params['W'][:,:,j,l], dv_output[:,:,l,k], mode='valid')

        #for l in range(num_filters):
        #    for j in range(filter_depth):
        #        for k in range(batch_size):
        #            result = scipy.signal.correlate(np.flip(params['W'][:,:,j,l]), dv_output[:,:,l,k], mode='full')
        #            dv_input[:,:,j,k] = result

        for j in range(filter_depth):
            for k in range(batch_size):
                sum = 0
                for l in range(num_filters):
                    sum += scipy.signal.convolve(params['W'][:,:,j,l], dv_output[:,:,l,k], mode='full')
                dv_input[:,:,j,k] = sum

        #for j in range(filter_depth):
         #   for k in range(batch_size):
          #      result = scipy.signal.correlate(params['W'][:,:,j,:], dv_output[:,:,:,k],)
           #     dv_input[:,:,j,k] = result

        #for l in range(num_filters):
        #    for k in range(batch_size):
        #        result = scipy.signal.correlate(params['W'][:,:,:,l], dv_output[:,:,l,k])
    

        ###########################
    return output, dv_input, grad
