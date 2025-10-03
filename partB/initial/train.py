import sys
sys.path += ['layers']
import numpy as np
from layers.loss_crossentropy import loss_crossentropy
import copy

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = True

# You can modify the imports of this section to indicate
# whether to use the provided pyc or your own code for each of the four functions.
if use_pcode:
    # import the provided pyc implementation
    sys.path += ['pyc_code']
    from pyc_code.inference_ import inference
    from pyc_code.calc_gradient_ import calc_gradient
    from pyc_code.update_weights_ import update_weights
else:
    # import your own implementation
    from inference import inference
    from calc_gradient import calc_gradient
    from update_weights import update_weights
######################################################

def train(model, input, label, val_input, val_label, params, numIters, numItersPerEval):
    '''
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [num_inputs]
        label: [num_inputs]
        val_input: [any_dimensions] x [val_num_inputs]
        val_label: [val_num_inputs]
        params: Paramters for configuring training
            params["learning_rate"]
            params["weight_decay"]
            params["batch_size"]
            params["save_file"]
            Free to add more parameters to this dictionary for your convenience of training.
        numIters: Number of training iterations
        numItersPerEval: Number of iterations before computing loss / accuracy on validation set 
    '''
    # Initialize training parameters
    # Learning rate
    lr = params.get("learning_rate", .01)
    # Weight decay
    wd = params.get("weight_decay", .0005)
    # Batch size
    batch_size = params.get("batch_size", 128)
    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", 'model.npz')

    # Optimization method, default SGD
    optimization = params.get("optimization", "SGD")
    # momentum friction
    rho = params.get("rho", 0.9)

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr,
                     "weight_decay": wd }

    train_accuracies, train_losses = [], []
    val_accuracies, val_losses = [], []

    num_inputs = input.shape[-1]
    loss = np.zeros((numIters,))

    num_val = val_input.shape[-1]
    

    best_accuracy = 0
    best_model = {}
    num_layers = len(model["layers"])
    velocities = [None, ] * num_layers

    num_inputs = input.shape[-1]
    num_val = val_input.shape[-1]


    epoch_loss = 0
    for i in range(numIters):
        # TODO: One training iteration
        # Steps:
        #   (1) Select a subset of the input to use as a batch
        #   (2) Run inference on the batch
        #   (3) Calculate loss and determine accuracy
        #   (4) Calculate gradients
        #   (5) Update the weights of the model
        # Optionally,
        #   (1) Monitor the progress of training
        #   (2) Save your learnt model, using ``np.savez(save_file, **model)``

        j = np.random.randint(0, num_inputs-batch_size)
        batch = input[...,j:j+batch_size] # random sample of data 
        batch_labels = label[j:j+batch_size]

        #forward
        output, activations = inference(model, batch)
        
        #loss
        batch_loss, dv_output = loss_crossentropy(output, batch_labels, {}, True)
    
        #accuracy
        train_accuracies.append(np.sum(np.array([np.argmax(output[j])==np.argmax(batch_labels[j]) for j in range(batch_size)]))/batch_size)
        train_losses.append(batch_loss)

        val_output, _ = inference(model, val_input)
        val_loss, _ = loss_crossentropy(val_output, val_label, {}, False)
        val_accuracies.append(np.sum(np.array([np.argmax(val_label[j])==np.argmax(val_output[j]) for j in range(len(val_input))]))/len(val_input))
        val_losses.append(val_loss)
        
        #backprop
        grads = calc_gradient(model, batch, activations, dv_output)

        #update 
        model= update_weights(model, grads, update_params)
        
        print(f"Iter {i+1}/{numIters} | "
            f"Train loss {batch_loss:.4f}, acc {train_accuracies:.4f} | "
            f"Val loss {val_loss:.4f}, acc {val_accuracies:.4f}")
    
    ########
    return model, loss, val_losses, val_accuracies, train_losses, train_accuracies 
