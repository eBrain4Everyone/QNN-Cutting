# This file is the user-friendly version of the wire cutting program that prompts user input to build
# a hybrid quantum circuit and runs it

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from silence_tensorflow import silence_tensorflow
import tensorflow as tf
silence_tensorflow()
import cutter as cut

import pennylane as qml
import numpy as np
import tensorflow as tf
import tf_keras as keras
import ast
from sklearn.datasets import load_digits
import time
tf.keras.backend.set_floatx('float32')

#initialize data here as X_train, y_train, X_test, y_test

n_train = int(input("Enter amount of training data for MNIST dataset: "))    # Size of the train dataset
n_test = int(input("Enter the amount of testing data: "))     # Size of the test dataset

mnist_dataset = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist_dataset.load_data()

X_train=X_train.reshape((len(X_train), -1))
X_test=X_test.reshape((len(X_test), -1))
y_train=y_train.reshape((len(y_train), -1))
y_test=y_test.reshape((len(y_test), -1))


# Reduce dataset size
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_test = X_test[:n_test]
y_test = y_test[:n_test]

from sklearn.preprocessing import normalize
X_train = normalize(X_train)
X_test = normalize(X_test)



print("""\n\nHello! This program will build a large qubit quantum machine learning model on smaller qubit devices.
\nIt will build a hybrid quantum neural network built of classical and quantum layers. There will be classical layers for pre-processing and postprocessing.
The classical layers will be Keras Dense layers made of a specified amount of neurons. The quantum layer consists of single qubit parametrized gates and double qubit nonparametrized gates.This program will walk you through the process of building a circuit.
\nPlease read the instructions and follow it carefully. Thank you! \n""")

#model details
input_shape=input("Please enter the data shape of each data entry as an array (e.g. (784,)): ")
input_shape=ast.literal_eval(input_shape)

num_classical_before=int(input("Please enter the number of classical dense layers you wish to use before the quantum: "))

print(f"You entered {num_classical_before} classical layers BEFORE the quantum circuit\n")
list_neurons_before=[]
list_activation_before = []  # declare each layer activation function, if none, put None var in list

if num_classical_before!=0:
    for i in range(num_classical_before):

        prompt=f"Please enter the number of neurons for classical layer #{i}"
        if i+1==num_classical_before:
            prompt+=", this is the last layer before the quantum circuit. The neurons must match the number of qubits in your full circuit"
        x=int(input(prompt + ": "))
        list_neurons_before.append(x)
        x = input(f"Please enter the activation function for classical layer #{i}, use None if no activation: ")
        if x == "None":
            list_activation_before.append(None)
        else:
            list_activation_before.append(x)

while True:
    old_qubits = int(input("Please enter the number of qubits your large circuit uses: "))
    if old_qubits != list_neurons_before[-1]:
        print("Number of qubits do not match number of neurons in previous classical layer")
        continue
    else:
        break

q_string, num_params, num_cnot, is_hadamard, hadamard_list =cut.make_string(old_qubits)

n_layers = int(input("Please enter the number of layers/repetitions of your circuit you wish to make: "))  # Number of repetitions for qnode

print("All measurements of the quantum layer are measured using PauliZ observable.")
measure_list = input("Please enter the wires you wish to measure. Use commas in between the wire numbers, e.g. 0,2,4 or All: ")  # Number of repetitions for qnode
if measure_list=='All':
    measure_list=range(old_qubits)
else:
    new_measure_list=[]
    for x in measure_list.split(','):
        new_measure_list.append(int(x))
    measure_list=new_measure_list

num_classical_after=int(input("Please enter the number of classical dense layers you wish to use AFTER the quantum: "))

print(f"You entered {num_classical_after} classical layers after the quantum circuit\n")
list_neurons_after=[]
list_activation_after = []  # declare each layer activation function, if none, put None var in list

if num_classical_after != 0:
    for i in range(num_classical_after):

        prompt=f"Please enter the number of neurons for classical layer #{i}"
        if i+1==num_classical_after:
            prompt+=", this is the last layer. The neurons must match the number of categories your model must identify"
        x=int(input(prompt + ": "))
        list_neurons_after.append(x)
        x = input(f"Please enter the activation function for classical layer #{i}, use None if no activation: ")
        if x == "None":
            list_activation_after.append(None)
        else:
            list_activation_after.append(x)

print("Here is a summary of your model:\n")
print(f"You have {num_classical_before} classical layers BEFORE the quantum circuit")
for i in range(num_classical_after):
    print(f"Before classical layer #{i}: {list_neurons_before[i]} Neurons and activation function of {list_activation_before[i]}")

print("\nQuantum Layer details:")
draw_string, ignored = cut.string_sort(q_string, n_layers)
full_circuit=cut.build_sub(draw_string, range(old_qubits), measure_list)
draw_weights = [x / 10.0 for x in range(num_params*n_layers)]
draw_input = [x / 10.0 for x in range(old_qubits)]


if is_hadamard=='Y':
    print(f"Applied Hadamard gates to the following wires: {hadamard_list}")
else:
    print("No Hadamard gates applied")
print(f"Your model has {num_params*n_layers} parametrized Rotation gates and {num_cnot*n_layers} Entangling gates" )

print("Full Circuit Diagram")
print(qml.draw(full_circuit)(draw_input, draw_weights))

print(f"\nYou have {num_classical_after} classical layers AFTER the quantum circuit")
for i in range(num_classical_after):
    print(f"After classical layer #{i}: {list_neurons_after[i]} Neurons and activation function of {list_activation_after[i]}")

loss_metric = input("Please enter the metric of loss form the Keras library (e.g. 'sparse_categorical_crossentropy': ") #declare loss type string
while True:
    target_qubits = int(input("Please enter the number of qubits of the smaller device: "))
    if old_qubits < target_qubits:
        print("Invalid values, smaller device has more qubits than full circuit\n")
        continue
    else:
        break

#training details
n_epochs = int(input("Please enter the number of training epochs of the model: "))   # Number of optimization epochs
n_batch = int(input("Please enter the batch size: ")) #batch size
print("The default optimizer is the Adam Optimizer with a 0.01 learning rate")
opt = tf.keras.optimizers.Adam(learning_rate=0.01) #declare optimizer

sorted_circuit, ignored_string =cut.string_sort(q_string, n_layers)

#initialize weights of model
weights = np.random.uniform(high=2 * np.pi, size=(1, num_params*n_layers))
weight_shapes = {"weights": num_params*n_layers}

cut_circuit, wire_list=cut.cut_placement(sorted_circuit,target_qubits)

cutmodel=cut.build_model(input_shape, num_classical_before,
            list_neurons_before,
            list_activation_before,
            num_classical_after,
            list_neurons_after,
            list_activation_after,
            cut_circuit, wire_list,
            weight_shapes,
            weights,
            num_params*n_layers,
            measure_list)

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Convert the model to a concrete function
model_func = tf.function(lambda x: cutmodel(x))
model_func = model_func.get_concrete_function(tf.TensorSpec([None, 784], tf.float32))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(model_func)
layers = [op.name for op in frozen_func.graph.get_operations()]

# Calculate FLOPs for the forward pass
run_meta = tf.compat.v1.RunMetadata()
opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

# Use the profiler to get the number of FLOPs for the forward pass
flops_forward = tf.compat.v1.profiler.profile(graph=frozen_func.graph, run_meta=run_meta, cmd='op', options=opts)
print(f"FLOPs (Forward Pass): {flops_forward.total_float_ops}")

# Define a dummy loss function for the backward pass
loss_object = tf.keras.losses.MeanSquaredError()

# Dummy input and target data for gradient computation
input_data = tf.random.uniform([784])
target_data = tf.random.uniform([10])

# Define the function to compute gradients
@tf.function
def compute_gradients(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = cutmodel(inputs)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, cutmodel.trainable_variables)
    return gradients

# Convert the compute_gradients function to a concrete function
grad_func = compute_gradients.get_concrete_function(tf.TensorSpec([None,784], tf.float32), tf.TensorSpec([None,10], tf.float32))

# Get frozen ConcreteFunction for gradients
frozen_grad_func = convert_variables_to_constants_v2(grad_func)
layers_grad = [op.name for op in frozen_grad_func.graph.get_operations()]

# Calculate FLOPs for the backward pass
run_meta_grad = tf.compat.v1.RunMetadata()
opts_grad = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

# Use the profiler to get the number of FLOPs for the backward pass
flops_backward = tf.compat.v1.profiler.profile(graph=frozen_grad_func.graph, run_meta=run_meta_grad, cmd='op', options=opts_grad)
print(f"FLOPs (Backward Pass): {flops_backward.total_float_ops}")

# Total FLOPs (Forward + Backward)
total_flops = flops_forward.total_float_ops + flops_backward.total_float_ops
print(f"Total FLOPs (Forward + Backward): {total_flops}")

cutmodel.compile(loss=loss_metric,
      optimizer=opt,
      metrics=['accuracy'])


q_history = cutmodel.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch, validation_data=(X_test, y_test), verbose=2)

import pandas as pd

hist_df = pd.DataFrame(q_history.history)
 # or save to csv:
save_name=input("Please enter filename to save results, must end in .csv: ")
hist_csv_file = save_name
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
print(f"File saved as {save_name}.csv")

cut.graph(q_history)



