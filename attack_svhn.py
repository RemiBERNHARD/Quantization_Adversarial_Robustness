import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[2]

from keras.models import Model, load_model
from keras.utils import np_utils
from keras import backend 
import tensorflow as tf
import numpy as np
import random
random.seed(123)

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import ProjectedGradientDescent #BIM
from cleverhans.attacks import SPSA
from ZOO_attack import BlackBoxL2
from utils_func import metrics
from utils_quantized import load_quantized_model

sess = tf.Session()
backend.set_session(sess)
backend._LEARNING_PHASE = tf.constant(0)
backend.set_learning_phase(0)

####Load data set
X_train = np.load("SVHN_data/X_train.npy")
X_test = np.load("SVHN_data/X_test.npy")
y_train = np.load("SVHN_data/y_train.npy")
y_test = np.load("SVHN_data/y_test.npy")

y_train[y_train==10]=0
y_test[y_test==10]=0
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


####Load model
model_type = sys.argv[1]
print("Attacking model: " + model_type)
model = load_quantized_model("SVHN", model_type)
print("Accuracy on test set: " + str(model.evaluate(X_test, Y_test, verbose=0)[1]))

pred_clean = np.argmax(model.predict(X_test), axis = 1)
well_pred = np.arange(0, len(X_test))[pred_clean == y_test]
indices = np.random.choice(well_pred, 1000, replace=False)
indices_spsa = indices[:int(np.floor(len(indices))/10)]
batch_attack = 100


####Perform attacks
wrap = KerasModelWrapper(model)

#####FGSM
print("FGSM")
fgsm_params = {'eps': 0.03,
               'clip_min': 0.,
               'clip_max': 1.}
fgsm = FastGradientMethod(wrap, sess=sess)
X_adv = np.zeros((len(indices),32,32,3))
for i in range(0,len(indices),batch_attack):
    X_adv[i:i+batch_attack] = fgsm.generate_np(X_test[indices[i:(i+batch_attack)]], **fgsm_params)
print("results")
results = metrics(model, X_adv, X_test, y_test, indices)
print(results)    

#####BIM
print("BIM")
bim_params = {'eps': 0.03,
              'nb_iter': 300,
              'eps_iter': 0.03/100,
              'ord': np.inf,
               'clip_min': 0.,
               'clip_max': 1.,
               'rand_init': False
               }
bim = ProjectedGradientDescent(wrap, sess=sess)
X_adv = np.zeros((len(indices),32,32,3))
for i in range(0,len(indices),batch_attack):
    X_adv[i:i+batch_attack] = bim.generate_np(X_test[indices[i:(i+batch_attack)]], **bim_params)
print("results")
results = metrics(model, X_adv, X_test, y_test, indices)
print(results)    

#####CWL2
print("CWL2")
cwl2_params = {'binary_search_steps': 10,
               'max_iterations': 100,
               'learning_rate': 0.1,
               'batch_size': 100,
               'initial_const': 0.5,
               'clip_min': 0.,
               'clip_max': 1.
               }
cwl2 = CarliniWagnerL2(wrap, sess=sess)
X_adv = np.zeros((len(indices),32,32,3))
for i in range(0,len(indices),batch_attack):
    X_adv[i:i+batch_attack] = cwl2.generate_np(X_test[indices[i:(i+batch_attack)]], **cwl2_params)
print("results")
results = metrics(model, X_adv, X_test, y_test, indices)
print(results) 

#####ZOO
binary_search_steps = 10
max_iterations = 100
learning_rate = 0.1
targeted = False
confidence = 0
initial_const = 0.5
use_log = True

zoo = BlackBoxL2(sess, model, batch_size=128, max_iterations=max_iterations, learning_rate=learning_rate, targeted=False, confidence=0, 
        binary_search_steps=binary_search_steps, initial_const = initial_const, use_log = True,
        image_size=32, num_channels=3, num_labels=10)

X_adv = np.zeros((len(indices),32,32,3))
for i in range(len(indices)):
    ind = indices[i]
    X_adv[i] = zoo.attack(X_test[ind:(ind+1)], Y_test[ind:(ind+1)])    
print("results")
results = metrics(model, X_adv, X_test, y_test, indices)
print(results) 

#####SPSA
spsa_params = {'eps': 0.03,
               'learning_rate': 0.01,
               'delta': 0.01,
               'spsa_samples': 128,
               'spsa_iters': 1,
               'nb_iter': 100,
               'clip_min': 0.,
               'clip_max': 1.
               }
spsa = SPSA(wrap, sess=sess)
X_adv = np.zeros((len(indices_spsa),32,32,3))
for i in range(0,len(indices_spsa),batch_attack):
    X_adv[i:i+batch_attack] = spsa.generate_np(X_test[indices[i:(i+batch_attack)]], **spsa_params)
print("results")
results = metrics(model, X_adv, X_test, y_test, indices)
print(results) 
