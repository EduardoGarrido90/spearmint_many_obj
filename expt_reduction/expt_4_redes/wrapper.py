from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.utils.np_utils import to_categorical
import numpy as np
import time
import os
import keras
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import  os, sys

def main(job_id, params):
    np.random.seed(job_id)
    params[ 'num_hidden_units' ] = np.int(np.round(params[ 'num_hidden_units' ]))
    params[ 'num_hidden_layers' ] = np.int(np.round(params['num_hidden_layers']))

    #LOAD DIGITS DATASET. MAKE A TRAIN TEST SPLIT OF 20% OF THE DATA TO PREDICT IT USING THE FAKE DEEP NEURAL NET.
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, shuffle=False)
    
    #Create a one-hot encoding vector of the classes.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    #Input layer.
    model = Sequential()
    model.add(Dense(params['num_hidden_units'], input_dim=64, activation='relu', name = 'capa_0'))#, kernel_regularizer=L1L2(l1=np.exp(params['log_l1_reg']),  l2=np.exp(params['log_l2_reg']))))
    model.add(Dropout(0.2))

    #Hidden layers.        
    for i in range(params['num_hidden_layers'] - 1):
        model.add(Dense(params['num_hidden_units'], activation='relu', name = 'capa_'+ str(i+1)))#, kernel_regularizer=L1L2(l1=np.exp(params['log_l1_reg']), \ l2=(np.exp(params['log_l2_reg'])))))
        model.add(Dropout(0.2))

    # Output layer.
    model.add(Dense(10, activation='softmax', name = 'capa_ultima'))

    # Compiling and finishing configuring the Fake deep net.
    adam = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #No entrenamos ya que no tenemos un ordenador lo suficientemente potente para ello (Restriccion TFG).

    #Evaluamos la red (idealmente entrenada) en el split de test que habras hecho de Digits.
    start_time = time.time()
    scores = model.evaluate(X_test, y_test)
    end_time = time.time()
    time_prediction = end_time - start_time
    prediction_error_keras = 1.0 - scores[1]
    print("\nKeras error:  %f\n" % (prediction_error_keras*100.0))
    print("\nKeras time:  %f\n" % time_prediction)

    file_name = "red_" + "{:02d}".format(params['num_hidden_layers']) + '_' + "{:02d}".format(params['num_hidden_units']) + ".h5"
    model.save(file_name)
    model_size = os.path.getsize(file_name)
    print("\nKeras model size:  %f\n" % model_size)
    
    return {'o1_error' : prediction_error_keras, 'o2_time' : time_prediction, 'o3_size': model_size}

#if __name__ == "__main__":
#    main(0, {'num_hidden_layers' : int(sys.argv[1]), 'num_hidden_units' : int(sys.argv[2])})
