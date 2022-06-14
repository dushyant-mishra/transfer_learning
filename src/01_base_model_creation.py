import argparse
import os
import numpy as np
#from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import tensorflow as tf


STAGE = "creating base model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path): #, params_path):
    ## read config files
    config = read_yaml(config_path)
    #params = read_yaml(params_path)
    #get the data
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    validation_datasize = 5000
    # create a validation dataset from the full training data
    # scale the data between 0 and 1 by dividing it by 255. as its an unsigned data beyween 0-255 range 
    X_valid, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    # scale the test set as well
    X_test = X_test / 255.

    #set the seeds
    seed = 2021 ##get it from config file
    tf.random.set_seed(seed)
    np.random.seed(seed)

    #define layers
    layers = [tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
              tf.keras.layers.Dense(300, name="hiddenLayer1"),
              tf.keras.layers.LeakyReLU(),
              tf.keras.layers.Dense(100, name="hiddenLayer2"),
              tf.keras.layers.LeakyReLU(),
              tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")
              ]

    #define model and compile
    model = tf.keras.models.Sequential(layers)

    loss = "sparse_categorical_crossentropy"
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    metrics = ["accuracy"]      
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.summary()

    #train the model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_valid, y_valid),
        verbose=2)

    #save the model
    model_dir_path = os.path.join("artifacts", "models")
    create_directories(model_dir_path)
    model_file_path = os.path.join(model_dir_path, "model.h5")
    model.save(model_file_path)

    logging.info(f"base model is saved to {model_file_path}")
    logging.info(f"evaluation on test set: {model.evaluate(X_test, y_test, verbose=0)}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    #args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config) #, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e