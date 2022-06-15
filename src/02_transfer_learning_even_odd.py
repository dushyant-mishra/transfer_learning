import argparse
import os
import numpy as np
#from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import tensorflow as tf
import io


STAGE = "transfer learning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_even_odd_labels(list_of_labels):
    """
    Update the labels to be even or odd.
    """
    for idx, label in enumerate(list_of_labels):
        even_condition = label % 2 == 0
        list_of_labels[idx] = np.where(even_condition, 1, 0)
    return list_of_labels

    # new_list = []
    # for label in list_of_labels:
    #     if label % 2 == 0:
    #         new_list.append(0)
    #     else:
    #         new_list.append(1)
    # return new_list

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

    y_train_binary, y_test_binary, y_valid_binary = update_even_odd_labels([y_train, y_test, y_valid])

    #set the seeds
    seed = 2021 ##get it from config file
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## log our model summary information
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    #loading the base model
    base_model_path = os.path.join("artifacts", "models", "base_model.h5")
    base_model = tf.keras.models.load_model(base_model_path)
    # model summary information
    logging.info(f"loaded base model summary: \n{_log_model_summary(base_model)}")


    #freeze the weights of base model
    for layer in base_model.layers[:-1]:
        print(f"trainable status before: {layer.name} - {layer.trainable}")
        layer.trainable = False
        print(f"trainable status after: {layer.name} - {layer.trainable}")  

    base_layer = base_model.layers[:-1]

    #define model and compile
    new_model = tf.keras.models.Sequential(base_layer)
    new_model.add( tf.keras.layers.Dense(2, activation="softmax", name = "output_layer")
    )

    logging.info(f"{STAGE} model summary: \n{_log_model_summary(new_model)}")
    
    loss = "sparse_categorical_crossentropy"
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    metrics = ["accuracy"]      
    new_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    
    #train the model
    history = new_model.fit(
        X_train, y_train_binary,   # << y_train_binary is the updated y_train
        epochs=10,
        validation_data=(X_valid, y_valid_binary),  # << y_valid_binary is the updated y_valid
        verbose=2)

    #save the model
    model_dir_path = os.path.join("artifacts", "models")
    #create_directories([model_dir_path])
    model_file_path = os.path.join(model_dir_path, "even_odd_model.h5")
    new_model.save(model_file_path)

    logging.info(f"base model is saved to {model_file_path}")
    logging.info(f"evaluation on test set: {new_model.evaluate(X_test, y_test_binary, verbose=0)}")  # << y_test_binary is the updated y_test

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