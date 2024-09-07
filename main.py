import tqdm
import random
import os
import argparse
import importlib.util
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vae_utility import train_vae, detect_malicious_modifications, vae_threshold, detect_malicious_modifications_2
from utility import *
'''
This function loads a configuration file from the settings directory defined by the "-s" or "--settings" attribute 
and returns the parameters specified in that file.

This function is used to select the desired dataset as the shadow dataset and main dataset.
'''
def load_settings(settings_file):
    full_path = os.path.join("settings", settings_file)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"The settings file '{full_path}' does not exist.") 
    spec = importlib.util.spec_from_file_location("settings", full_path)
    settings_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings_module)
    return settings_module.settings

def load_datasets(settings_file):
    settings = load_settings(settings_file)
    shadow_dataset_path = settings.get("shadow_dataset")
    main_dataset_path = settings.get("main_dataset")
    if not shadow_dataset_path or not main_dataset_path:
        raise ValueError("Both 'shadow_dataset' and 'main_dataset' must be specified in the settings.")
    shadow_dataset = pd.read_csv(shadow_dataset_path)
    main_dataset = pd.read_csv(main_dataset_path)
    return shadow_dataset, main_dataset

def initialize_model(input_shape):
    return initialiseMLP(input_shape)

def federated_learning(shadow_dataset, num_clients, num_rounds):
    X_train, _ = preprocess(shadow_dataset)
    input_shape = (X_train.shape[1],)
    global_model = initialize_model(input_shape)
    print(f"Building {num_clients} virtual client and generating {num_rounds} instances")
    client_data_chunks = data_distribution(shadow_dataset, num_clients)
    clients = [Client(client_id, data_chunk) for client_id, data_chunk in enumerate(client_data_chunks)]
    
    global_models = []
    for _ in tqdm.tqdm(range(num_rounds)):
        for client in clients:
            client.set_global_model(global_model)
        client_updates = [client_update(client.get_local_model(), client.get_data()) for client in clients]
        aggregated_gradients = server_aggregate(global_model, client_updates)
        global_model.optimizer.apply_gradients(zip(aggregated_gradients, global_model.trainable_variables))
        global_models.append(global_model)
    
    return model_to_vector(global_models), input_shape

def train_and_evaluate_vae(global_models):
    x_train, x_test = train_test_split(global_models, test_size=0.2, random_state=42)
    print("# Define VAE Hyper-parameters")
    learning_param = 0.001
    epochs = 3000
    batch_size = 32
    input_dimension = global_models[0].shape[0]
    neural_network_dimension = 512
    latent_variable_dimension = 2
    print("# Train the VAE")
    total_losses, _, _, Final_Weight, Final_Bias = train_vae(x_train, epochs, batch_size, input_dimension, neural_network_dimension, latent_variable_dimension, learning_param)
    threshold = vae_threshold(total_losses)
    return x_test, Final_Weight, Final_Bias, threshold

def detect_anomalies(x_test, Final_Weight, Final_Bias, threshold):
    print("# Test the VAE")
    print("Testing benign instances from x_test")
    reconstruction_errors = detect_malicious_modifications(x_test, Final_Weight, Final_Bias)
    malicious_indices = [i for i, error in enumerate(reconstruction_errors) if error > threshold]
    print("Potentially malicious modifications detected at indices:", malicious_indices)
    return malicious_indices

def process_inconsistent_models(main_dataset, input_shape, base_model, Final_Weight, Final_Bias, threshold):
    print("Initialiser la vérité de terrain")
    batch_A_true = ground_truth(8)
    print("Préparation de 4 instances inconsistentes")
    properties_config = [
        {"feature": "FLOW_DURATION_MILLISECONDS", "value": 4294966, "comparison": ">="},
        {"feature": "IN_BYTES", "value": 9672, "comparison": ">="},
        {"feature": "L4_DST_PORT", "value": 3000, "comparison": ">="},
        {"feature": "L4_DST_PORT", "value": 1500, "comparison": "<="}
    ]
    property_datasets = create_property_datasets(main_dataset, properties_config)
    inc_models = []
    for i, property_dataset in enumerate(property_datasets, start=1):
        print(f"Préprocessing du dataset de propriété {i}...")
        X_train, y_train = preprocess(property_dataset)
        inc_model = initialisePropMLP(input_shape)
        inc_model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1)
        inc_model = insert_weights_MI_to_GM(inc_model, base_model)
        inc_models.append(inc_model)
    
    inc_models = model_to_vector(inc_models)
    reconstruction_errors = detect_malicious_modifications(inc_models, Final_Weight, Final_Bias)
    inconsistent_inputs = [i for i, error in enumerate(reconstruction_errors) if error >= threshold]
    print("Potentially malicious modifications detected at indices:", inconsistent_inputs)
    return inc_models, inconsistent_inputs, batch_A_true

def evaluate_benign_instances(x_test, Final_Weight, Final_Bias, threshold):
    print("Préparation de 4 instances aléatoires normales")
    bening_inputs = random.sample(list(x_test), 4)
    reconstruction_errors_2 = detect_malicious_modifications(bening_inputs, Final_Weight, Final_Bias)
    inconsistent_inputs_2 = [i for i, error in enumerate(reconstruction_errors_2) if error > threshold]
    print("Potentially malicious modifications detected at indices:", inconsistent_inputs_2)
    return bening_inputs, inconsistent_inputs_2

def calculate_accuracy(inc_models, bening_inputs, batch_A_true, Final_Weight, Final_Bias, threshold):
    print("Calcul de l'accuracy : ")
    batch_A = np.concatenate((inc_models, bening_inputs))
    batch_A_pred = detect_malicious_modifications_2(batch_A, threshold, Final_Weight, Final_Bias)
    accuracy_batch_A = accuracy_score(batch_A_true, batch_A_pred)
    print("Accuracy: {:.2f}%".format(accuracy_batch_A * 100))

def main(settings_file, num_clients, num_rounds):
    shadow_dataset, main_dataset = load_datasets(settings_file)
    X_train, _ = preprocess(main_dataset)
    input_shape = (X_train.shape[1],)
    base_model = initialize_model(input_shape)
    global_models, input_shape = federated_learning(shadow_dataset, num_clients, num_rounds)
    x_test, Final_Weight, Final_Bias, threshold = train_and_evaluate_vae(global_models)
    detect_anomalies(x_test, Final_Weight, Final_Bias, threshold)
    inc_models, _, batch_A_true = process_inconsistent_models(main_dataset, input_shape, base_model, Final_Weight, Final_Bias, threshold)
    bening_inputs, _ = evaluate_benign_instances(x_test, Final_Weight, Final_Bias, threshold)
    calculate_accuracy(inc_models, bening_inputs, batch_A_true, Final_Weight, Final_Bias, threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load settings file, number of clients, number of rounds")
    parser.add_argument("-s", "--settings", default="default_settings.py", help="Name of the settings file in the settings directory (e.g., setting1.py).")
    parser.add_argument("-n", "--num_clients", type=int, default=50, help="Number of clients to use. Default is 50.")
    parser.add_argument("-r", "--num_rounds", type=int, default=100, help="Number of rounds to run. Default is 100.")
    args = parser.parse_args()
    main(args.settings, args.num_clients, args.num_rounds)
