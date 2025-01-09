# Detecting Model Inconsistency Attacks Against Federated Learning Systems
![Version](https://img.shields.io/badge/version-1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Description**:
Federated learning enables collaborative training of a shared neural network by exchanging model parameters among multiple workers without outsourcing local data. 

The distributed nature of federated learning gives rise to different threats, such as inference attacks, poisoning attacks, and identity theft.  

We focus on inference attacks where a malicious subset of workers attempts to infer information about the victim's training data.

Existing mitigation techniques against inference use secure aggregation to hide local updates provided by the workers. However, an adaptive adversary still has the ability to circumvent secure aggregation methods and learn patterns in private data using model inconsistency. %This tries to retrieve the target's local model from the securely computed aggregation function, which in turn will be used to perform the inference attack. The adversary sends inconsistent models, one for the target victim and another for non-target workers.

This work addresses model inconsistency attacks, in which the parameter server - a central server that aggregates model updates from workers - can maliciously overfit a part of the model of the target worker. The server then injects detectors that trigger an activation if a given sample is present in the training data of the target victim. The server sends a crafted model to non-target workers that suppresses the entries of their local updates at the position of the detectors. Therefore, the server infers the sample membership only from the aggregated model, even when secure aggregation is enabled. We evaluate a client-side defense strategy based on variational auto-encoders. The suggested approach, combined with secure aggregation, performs a conditional secure aggregation scheme to create the new local update rule for workers. We illustrate the performance of the suggested approach, which provides high detection accuracy and a low false negative rate.

![image](https://github.com/user-attachments/assets/c3977365-a1eb-41f4-b510-e2c2b8d69786)



**Requirements**:

* python3 / jupyter
  * TensorFlow2
  * keras 
  * numpy
  * matplotlib
  * tqdm
  * pandas
  * scikit-learn
 
  
**Usage**:
```
python3 main.py -s <settings file> -n <number of clients> -r <number of rounds>
```
