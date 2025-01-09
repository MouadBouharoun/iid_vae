# Detecting Model Inconsistency Attacks Against Federated Learning Systems
![Version](https://img.shields.io/badge/version-1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Description**:
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
