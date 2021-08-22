# Comparison of greenwave control effects
A greenwave refers to a scenario that occurs when a series of traffic lights coordinate with each other to enable continuous traffic flow along one given direction. Detailed information about greenwave can be found at https://trid.trb.org/view/310674. Our proposed method IPDALight can form some greenwave in practice.

## Experiments
Codes for comparison of greenwave control effects are in the folder *Greenwave-IPDALight*, and the dataset files can be found in the folder *data*. Running the main file *train.py*.

### 1. Add the dataset
Copy the folder *data* into the folder *Greenwave-IPDALight*.

### 2. Config
Dataset can be chosed in the dictionary *cityflow_config* in ``train.py``. Parameters (e.g., the number of episode, num_step for each episode) for simulation can be modified in the dictionary *config* in ``train.py``.

### 3. Run the code

``cd Greenwave-IPDALight``

``python train.py``

### Results
After the code is executed, we can get a result file named *Greenwave-dataset-IPDALight.txt*, which records the phase selection and phase duration for each intersection. Then we can plot space-time diagrams to demonstrate the phase plan of the traffic signal controllers. As shown in the figure below, we can observe the greenwaves.

![image](https://user-images.githubusercontent.com/29703034/130354286-d663a779-2d43-4fe5-8d0e-23ff40c4896d.png)