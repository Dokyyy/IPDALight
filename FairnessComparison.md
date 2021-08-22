# Fair-IPDALight
According to the reward mechanisms of MP-based methods (e.g., PressLight) and intensity-based IPDALight, the well-trained RL agent of an intersection will always choose the phase with the maximum intensity for traffic control. Consider an extreme scenario where there is only one vehicle *A* waiting on a certain phase of a traffic intersection, but a steady stream of vehicles pass by on other phases. In this case, the chance for *A* to pass
the intersection is extremely low, resulting in the serious unfairness problem.

By considering the vehicle waiting time when we design the intensity (i.e., vehicles with longer waiting time will have a higher intensity value), the fairness of the mentioned scenario can be guaranteed.

As shown in the formula (for caculating the intensity of vehicles) below, we use tw to represent the waiting time of the vehicle, and σ to denote the weight of the waiting time.

![image](https://user-images.githubusercontent.com/29703034/130350586-2b6b6963-4a48-4796-86b6-1eecea92fb3a.png)


## Experiments
Codes of Fair-IPDALight are in the folder *Fair-IPDALight*, and the dataset files can be found in the folder *data*. Running the main file *train.py*.

### 1. Add the dataset
Copy the folder *data* into the folder *Fair-IPDALight*.

### 2. Config
Dataset can be chosed in the dictionary *cityflow_config* in ``train.py``. Parameters (e.g., the number of episode, num_step for each episode) for simulation can be modified in the dictionary *config* in ``train.py``.

In our experiments, we set the weight of the vehicle waiting time to 0.01.

### 3. Run the code

``cd Fair-IPDALight``

``python train.py``

### Results
After the code is executed, we can get a result file named *Max Waiting Time-dataset-FAIR-IPDALight.csv*, which records the maximum vehicle waiting time of each episode. Then we take the average of the maximum waiting time after the algorithm converges as the result in the table below.

![image](https://user-images.githubusercontent.com/29703034/130352309-350e100e-10a3-436e-8083-7f9c8176e10e.png)