# Comparison of average travel time and convergence rate
Following existing studies, we use the average travel time to evaluate the performance. The average travel time of all vehicles is the most frequently used measure in the transportation field, which is calculated as the average travel time of all vehicles spent in the system.


## Experiments

### 1. Config
Dataset can be chosed in the dictionary *cityflow_config* in ``train.py``. Parameters (e.g., the number of episode, num_step for each episode) for simulation can be modified in the dictionary *config* in ``train.py``.

In our experiments, we set the simulation episode to 200, the timespan of each episode to 3600 seconds, the weight factor of vehicle speed to 1, the shrinkage factor of neighboring intersection impacts to 0.1, the batch size to 32, the discount factor to 0.95, the learning rate to 0.001 and the probability of selecting a random action to 0.05. 

### 2. Run the code

``python train.py``

### Results
After the code is executed, we can get a result file named *Travel Time-dataset-IPDALight.csv*, which record the average vehicle travel time of each episode. Then we can plot the average travel time convergence curve.

The table below compares the performance of different traffic signal control methods in terms of average vehicle travel time.

![image](https://user-images.githubusercontent.com/29703034/130348368-d8efffc0-25f7-4c78-9ae5-32500fe7f8c2.png)

The figure below shows the comparison results between IPDALight and PressLight in terms of convergence rate.

![image](https://user-images.githubusercontent.com/29703034/130360973-52064dcf-9fa2-46bb-89b3-9c0bf447092e.png)