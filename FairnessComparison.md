# Fair-IPDALight
According to the reward mechanisms of MP-based methods (e.g., PressLight) and intensity-based IPDALight, the well-trained RL agent of an intersection will always choose the phase with the maximum intensity for traffic control. Consider an extreme scenario where there is only one vehicle *A* waiting on a certain phase of a traffic intersection, but a steady stream of vehicles pass by on other phases. In this case, the chance for *A* to pass
the intersection is extremely low, resulting in the serious unfairness problem.

By considering the vehicle waiting time when we design the intensity (i.e., vehicles with longer waiting time will have a higher intensity value), the fairness of the mentioned scenario can be guaranteed.

![image](https://user-images.githubusercontent.com/29703034/130350586-2b6b6963-4a48-4796-86b6-1eecea92fb3a.png)

## Experiments
### Run the code
Codes of Fair-IPDALight are in the folder *Fair-IPDALight*, and the dataset files can be found in the folder *data*. Running the main file *train.py*.

``python train.py`` 

### Results
After the code is executed, we can get a result file named *Max Waiting Time-dataset-FAIR-IPDALight.csv*, which records the maximum vehicle waiting time of each episode. Then we take the average of the maximum waiting time after the algorithm converges as the result in the table below.

![image](https://user-images.githubusercontent.com/29703034/130352309-350e100e-10a3-436e-8083-7f9c8176e10e.png)