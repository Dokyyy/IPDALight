# Comparison of intersection pressure
Max Pressure has been widely used in the transportation field, and the concept of pressure (i.e., the difference between the number of vehicles entering the intersection and the number of vehicles leaving the intersection) is an objective and fair metric for the evaluation of traffic control effects.


## Experiments
### Run the code
Codes for cmparison of intersection pressure are in the folder Pressure-IPDALight, and the dataset files can be found in the folder *data*. Running the main file *train.py*.

``python train.py``

### Results
After the code is executed, we can get result files named *Pressure-dataset-intersection-IPDALight.txt*, which record the pressure changes for different intersections. Then we can plot the pressure changes after the algorithm converges. For example, the figure below compares the pressure of IPDALight and PressLight with a duration of 20 seconds.

![image](https://user-images.githubusercontent.com/29703034/130354274-0372a6f1-de71-47b2-9c82-76baaf4d7021.png)