# IPDALight
Reinforcement Learning (RL) has been recognized as one of the most effective methods to optimize traffic signal control. However, due to the inappropriate design of RL elements (i.e., reward and state) for complex traffic dynamics, existing RL-based approaches suffer from slow convergence to optimal traffic signal plans. Meanwhile, to simplify the traffic  modeling, most optimization methods assume that the phase duration of traffic signals is constant, which strongly limits the RL  capability to search for traffic signal control policies with shorter average vehicle travel time and better GreenWave  control. To address these issues, this  project proposes a novel intensity- and phase duration-aware RL  method named IPDALight for the optimization of traffic signal control. Inspired by the Max Pressure (MP)-based traffic control strategy used in the transportation field, we introduce a new concept named intensity, which ensures that our reward design and state representation can accurately reflect the status of vehicles. By taking the coordination of neighboring intersections into account, our approach enables the fine-tuning of phase duration of traffic signals to adapt to dynamic traffic situations. Comprehensive experimental results on both synthetic and real-world traffic scenarios show that, compared with the state-of-the-art RL methods, IPDALight can not only achieve better average vehicle travel time and greenwave control for various multi-intersection scenarios, but also converge to optimal solutions much faster.

## Requirements
- Python 3.5+
- cityflow
- tqdm 
- tensorflow 1.11.0
- keras
- pandas
- numpy


## Usage
### 1. Simulator installation
First, install CityFlow simulator. Detailed introduction guide files can be found in https://cityflow-project.github.io/

#### 1. Install cpp dependencies
``sudo apt update && sudo apt install -y build-essential cmake``

#### 2. Clone CityFlow project from github
``git clone https://github.com/cityflow-project/CityFlow.git``

#### 3. Go to CityFlow project’s root directory and run
``pip install .``

#### 4. Wait for installation to complete and CityFlow should be successfully installed
``import cityflow``

``eng = cityflow.Engine``

### 2.Dataset
For the experiments, we used both synthetic and realworld traffic datasets provided by https://traffic-signal-control.github.io/dataset.html.

* synthetic data

  We considered four synthetic traffic datasets with different scales (i.e., 1x3, 2x2, 3x3, 4x4). Traffic files can be found in ``data/template_lsr/new/1_3`` and ``data/template_lsr/new/2_2`` and ``data/template_lsr/new/3_3`` and ``data/template_lsr/new/4_4``.

* real-world data

We used two datasets collected from the real-world traffic of two cities (i.e., Hangzhou and Jinan) in China via roadside surveillance cameras. Traffic files of Hangzhou can be found in ``data/hangzhou``, which contains 16 intersections in the form of a 4x4 grid. Traffic files of Jinan can be found in ``data/jinan``, which contains 12 intersections in the form of a 3x4 grid.

### 3. Run the code

#### 1. Config
Dataset can be chosed in dict cityflow_config in ``train.py``. Parameters (e.g., the number of episode, num_step for each episode) for simulation can be modified in dict config in ``train.py``.
In our experiments, we set the simulation episode 200, the timespan of each episode 3600 seconds, the weight factor of vehicle speed to 1, the shrinkage factor of neighboring intersection impacts to 0.1, the batch size to 32, the discount factor to 0.95, the learning rate to 1 and the probability of selecting a random action to 0.05. 

#### 2. Run the main file
``python train.py``

## Comparison of average travel time
### 1. Methods
|Method|paper link|source code link|
|--|--|--|
|**Fixed Time**|***Traffic engineering***<br>A policy that selects control phases in a cyclical way with a predefined duration and phase sequence.<br>https://trid.trb.org/view/310674|-|
|**SOTL**|***Self organizing traffic lights: A realistic simulation***<br>A method that adaptively controls traffic lights based on a threshold indicating the number of waiting vehicles<br>https://arxiv.org/abs/nlin/0610040v1|https://github.com/tianrang-intelligence/TSCC2019/blob/master/sotl_agent.py|
|**GRL**|***Coordinated deep reinforcement learners for traffic light control***<br>An RL-based method based on Q-learning for coordinated traffic signal control, which can learn the joint Q-function of two adjacent intersections by using a coordination graph.<br>https://www.elisevanderpol.nl/papers/vanderpolNIPSMALIC2016.pdf|https://traffic-signal-control.github.io/code.html|
**CoLight**|***CoLight: Learning Network-level Cooperation for Traffic Signal Control***<br>A deep RL-based method that considers the neighboring intersection information, which uses graph attentional networks to facilitate the communication among intersections.<br>https://dl.acm.org/doi/abs/10.1145/3357384.3357902|https://github.com/wingsweihua/colight|
|**PressLight**|***PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network***<br>A deep RL-based method that can effectively select control phases for intersection pressure minimization based on the MP theory.<br>https://faculty.ist.psu.edu/jessieli/Publications/2019-KDD-presslight.pdf|https://github.com/wingsweihua/presslight|

### 2. Comparison results of average vehicle travel time
![image](https://user-images.githubusercontent.com/29703034/130348368-d8efffc0-25f7-4c78-9ae5-32500fe7f8c2.png)

## Other experiments
### 1. Comparison of fairness
[FairnessComparison](https://github.com/Dokyyy/IPDALight/blob/main/FairnessComparison.md)

### 2. Comparison of intersection pressure
[PressureComparison](https://github.com/Dokyyy/IPDALight/blob/main/PressureComparison.md)

### 3. Comparison of greenwave control effects
[GreenwaveComparison](https://github.com/Dokyyy/IPDALight/blob/main/GreenwaveComparison.md)
