# IPDALight
IPDALight for traffic signal control

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


### 2. Run the code
The main file is train.py

#### 1. Config
Dataset can be chosed in dict cityflow_config in ``train.py``. Parameters (e.g., the number of episode, num_step for each episode) for simulation can be modified in dict config in ``train.py``.

#### 2. Run the code
``python train.py``



## Dataset

* synthetic data

  Traffic files can be found in ``data/template_lsr/new/1_3`` && ``data/template_lsr/new/2_2`` && ``data/template_lsr/new/3_3`` && ``data/template_lsr/new/4_4``.

* real-world data

  Traffic files of Hangzhou City can be found in ``data/hangzhou``, which contains 16 intersections in the form of a 4x4 grid. Traffic files of Hangzhou City can be found in ``data/jinan``, which contains 12 intersections in the form of a 3x4 grid.
  
  
