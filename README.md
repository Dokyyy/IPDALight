# IPDALight
IPDALight for traffic signal control

## Usage
### 1. Simulator installation
First, install CityFlow simulator. Guide files can be found in https://cityflow-project.github.io/

### 2. Run the code
``python train.py``

### 3. Config
Dataset can be chosed in dict cityflow_config in ``train.py``. the parameters (e.g., the number of episode, num_step for each episode) for simulation can be modified in dict config in ``train.py``.

## Dataset

* synthetic data

  Traffic files can be found in ``data/template_lsr/new/1_3`` && ``data/template_lsr/new/2_2`` && ``data/template_lsr/new/3_3`` && ``data/template_lsr/new/4_4``.

* real-world data

  Traffic files of Hangzhou City can be found in ``data/hangzhou``, which contains 16 intersections in the form of a 4x4 grid. Traffic files of Hangzhou City can be found in ``data/jinan``, which contains 12 intersections in the form of a 3x4 grid.
  
  
