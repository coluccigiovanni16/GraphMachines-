# Graph Machines

Implementation of Graph Machines using python, and their application to a dataset composed of molecule. 

## Table of Contents

- [Project Organization](#project-organization)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Built With](#built-with)
- [Thanks to](#thanks-to)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)



## Project Organization

    ├── LICENSE
    ├── README.md          	<- The top-level README for developers using this project.
    ├── data
    │   ├── external       	<- Data from third party sources.
    │   ├── interim        	<- Intermediate data that has been transformed.
    │   ├── processed      	<- The final, canonical data sets for modeling.
    │   └── raw            	<- The original, immutable data dump.
    │
    ├── docs               	<- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             	<- Trained and serialized models, model predictions, or model
    │			           summaries
    │
    ├── notebooks          	<- Jupyter notebooks. Naming c
    │
    ├── references         	<- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            	<- Generated analysis as HTML, PDF, LaTeX, generated graphics and 
    │                              figures to be used in reporting
    │
    ├── Pipfile,Pipfile.lock    <- File used for pipenv 
    │ 
    │
    ├── src                	<- Source code for use in this project.
    │   ├── __init__.py    	<- Makes src a Python module
    │   │
    │   ├── data           	<- Scripts to download or generate data
    │   │   ├── load_dataset.py
    │   │   └── make_dataset.py
    │   │
    │   ├── scott       	<- Scripts to turn raw data into newick format
    │   │  
    │   │
    │   ├── models         	<- Scripts to train_regression models and then use trained models to
    │   │   ├── predict_model.py   make predictions
    │   │   └── train_model.py
    │   │
    │   ├── visualization  	<- Scripts to create exploratory and results oriented visualizations
    │   │      └── visualize.py
    │   │
    │   └── Net  	        <- Neural Network 
    │       └── FNN_GM_Net.py
    │
    ├──GM-Classification.py     <- Script for classification task
    │
    └──GM-Regression.py         <- Script for regression task




## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development 
and testing purposes. 

### Dataset
The datasets used can be downloaded from :
https://brunl01.users.greyc.fr/CHEMISTRY/

### Prerequisites

First of all you need to get this repo on your pc:
```
git clone https://github.com/elbarto91/GraphMachines-.git
```
The repository downloaded is ready-to-go, so you may only extract it.
To execute the script(train/predict) in a separate environment,
must be installed python 3.7(.5) and pipenv.

### Installing

A step by step series of examples that tell you how to get a development env running

Get python3:
```
sudo apt-get update
sudo apt-get install python3.7
```
Get pipenv:
```
sudo pip/pip3 install pipenv
```
From the main folder of the project,in order to install the environment:
```
sudo pipenv install
```


## Usage

```
usage: GM-Regression.py [-h] [-d DEVICE] [-e NUM_EPOCHS]
                        [-hln HIDDEN_LAYER_SIZE] [-lr LEARNING_RATE]
                        [-r REPORT] [-rdd ROOTDIRDATASET] [-trf TRAINFILE]
                        [-tef TESTFILE] [-s SAVE] [-l LOAD] [-b BIAS]
                        [-rn REPORTNAME] [-mp MODELPATH]

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        device to use(GPU or CPU(defualt))
  -e NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        number of epochs,default=10000
  -hln HIDDEN_LAYER_SIZE, --hidden_layer_size HIDDEN_LAYER_SIZE
                        number of nodes for the hidden layer, default = 4
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate for the optimizer, default = 0.001
  -r REPORT, --report REPORT
                        save result in a report file
  -rdd ROOTDIRDATASET, --rootDirDataset ROOTDIRDATASET
                        directory of dataset files
  -trf TRAINFILE, --trainFile TRAINFILE
                        dataset containing the name on the trainset files
  -tef TESTFILE, --testFile TESTFILE
                        dataset containing the name on the testset files
  -s SAVE, --save SAVE  True if you want to save the model, default = False
  -l LOAD, --load LOAD  True if you want to load the model, default = False
  -b BIAS, --bias BIAS  bias value, default = 1
  -rn REPORTNAME, --reportName REPORTNAME
                        base name for the report's folder
  -mp MODELPATH, --modelPath MODELPATH
                        model's path

```
### TRAIN NEURAL NETWORK 
```
python GM-Regression.py -e 1000 -rdd /home/elbarto91/provapipEnv/graphmachines/data/processed/Acyclic/ -trf trainset_0.ds -tef testset_0.ds --reportName ACYCLIC --save True

```

### PREDICT USING A TRAINED NEURAL NETWORK 
(Check that yu have a saved model)
```
python GM-Regression.py -rdd data/processed/Acyclic/  --report True -tef testset_0.ds --reportName ACYCLIC --load True --modelPath models/ACYCLIC/model_testset_0.ds-Dvalue12-maxMValue4-Saved.pth
```

## Built With
* [Pycharm](https://www.jetbrains.com/pycharm/) - Integrated development environment (IDE)
* [Git](https://git-scm.com/) -  distributed version-control system for tracking changes in source code during software development. 
* [Python 3.7.5](https://www.python.org/) - Interpreted, high-level, general-purpose programming language
* [Pytorch](https://pytorch.org/) - An open source machine learning framework
* [Jupyter Notebook](https://jupyter.org/) - Open-source web application 
* [Pipenv](https://github.com/pypa/pipenv) - Packaging tool for Python 
####Based on:
* [Scott](https://github.com/theplatypus/scott) -  software able to compute, for any fully-labelled (edge and node) graph,
 a canonical tree representative of its isomorphism class, that can be derived into a canonical trace (string) or adjacency matrix 
* [Graph Machines and Their Applications to Computer-Aided Drug Design: A New Approach to Learning from Structured Data](https://www.researchgate.net/publication/221302023_Graph_Machines_and_Their_Applications_to_Computer-Aided_Drug_Design_A_New_Approach_to_Learning_from_Structured_Data) - Graph machines learn real numbers from graphs. 



## Thanks to
<a href="https://www.ensicaen.fr">
    <img alt="Ensicaen" title="Ensicaen" src="https://www.ensicaen.fr/wp-content/uploads/2017/03/LogoENSICAEN_institutionnel_couleurs_72dpi-e1576235617909.jpg" width="250">
  </a>
  <a href="https://www.greyc.fr/">
    <img alt="Greyc" title="Greyc" src="https://captil.greyc.fr/userfiles/images/logos/GREYC.jpg" width="250">
  </a>
  <a href="https://www.unisa.it/">
    <img alt="Unisa" title="Unisa" src="https://www.unisa.it/rescue/img/logo_standard.png" width="250">
  </a>
  <a href="https://www.diem.unisa.it/">
    <img alt="Diem" title="DIEM" src="https://elearning.diem.unisa.it/alr/logo.jpg" width="250">
  </a>
  

## Authors

* **Giovanni Colucci** - *Total work* - [Elbarto91](https://github.com/elbarto91)
* **Luc Brun** - *Supervisor* - [Luc Brun](https://brunl01.users.greyc.fr/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments

* If you want to use a different dataset, be sure to use the same layout of the dataset in processed.



