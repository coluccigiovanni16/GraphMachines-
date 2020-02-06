GraphMachines
==============================

Implementation of graph machines

Project Organization
------------

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



@inproceedings{bloyet2019scott,
  title={Scott: A method for representing graphs as rooted trees for graph canonization},
  author={Bloyet, Nicolas and Marteau, Pierre-Fran{\c{c}}ois and Frenod, Emmanuel},
  booktitle={International Conference on Complex Networks and Their Applications},
  pages={578--590},
  year={2019},
  organization={Springer}
}