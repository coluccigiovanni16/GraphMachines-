GraphMachines
==============================

Implementation of graph machines

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





<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Feedback](#feedback)
- [Contributors](#contributors)
- [Build Process](#build-process)
- [Backers](#backers-)
- [Sponsors](#sponsors-)
- [Acknowledgments](#acknowledgments)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


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