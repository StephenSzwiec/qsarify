# qsarify

qsarify is a library of tools for the analysis of QSAR/QSPR datasets and models. This library is intended to be used to produce models which relate a set of calculated chemical descriptors to a given numeric endpoint. Many great tools will take the geometry or string data of a given chemical and compute **descriptors**, which are numeric measures of the properties of these, but you can generate some of these with another one of my scripts, [Free Descriptors](https://github.com/StephenSzwiec/free_descriptors).

# Dependencies

- Python 3
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org)
- [matplotlib](https://matplotlib.org)


# Installation

`pip install qsarify`

# What is included right now?

- Data preprocessing tools: `data_tools`
- Dimensionality reduction via clustering: `clustering`
- Feature selection:
	- Single threaded: `feature_selection_single`
	- Multi-threaded: `feature_selection_multi`
- Model Export and Visualization: `model_export`
- Cross Valiidation: `cross_validation`

# How to use

The best way to learn how to use this library is to look at the example notebook in the `examples` folder. This notebook will walk you through the workflow of using this library to build a QSAR model.

# Future Plans

- Massively parallel feature selection methods:
	- CUDA acceleration
	- MPI acceleration
- Include Shannon Entropy as a dimensionality reduction metric in clustering
- Embedded kernel methods
- More visualization tools
- More cross validation tools
- Feature selection tools for categorical data

# Contributing

If you would like to contribute to this project, please feel free to fork this repository and submit a pull request. Otherwise, you may also submit an issue. I will try to respond to issues as quickly as possible.

# License

This project is licensed under the GNU GPLv3 license. See the LICENSE file for more details.

# Acknowledgements

- Authors acknowledge the support from the National Science Foundation for the grant award NSF/IUCRC/CB2-2113804 "Center for Bioplastic and Biocomposites (CB2)", and from the United States Navy’s Office of Naval Research under grant number N00014-22-1-2129.
- This work was also supported in part by the National Science Foundation NSF-MRI award OAC-2019077.
- The authors are grateful for financial and administrative support provided by the Department of Coatings and Polymer Materials (NDSU).
- Finally, This work used resources of the Center for Computationally Assisted Science and Technology (CCAST) at North Dakota State University, which were made possible in part by NSF MRI Award No. 2019077. 

# Citation

If you use this library in your work, please cite it as follows:

Szwiec, Stephen. (2023). qsarify: A high performance library for QSAR model development.

BibTex:
```
@misc{szwiec2023qsarify,
  authors = {Szwiec, Stephen and Rasulev, Bakhtiyor},
  title = {qsarify: A high performance library for QSAR model development},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/stephenszwiec/qsarify}},
  }
```
