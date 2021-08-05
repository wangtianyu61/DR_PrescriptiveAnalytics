# Codes for the NeurIPS 2021 Submission
## Setup for 5.1 Newsvendor Problem
We follow the simulation in [Bertsimas and Van Parys (2017)](https://arxiv.org/abs/1711.09974). 

Run the code ``Codes/newsvendor_upd.py`` and adjust the experiment parameter ``exp_choice``. Different experiment hyperparameters in $\{1,2,3,4\}$ correspond to the subfigure in the paper with $1:(a), 2:(b), 3:(c), 4:(d).$ in Figure 1. The output is in the folder ``Output``.

## Setup for 5.2 Portfolio Allocation
The empirical datasets are retrieved from [Kenneth French's Website](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). We download the monthly datasets from July 1963 to December 2018 in the folder ``PortfolioData``.  

Run the code ``portfolio.py`` and change the parameter of datasets (i.e. Line 171) to different datasets. Then we would obtain the performance of different metrics across different methods.