# active-learning-emi

This code can be used to replicate the numerical examples in the 2025 AIAA Aviation paper: https://arc.aiaa.org/doi/abs/10.2514/6.2025-3797

This is rough "research quality" code. 

The main file for the Neural Network examples is main_e2nn_example.py. Select which (and how many) iterations to run based on line 39. Select the desired example problem to run by uncommenting the corresponding block of code (section starting on line 77). Options are:
(1) 2D Rosenbrock example
(2) Multimodal system
(3) Aircraft lift force


The main file for the Gaussian Process Regression examples is main_gpr_example.py. This will run both the EGRA and EMI acquisition functions. To only run one of the acquisition functions, modify line 34 to only contain one acquisition function name. To change which example problem to run, uncomment the corresponding block of code (section starting on line 68). 


