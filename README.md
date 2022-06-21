# Research Project CSE3000

This repository links to the work of students for the Research project course of the CSE bachelor at TU Delft.

Please see their projects [here](https://cse3000-research-project.github.io/).
<br/>
<br/>
This code supports the research paper "Can Invariant Risk Minimization resist the temptation of learning spurious correlations?".

### Author
Jochem van Lith <br/>
j.a.e.vanlith@student.tudelft.nl

### Supervisors
Rickard Karlsson <br/>
Stephan Bongers <br/>
Jesse Krijthe <br/>

### Instructions
Run main.py to obtain the results of the experiments. <br/>
Use command --experiment to specify the wanted experiment, choose from: <br/>
1. y_noise
2. values_training_environments
3. deviation_training_environments
4. sample_complexity

Run data_plot.py to obtain instances of the training environments. 
<br/> <br/>
The results will appear in the directory ./results

### Sources
This code builds upon this original code: https://github.com/facebookresearch/InvariantRiskMinimization
