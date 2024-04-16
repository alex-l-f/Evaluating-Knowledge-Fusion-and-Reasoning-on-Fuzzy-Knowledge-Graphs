# Files
## PCA_plot.py
Generates and creates a video of a 3D scatter plot from a PCA analysis of a specificed pretrained UKGE model.
## evaluate_model.py
Evaluates pretrained models across multiple folds with the relevant test set for each one. Computes loss and ranking metrics.
## plot_epochs.py
Creates a plot of the epochs till convergence based on the training logs of multiple models.
## plot_evals.py
Takes the output of **evaluate_model.py** or **run_PRA.py** and produces plots and tables.
## run_PRA
Runs my naieve path ranking algorithm implementation for a configured dataset. 
It's EXTREMELY slow, and I strongly recommened running it through [PyPy](https://www.pypy.org/index.html) if you want it to complete any trials in a reasonable amount of time.
