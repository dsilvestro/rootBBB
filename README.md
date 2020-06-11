## Root age estimation using a Bayesian Brownian bridge

### Requirements
The program requires Python v.3 (it has been tested with 3.7 and 3.8) and the following libraries: matplotlib, scipy, numpy.

### Main commands

~~~python
'-fossil_data': type=str, help='table with fossil counts per bin (see example files)'

'-div_table': type=str, help='table with present diversity per lineage (see example files)'

'-clade_name', type=str, help='Name of the analyzed clade in input files'

'-n': type=int, help='n. MCMC iterations (default = 25000)'

'-s': type=int, help='sampling freq (default = 100)'

'-seed': type=int, help='random seed (default = U[1000,9999])'

~~~

### Usage
```
python3 rootBBB.py -fossil_data fossil_counts_2.5.txt -div_table family_diversity.txt -clade_name acanthaceae

```

### Running on simulated data
```
python3 rootBBB.py -sim 10 -seed 9600

```
This command will sequentially simulate and analyze 10 datasets. The output files are saved in the same directory as the program and include:
1. a plot of the simulated trajectory 
