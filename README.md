## Root age estimation using a Bayesian Brownian bridge

The BBB model implemented in this program is described in the upcoming paper _"Fossil data support a pre-Cretaceous origin of flowering plants"_ by 
Daniele Silvestro, Christine D. Bacon, Wenna Ding, Qiuyue Zhang, Philip C. J. Donoghue, Alexandre Antonelli, and Yaowu Xing.


### Requirements
The program requires Python v.3 (it has been tested with 3.7 and 3.8) and the following libraries: `matplotlib`, `scipy`, `numpy`.

### Main commands

~~~python
'-fossil_data': type=str, help='table with fossil counts per bin (see example files)'

'-div_table': type=str, help='table with present diversity per lineage (see example files)'

'-clade_name', type=str, help='Name of the analyzed clade in input files'

'-n': type=int, help='n. MCMC iterations (default = 25000)'

'-s': type=int, help='sampling frequency (default = 100)'

'-seed': type=int, help='random seed (default = U[1000,9999])'

'-q_var': type=int, help='0) constant sampling rate 1) linearly increasing sampling rate'

'-max_age',  type=int,   help='Max boundary of uniform prior on the root age' (default = 300)
~~~

### Usage
```
python3 rootBBB.py -fossil_data fossil_counts_2.5.txt -div_table family_diversity.txt -q_var 1 -clade_name acanthaceae

```

### Running on simulated data
```
python3 rootBBB.py -sim 10 -seed 9600

```
This command will sequentially simulate and analyze 10 datasets. The output files are saved in the same directory as the program and include:  
1. a plot of the simulated trajectory  
2. a log file with the posterior samples  
3. a text file with the summary of true and estimated parameters across all simulations.

### Example datasets
The file `family_diversity.txt` includes a list of angiosperm families and they modern species richness. The `fossil_counts*.txt` files include counts of number of species sampled for each family in time bins of 1, 2.5, and 5 Myr (files named `*with_pollen` include few selected pollen occurrences).  