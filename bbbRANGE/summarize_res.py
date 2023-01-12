import glob, os
import numpy as np
import pandas as pd
import .rangeBBB as bbb

wd = "/Users/dsilvestro/Documents/Projects/Ongoing/bbbRANGE/bbbRANGE"
files=glob.glob(os.path.join(wd, "*.log"))


for indx in range(len(files)):
    tbl = pd.read_csv(files[indx], delimiter="\t")     
    
