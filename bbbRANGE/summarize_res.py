import glob, os
import numpy as np
import pandas as pd
import rangeBBB as bbb

wd = "/Users/dsilvestro/Documents/Projects/Ongoing/bbbRANGE/bbbRANGE"
files=glob.glob(os.path.join(wd, "*.log"))
burnin=10

res = []

for indx in range(len(files)):
    tbl = pd.read_csv(files[indx], delimiter="\t") 
    # exclude burnin
    tbl = tbl.iloc[burnin:,:]    
    
    
    Nfossils = tbl['Nfossils'][burnin]
    root_obs = tbl['root_obs'][burnin]
    
    root_est = tbl['root_est']
    ext_est = tbl['ext_est']
    root_true = tbl['root_true'][burnin]
    ext_true = tbl['ext_true'][burnin]
    
    root_hpd = list(bbb.calcHPD(root_est))
    root_mean = [np.mean(root_est)]
    
    res_sim = [Nfossils, root_obs]
    res_sim = res_sim + [root_true] + root_mean + root_hpd
    
    if ext_true > 0:
        ext_hpd = list(bbb.calcHPD(ext_est))
        ext_mean = [np.mean(ext_est)]
    else:
        ext_hpd = [0, 0]
        ext_mean = [0]
            
    res_sim = res_sim + [ext_true] + ext_mean + ext_hpd
    res.append(res_sim)


res_tbl = pd.DataFrame(res)
res_tbl.columns = [
    'Nfossils', 'root_obs',
    'root_true', 'root_est', 'root_m', 'root_M',
    'ext_true', 'ext_est', 'ext_m', 'ext_M'
]    

res_tbl.to_csv(os.path.join(wd, "summary_res.txt"), sep="\t")