import argparse, os,sys
from os.path import basename
from numpy import *
import numpy as np
import os
import glob
import scipy
np.set_printoptions(suppress=True, precision=3)  

def write_to_file(f, o):
    sumfile = open(f , "w") 
    sumfile.writelines(o)
    sumfile.close()

def print_update(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()


output_wd = ""
bin_size = 1.
max_root = 40
min_q, max_q = 0.005, 0.05
# epochs + early/mid/late Miocene
rate_shifts = np.array([0, 
                        2.58,
                        # 3.6,
                        5.333,
                        # 7.246,
                        11.63,
                        # 13.82,
                        15.97,
                        # 20.44,
                        23.03,
                        # 28.1,
                        33.9])[::-1]
                        

rate_shifts_sorted = np.sort(rate_shifts)
lower_res = True

def write_pyrate_input(sim_record, sim_record_LR, sp_names, filename="test"):
        data = "#!/usr/bin/env python\nfrom numpy import * \n\n"
        d = "\nd=[sim_data]"
        names = "\nnames=['%s']" % (filename)
        data += "\nsim_data = %s"  % (sim_record)
        taxa_names="\ntaxa_names=["
        for i in range(len(sp_names)): 
            taxa_names+= "'%s'" % (sp_names[i])
            if i < len(sim_record)-1: taxa_names +=","
        taxa_names += "]\ndef get_taxa_names(): return taxa_names\n"                     
        f = "\ndef get_data(i): return d[i]\ndef get_out_name(i): return names[i]"
        all_d = data + d + names + taxa_names + f
        write_to_file(os.path.join(output_wd, "%s.py" % filename), all_d)     

        data="#!/usr/bin/env python\nfrom numpy import * \n\n"
        d="\nd=[sim_data]"
        names="\nnames=['%s']" % (filename)
        data += "\nsim_data = %s"  % (sim_record_LR)
        taxa_names="\ntaxa_names=["
        for i in range(len(sp_names)): 
            taxa_names+= "'%s'" % (sp_names[i])
            if i < len(sim_record_LR)-1: taxa_names +=","
        taxa_names += "]\ndef get_taxa_names(): return taxa_names\n"                     
        f="\ndef get_data(i): return d[i]\ndef get_out_name(i): return names[i]"
        all_d=data+d+names+taxa_names+f
        write_to_file(os.path.join(output_wd, "%s_lr.py" % filename), all_d)     

def fossilize(ts, te, min_q, max_q, rate_shifts):
    rates = np.random.uniform(min_q, max_q, len(rate_shifts))
    sim_record = []
    sim_record_LR = []
    sp_names = []
    for sp_i in range(len(ts)):
        fad = ts[sp_i]
        lad = te[sp_i]
        
        rate_shifts_included = rate_shifts[rate_shifts > lad]
        rate_shifts_included = rate_shifts_included[rate_shifts_included < fad]

        comb = np.sort(np.concatenate((np.array([fad,lad]), rate_shifts_included)))

        indx = np.digitize(comb, rate_shifts)
        rates_i = rates[indx]

        sp_record = []
        for i in range(len(comb)-1):
            dt = comb[i+1] - comb[i]
            n_foss = np.random.poisson(rates_i[i] * dt)
            if n_foss:
                foss = np.sort(np.random.uniform(comb[i], comb[i+1], n_foss))
                sp_record = sp_record + list(foss)
        
        if len(sp_record):
            x = np.sort(np.array(sp_record))
            d = np.digitize(x, rate_shifts_sorted)
            low_res_x = np.random.uniform(rate_shifts_sorted[d-1], rate_shifts_sorted[d])
            if lad == 0: 
                x = np.insert(x, 0, 0) 
                low_res_x = np.insert(low_res_x, 0, 0) 
            sim_record.append(x)
            sim_record_LR.append(low_res_x)
            sp_names.append("sp_%s" % sp_i)
    return sim_record, sim_record_LR, sp_names

def getDT_equalbin(T,s,e): 
    B=np.sort(np.append(T,T[0]+1))+.0001
    bin_size= abs(T[0]-T[1])
    ss1 = np.histogram(s,bins=B)[0]
    e[s==e] -= bin_size
    ee2 = np.histogram(e,bins=B)[0]
    DD=(ss1-ee2)[::-1]
    return np.cumsum(DD)[0:len(T)] 

def get_fad_lad(fossils):
    fadlad_tbl = np.zeros((len(fossils), 2))
    for i in range(len(fossils)):
        fadlad_tbl[i, 0] = np.max(fossils[i])
        fadlad_tbl[i, 1] = np.min(fossils[i])        
    return fadlad_tbl

def get_fossil_count(T, s, e):
    B=np.sort(np.append(T,T[0]+1))+.0001
    ss1 = np.histogram(s,bins=B)[0]
    ee1 = np.histogram(e[e != s],bins=B)[0]
    return (ss1 + ee1)[::-1]
    

if __name__ == '__main__': 
    # np.random.seed(1234)
    from bd_sim import *
    ts, te = run_sim()
    # print("ts", ts)
    # print("te", te)
    sim_record, sim_record_LR, sp_names = fossilize(ts, te, min_q, max_q, rate_shifts)
    # write_pyrate_input(sim_record, sim_record_LR, sp_names)
    tbl = get_fad_lad(sim_record)
    print(tbl)
    # get range-through trajectory
    n_bins = np.floor(max_root / bin_size) + 1
    time_bins = np.linspace(0, max_root, int(n_bins))[::-1]
    range_through_traj = getDT_equalbin(time_bins, tbl[:,0], tbl[:,1])
    print(range_through_traj) # min boundary for BB
    fossil_count = get_fossil_count(time_bins, tbl[:,0], tbl[:,1])
    print(fossil_count)
    
    
    
    
    #TODO: histogram per species to get range-through diversity and fossils per time bin


        
