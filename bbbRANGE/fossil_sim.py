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

def fossilize(ts, te, min_q, max_q, rate_shifts, freq_zero_preservation, q_log_mean_sd=None, alpha=0.005):
    if q_log_mean_sd is not None:
        rates = np.exp(np.random.normal(q_log_mean_sd[0], q_log_mean_sd[1], len(rate_shifts)))
    else:
        rates = np.random.uniform(min_q, max_q, len(rate_shifts))
    
    if freq_zero_preservation:
        rates = rates * np.random.binomial(1, 1-freq_zero_preservation, len(rates))
    
    rate_shifts_sorted = np.sort(rate_shifts)
    sim_record = []
    sim_record_LR = []
    sp_names = []
    
    # gamma model
    sp_gamma_mul = np.random.gamma(alpha, 1/alpha, len(ts))
    
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
            n_foss = np.random.poisson(rates_i[i] * dt * sp_gamma_mul[sp_i])
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
    return sim_record, sim_record_LR, sp_names, rates

def getDT_equalbin(T,s,e): 
    # T = np.sort(T)
    # return np.array([len(s[s>t])-len(s[e>t]) for t in T])
    B=np.sort(np.append(T,T[0]+1))+.0001
    bin_size= abs(T[0]-T[1])
    ss1 = np.histogram(s,bins=B)[0]
    # e[s==e] -= bin_size
    ee2 = np.histogram(e,bins=B)[0]
    DD=(ss1-ee2)[::-1]
    # print("DIVERSITY")
    # print(np.cumsum(DD)[0:len(T)][::-1])
    
    T = np.sort(T)
    counts = []
    for i in range(1, len(T)):
        e_tmp = np.zeros(len(e))
        s_tmp = np.zeros(len(e))
        e_tmp[e > T[i-1]] = 1
        e_tmp[e > T[i]] = 0
        s_tmp[s > T[i-1]] = 1
        s_tmp[s > T[i]] = 0
        s_tmp[e_tmp == 1] = 0
        
        sampled_in_bin = np.sum(s_tmp) + np.sum(e_tmp)
        
        e_tmp = np.zeros(len(e))
        s_tmp = np.zeros(len(e))
        e_tmp[e <= T[i - 1]] = 1
        s_tmp[s > T[i]] = 1
        sampled_in_bin += np.sum(s_tmp * e_tmp)
        counts.append(sampled_in_bin)

    # print(np.array(counts).astype(int))
    return np.array(counts).astype(int)
    
def get_fad_lad(fossils):
    fadlad_tbl = np.zeros((len(fossils), 3))
    for i in range(len(fossils)):
        fadlad_tbl[i, 0] = np.max(fossils[i])
        fadlad_tbl[i, 1] = np.min(fossils[i])        
        fadlad_tbl[i, 2] = len(fossils[i])        
    return fadlad_tbl

def get_fossil_count(T, s, e):
    T = np.sort(T)
    counts = []
    for i in range(1, len(T)):
        e_tmp = np.zeros(len(e))
        s_tmp = np.zeros(len(e))
        e_tmp[e > T[i-1]] = 1
        e_tmp[e > T[i]] = 0
        s_tmp[s > T[i-1]] = 1
        s_tmp[s > T[i]] = 0
        s_tmp[e_tmp == 1] = 0 # count singletons only once
        counts.append(np.sum(s_tmp) + np.sum(e_tmp))

    return np.array(counts).astype(int)
    
def generate_bbb_data(ts, te,
                      bin_size=1.,
                      time_bins=None,
                      max_root=0,
                      q_range=[0.005, 0.05],
                      q_log_mean_sd=None,
                      rate_shifts=None,
                      avg_n_q_rate_shifts=0,
                      freq_zero_preservation=0,
                      debug=False,
                      ):
    fossil_count = np.zeros(2)
    
    while len(fossil_count[fossil_count > 0]) < 2: # fossils in at least 2 bins
        true_root = np.max(ts)
        if max_root < true_root:
            max_root = true_root + bin_size * 5
        if rate_shifts is None:
            rate_shifts = np.linspace(0, max_root, 2 + np.random.poisson(avg_n_q_rate_shifts))[::-1] # +2 to ensure the boundaries are included
        
        [min_q, max_q] = q_range
        sim_record, sim_record_LR, sp_names, q_rates = fossilize(ts, te, min_q, max_q, rate_shifts, freq_zero_preservation, q_log_mean_sd)
        tbl = get_fad_lad(sim_record)
        # get range-through trajectory
    
        if time_bins is None:
            time_bins = np.linspace(0, true_root, int(n_bins)) #[::-1]
            n_bins = np.floor(max_root / bin_size) + 1
        else:
            n_bins = len(time_bins)
        range_through_traj = getDT_equalbin(time_bins, tbl[:,0], tbl[:,1])
        true_range_through_traj = getDT_equalbin(time_bins, ts, te)
        fossil_count = get_fossil_count(time_bins, tbl[:,0], tbl[:,1])
        if debug:
            print(tbl)
            print("len(fossil_count[fossil_count > 0])", len(fossil_count[fossil_count > 0]))
            print(fossil_count)
            print("time_bins", time_bins)
            print(range_through_traj) # min boundary for BB
            print('q_rates', q_rates)
            print('rate_shifts (q)', rate_shifts)
        try:
            res = {
                'ts': ts,
                'te': te,
                'n_bins': n_bins,
                'time_bins': time_bins,
                'range_through_traj': range_through_traj,
                'fossil_count': fossil_count,
                'n_extant': len(te[te == 0]),
                'oldest_occ': np.max(tbl[:,0]),
                'fadlad_tbl': tbl,
                'youngest_occ': np.min(tbl[:,1]),
                'true_range_through_traj': true_range_through_traj,
                'avg_q': np.mean(q_rates),
                'q_rates': q_rates
        
            }
        except:
            pass
    return(res)

if __name__ == '__main__': 
    

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


        
