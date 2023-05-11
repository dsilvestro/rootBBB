from datetime import datetime
import sys
import argparse
import numpy as np
np.set_printoptions(suppress=True, precision=3)  
import scipy.stats
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('-fadlad_data',type=str,   help='table with fossil counts per bin', default = None)
p.add_argument('-fossil_data',type=str,   help='table with fossil counts per bin', default = "")
p.add_argument('-div_table' ,type=str,   help='table with present diversity per lineage', default = "")
p.add_argument('-n',        type=int,   help='n. MCMC iterations', default = 25000)
p.add_argument('-s',        type=int,   help='sampling freq', default = 100)
p.add_argument('-p',        type=int,   help='print freq', default = 1000)
p.add_argument('-plot',     type=int,   help='plot simulated Brownian bridge', default = 1)
p.add_argument('-seed',     type=int,   help='random seed', default = -1)
p.add_argument('-verbose',  type=int,   help='verbose', default = 1)
p.add_argument('-sim',      type=int,   help='if >1 run simulations', default = 0)
p.add_argument('-sim_extinct', type=int,  help='0: simulate extant clades; 1: simulate extinct clades', default = 0)
p.add_argument('-sim_range', type=int,  help='1: simulate range data', default = 0)
p.add_argument('-biased_q', type=int,   help='if 1 set increasing q through time', default = 0)
p.add_argument('-freq_q0',  type=float, help='frequency of 0-sampling rate', default = 0.1)
p.add_argument('-q_var',    type=int,   help='0) constant q 1) linearly increasing q', default = 0)
p.add_argument('-q_exp',    type=int,   help='magnitude of variation', default = 1)
p.add_argument('-q_gap',    type=int,   help='0) constant q 1) gap model', default = 0)
p.add_argument('-clades',   type=int,   help='range of clade to be analyzed', default = [0,0], nargs=2)
p.add_argument('-outpath',  type=str,   help='path output', default = ".")
p.add_argument('-clade_name', type=str,   help='', default = "")
p.add_argument('-f',        type=float, help='freq. DA runs with updates', default = 0.95)
p.add_argument('-out',      type=str,   help='add string to output', default = "")
p.add_argument('-nDA',      type=int,   help='DA samples', default = 1000)
p.add_argument('-DAbatch',  type=int,   help='DA batch size (if set to 0: auto-tune)', default = 0)
p.add_argument('-ws',       type=float, help='win sizes root, sig2, q', default = [10,1.25,1.25], nargs=3)
p.add_argument('-max_age',  type=int,   help='Max boundary of uniform prior on the root age', default = 300)
p.add_argument('-q_prior',  type=float,   help='shape and rate (default: 1.1, 1)', default = [1.1, 1], nargs=2)
p.add_argument('-q_min',    type=float,   help='offset for q', default = 0)
p.add_argument('-debug', type=int,  help='1: debug mode', default = 0)



print("""

Clade age estimator using a Bayesian Brownian bridge.

""")


args = p.parse_args()

if args.seed== -1:
    seed =np.random.randint(1000,9999)
else:
    seed =args.seed
np.random.seed(seed)
small_number = 10e-10

n_simulations = args.sim
save_mcmc_samples = 1
max_age = args.max_age
mid_points = np.linspace(0,2*max_age,int(2*max_age/2.5)+1)
bin_size = np.abs(np.diff(mid_points)[0])
n_DA_samples = args.nDA
freq_par_updates = args.f
verbose = args.verbose
DAbatch = args.DAbatch
if DAbatch == 0:
    DAbatch += n_DA_samples
q_var_model = args.q_var
[alpha_q, beta_q] = args.q_prior
q_offset = args.q_min

# simulation settings
increasing_q_rates = args.biased_q
freq_zero_preservation = args.freq_q0
bias_exp = args.q_exp

simulate_extinct = args.sim_extinct
run_range_simulations = args.sim_range

run_simulations = np.min([1,n_simulations])
gap_model = args.q_gap


DEBUG = args.debug

def print_update(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()

def approx_log_fact(n):
    # http://mathworld.wolfram.com/StirlingsApproximation.html
    return np.log(np.sqrt((2*n+1./3)*np.pi)) + n*np.log(n) -n

def get_log_factorial(n):
    if n < 100: return np.log(scipy.special.factorial(n))
    else: return approx_log_fact(n)

def get_log_binomial_coefficient(n, k):
    # np.log(scipy.special.binom(n, k))
    return get_log_factorial(n) - (get_log_factorial(k) + get_log_factorial(n-k))

def approx_log_binomiam_pmf(n, k, p):
    return get_log_binomial_coefficient(n, k) + k*np.log(p) + (n-k)*np.log(1-p)

def binomial_pmf(x,n,p):
    # binomial distribution 
    " P(k | n, p)  = n!/(k! * (n-k)!) * p^k * (1-p)^(n-k)   "
    " scipy.stats.logpmf(x, n, p, loc=0) "
    pmf = scipy.stats.binom.logpmf(x, n, p, loc=0)
    return pmf
    #if pmf > -np.inf:
    #    return pmf
    #else: 
    #    return approx_log_binomiam_pmf(x,n,p)

def normal_pdf(x,m,s):
    return scipy.stats.norm.logpdf(x, loc=m, scale=s)

def gamma_pdf(x,a,b):
    return scipy.stats.gamma.logpdf(x, a, scale=1./b)

def beta_pdf(x,a,b):
    return scipy.stats.beta.logpdf(x,a,b)

def update_normal(q,d=1,m=-1,M=1):
    ii = np.random.normal(q,d)
    if ii<m: ii=(ii-m)+m
    if ii>M: ii=(M-(ii-M))
    if ii<m: ii=q
    return ii, 0

def update_multiplier(q,d=1.1):
    u = np.random.uniform(0,1)
    l = 2*np.log(d)
    m = np.exp(l*(u-.5))
    new_q = q * m
    return new_q, np.log(m)

def update_uniform(i, m, M, d): 
    ii = i+(np.random.random()-.5)*d
    if ii<m: ii=(ii-m)+m
    if ii>M: ii=(M-(ii-M))
    if ii<m: ii=i
    return ii, 0

def calcHPD(data, level=0.95) :
    assert (0 < level < 1)    
    d = list(data)
    d.sort()    
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        raise RuntimeError("not enough data")    
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)) :
        rk = d[k+nIn-1] - d[k]
        if rk < r :
            r = rk
            i = k
    assert 0 <= i <= i+nIn-1 < len(d)    
    return np.array([d[i], d[i+nIn-1]])

def get_fossil_count_occs(data):
    x = data[::-1] + 0  # order is temporarily reversed
    j,c=0,0
    for i in range(len(x)):
        if x[i]==0 and j==0:
            c+=1
        else:
            break
    
    x = x[c:][::-1]
    return x

def sample_path_batch_discrete(time_bins = np.arange(100), n_reps = 1, y_start = 0, sig2 = 1., positive=0):
    timespan = np.max(time_bins) - np.min(time_bins)
    n_steps = len(time_bins)
    dt = timespan / (n_steps -1) # assumes equal bins 
    dt_sqrt = np.sqrt(dt*sig2)
    B = np.zeros((n_reps, n_steps))
    B[:, 0] = y_start
    for n in np.arange(n_steps - 2):
        t = n * dt
        xi = np.random.randn(n_reps) * dt_sqrt
        new_val = B[:, n] * (1 - dt / (timespan - t)) + xi
        if positive:
            new_val = np.abs(new_val)
        
        B[:, n + 1] = new_val
    B[:, -1] = 0 # added: set the last B to Zero
    return B
    
def get_imputations(Nobs, fossil_data, est_root_r, est_ext, est_sig2, n_samples=1000, DAbatch=1000):    
    est_root = est_root_r - est_ext
    mid_points_shift = mid_points[mid_points < est_root_r] 
    if DEBUG:
        print(mid_points_shift, len(mid_points_shift))
        print(est_root_r, est_ext)
        print(fossil_data, len(fossil_data))
    
    x_augmented = np.zeros(len(mid_points_shift))
    x_augmented[0:len(fossil_data)] = fossil_data + 0    
    
    x_augmented = x_augmented[mid_points_shift > est_ext]
    mid_points_shift = mid_points_shift[mid_points_shift > est_ext]
    if DEBUG:
        print("mid_points_shift", est_ext)
    
    time_bins = mid_points_shift
    simTraj_all = np.zeros((n_samples,len(time_bins)))
    j=0
    counter = 0
    while j <= n_samples:
    #if 2>1:
        # simulate expected trajectories
        simTraj = sample_path_batch_discrete(time_bins, n_reps=DAbatch, sig2=est_sig2, y_start=Nobs)
        simTraj = np.rint(simTraj)+1
        m = np.min(simTraj-x_augmented,axis=1)
                
        #remove incompatible trajectories
        simTraj = simTraj[m>=0]
        #remove a priori incompatible trajectories
        m = np.min(simTraj,axis=1)
        simTraj = simTraj[m>=1]
        valid_samples = simTraj.shape[0]
        simTraj_all[j:np.min([n_samples, j+valid_samples]), :] = simTraj[0:np.min([valid_samples, n_samples-j]), :]
        
        j += valid_samples
        counter +=1
        if counter > 100:
            #print("reach max limit, found:",j)
            return [], 0
    return x_augmented, simTraj_all
    
def get_avg_likelihood(Nobs, fossil_data, est_root, est_sig2, est_q, est_a, gap_pr, x_augmented, simTraj_all):
    if len(x_augmented)==0:
        return -np.inf, 0
    else:
        dt = bin_size/100 # rescale time axis to determine the slope of the q-increase
        # print(dt, 1./age_oldest_obs_occ)
        time_vec = np.arange(len(x_augmented)).astype(float)[::-1] 
        time_vec *= dt
        q_vec = est_q * np.exp(est_a*time_vec)
        # print(np.log(q_vec)) #<- with est_a > 0 is highest at the recent
        # print(np.log(est_q))# <- est_q ~ q at the root
        
        sampling_prob_vec = np.ones(len(x_augmented))- np.exp(-q_vec)
        # print(np.log(sampling_prob_vec))
        log_lik_matrix = binomial_pmf(x_augmented,simTraj_all,sampling_prob_vec)
        log_lik_matrix_ =log_lik_matrix +0
        if gap_model > 0:
            p1 = np.exp(log_lik_matrix) * (1 - gap_pr)
            p2 = np.ones(len(x_augmented)) * (x_augmented == 0) * gap_pr
            p3 = p1 + p2
            p3[p3 == 0] = small_number

            log_lik_matrix = np.log(p3)
            # print(p2)
        # print(x_augmented)
        # print(simTraj_all[0])
        # print(np.sum(log_lik_matrix), gap_pr, est_root,
        # np.sum(np.log(np.exp(log_lik_matrix_) * (1 - gap_pr))),
        # np.log(gap_pr ** len(x_augmented[x_augmented == 0]) + (1-gap_pr) ** len(x_augmented[x_augmented > 0])),
        # # np.sum(np.log(np.exp(log_lik_matrix))),
        # np.sum(log_lik_matrix_)
        # )
       
        
        
        # lik_avg = np.log(np.mean(np.exp( np.sum(lik_matrix,axis=1) )))
        # to avoid underflow:
        lik_i = np.sum(log_lik_matrix,axis=1)
        lik_max = np.max(lik_i)
        lik_avg = np.log(np.sum(np.exp( lik_i-lik_max ))) - np.log(simTraj_all.shape[0]) + lik_max
        #print( lik_avg,lik_max , len(lik_i[np.exp(lik_i)>0]), len(lik_i), n_samples*counter, counter )
        #print(lik_i)
        #quit()
        return lik_avg, len(lik_i[np.exp(lik_i-lik_max)>0])

def run_mcmc(age_oldest_obs_occ, age_youngest_obs_occ, x, log_Nobs, Nobs, sim_n = 0, DAbatch=DAbatch, bbb_condition=None):
    lik_A = np.nan
    tries = 0
    
    if bbb_condition is None:
        bbb_condition = x + 0 
    # initialize MCMC
    print("\n\nInitializing the model...")
    while np.isnan(lik_A):
        gap_pr_A = 0.
        est_a_A = 0.
        est_ext_A = 0. # ext age multiplier of age_youngest_obs_occ \in (0, 1)
        if Nobs == 0:
            est_ext_A = np.random.uniform(age_youngest_obs_occ,0) 
        if tries % 10 == 0: 
            print_update("Attempt %s" % tries)
            DAbatch += 100 # auto-adjust batch size
        
        if tries <= 100:
            # print("Attempt 1...")
            # init root age
            #est_root_A = np.random.random()
            est_root_A = np.min([age_oldest_obs_occ*(1+np.random.uniform(0.05,0.25 )), max_age])
            # init sig2
            est_sig2_A = np.log(1 + np.random.uniform(10, 50)*np.max([1, Nobs]))
            # init q_rate
            est_q_A = np.random.uniform(0.0005, 0.002)
            x_augmented_A, simTraj_all_A = get_imputations(Nobs, bbb_condition, est_root_A, est_ext_A, np.exp(est_sig2_A), 
                                                           n_samples=n_DA_samples,
                                                           DAbatch=DAbatch)
            if len(x_augmented_A)==0:
                # ie imputation didn't work
                lik_A = np.nan
            else:
                lik_A, DA_counts = get_avg_likelihood(Nobs, x, est_root_A, np.exp(est_sig2_A), est_q_A, est_a_A, gap_pr_A, x_augmented_A, simTraj_all_A)
                #prior_A = gamma_pdf(est_sig2_A,a=1.,b=1.)
                prior_A = gamma_pdf(np.exp(est_sig2_A-log_Nobs),a=1.,b=.1) + gamma_pdf(est_q_A,a=alpha_q,b=beta_q) + gamma_pdf(est_a_A,1,0.01) 
        elif tries <= 200:
            # print("Attempt 2...")
            # init root age
            #est_root_A = np.random.random()
            est_root_A = np.min([age_oldest_obs_occ*(1+np.random.uniform(0.05,0.25 )), max_age])
            # init sig2
            est_sig2_A = np.log(np.random.uniform(0, 5)*np.max([1, Nobs]))
            # init q_rate
            est_q_A = np.random.uniform(0.0005, 0.002)
            x_augmented_A, simTraj_all_A = get_imputations(Nobs, bbb_condition, est_root_A, est_ext_A, np.exp(est_sig2_A), 
                                                           n_samples=n_DA_samples,
                                                           DAbatch=DAbatch)
                                                            
            if len(x_augmented_A)==0:
                # ie imputation didn't work
                lik_A = np.nan
            else:
                lik_A, DA_counts = get_avg_likelihood(Nobs, x, est_root_A, np.exp(est_sig2_A), est_q_A, est_a_A, gap_pr_A, x_augmented_A, simTraj_all_A)
                #prior_A = gamma_pdf(est_sig2_A,a=1.,b=1.)
                prior_A = gamma_pdf(np.exp(est_sig2_A-log_Nobs),a=1.,b=.1) + gamma_pdf(est_q_A,a=alpha_q,b=beta_q) + gamma_pdf(est_a_A,1,0.01) 
        elif tries <= 10000:
            # print("Attempt 3...")
            est_root_A =  np.min([age_oldest_obs_occ*(1+np.random.uniform(0.05,1 )), max_age])
            # init sig2
            est_sig2_A = np.log(np.random.uniform(0.1, 100)*np.max([1, Nobs]))
            # init q_rate
            est_q_A = np.random.uniform(0.0005, 0.1)

            x_augmented_A, simTraj_all_A = get_imputations(Nobs, bbb_condition, est_root_A, est_ext_A, np.exp(est_sig2_A),
                                                           n_samples=n_DA_samples,
                                                           DAbatch=DAbatch)
            
            if len(x_augmented_A)==0:
                # ie imputation didn't work
                lik_A = np.nan
            else:
                lik_A, DA_counts = get_avg_likelihood(Nobs, x, est_root_A, np.exp(est_sig2_A), est_q_A, est_a_A, gap_pr_A, x_augmented_A, simTraj_all_A)
                #prior_A = gamma_pdf(est_sig2_A,a=1.,b=1.)
                prior_A = gamma_pdf(np.exp(est_sig2_A-log_Nobs),a=1.,b=.1) + gamma_pdf(est_q_A,a=alpha_q,b=beta_q) + gamma_pdf(est_a_A,1,0.01) 
        else:
            sys.exit("Failed to initialize model.")
        tries+=1
    print_update("Model initialized after %s iterations (DAbatch: %s)." % (tries, DAbatch))
        
    print("\n\nRunning MCMC...")
    if save_mcmc_samples:
        model_out = ""
        if increasing_q_rates:
            model_out = model_out + "_qbias%s" % bias_exp
        if q_var_model:
            model_out = model_out + "_qvar"
        if gap_model:
            model_out = model_out + "_qgap"
        if run_simulations:
            out_name = "%s/mcmc_%s_%s_f%s%s%s.log" % (args.outpath, sim_n, seed, freq_par_updates, args.out, model_out)
            logfile = open(out_name, "w") 
            text_str = "iteration\tposterior\tlikelihood\tprior\tNobs\tNfossils\troot_obs\text_obs\troot_true\text_true\tq_med_true\tsig2_true\tDA_counts\troot_est\text_est\tq_est\ta_est\tsig2_est\tgap_pr"
            logfile.writelines(text_str)
        else:
            if args.out != "":
                out_name = "%s/%s_%s_mcmc_%s_f%s%s.log" % (args.outpath, sim_n, args.out, seed, freq_par_updates, model_out)
            else:
                out_name = "%s/%s_mcmc_%s_f%s%s.log" % (args.outpath, sim_n, seed, freq_par_updates, model_out)
                
            print("Writing output to:", out_name)
            logfile = open(out_name, "w") 
            text_str = "iteration\tposterior\tlikelihood\tprior\tNobs\tNfossils\troot_obs\text_obs\tDA_counts\troot_est\text_est\tq_est\t\ta_est\tsig2_est"
            logfile.writelines(text_str)
            
    res = np.zeros((int(args.n/args.s), 5))
    if verbose: print(res.shape)
    sampled_iteration=0
    
    #est_sig2_A = np.log(est_sig2_A) # rate is log-transformed in the MCMC
    for iteration in range(args.n):
        accepted = 0
        #start = datetime.now()
        est_root, est_ext, est_sig2, est_q, est_a = est_root_A+0, est_ext_A+0, est_sig2_A+0, est_q_A+0, est_a_A+0
        gap_pr = gap_pr_A + 0
        x_augmented, simTraj_all = x_augmented_A+0, simTraj_all_A+0
        h1,h2,h3 = 0,0,0
        accept = 0
        
        rr = np.random.random(3)
        if rr[0]< freq_par_updates: 
            update = 1
            accept = 0
            
        else: 
            x_augmented, simTraj_all = get_imputations(Nobs, bbb_condition, est_root, est_ext, np.exp(est_sig2), 
                                                       n_samples=n_DA_samples,
                                                       DAbatch=DAbatch)
            update = 0
            accept = 1
        
        if update:
            if rr[2]< 0.7:
                est_root, _ = update_normal(est_root_A , m=age_oldest_obs_occ, M=max_age, d=args.ws[0])
                if Nobs == 0:
                    est_ext, _ = update_normal(est_ext_A , m=0, M=age_youngest_obs_occ, d=args.ws[0])
            if rr[2]> 0.5:
                est_sig2EXP, h2 = update_multiplier(np.exp(est_sig2_A),args.ws[1])
                est_sig2 = np.log(est_sig2EXP)
                est_q   , h3 = update_multiplier(est_q_A ,d=args.ws[2])
                est_q += q_offset
                if q_var_model:
                    est_a, _ = np.abs(update_normal(est_a_A, d=1, m= -100, M=100))
                if gap_model:
                    gap_pr, _ = update_uniform(gap_pr_A, 0, 0.99, 0.05)
                    # print(gap_pr, gap_pr_A)
            x_augmented, simTraj_all = get_imputations(Nobs, bbb_condition, est_root, est_ext, np.exp(est_sig2), 
                                                       n_samples=n_DA_samples,
                                                       DAbatch=DAbatch)
            
            accept = 0
        
        
        
        lik, DA_counts = get_avg_likelihood(Nobs, x, est_root, np.exp(est_sig2), est_q, est_a, gap_pr, x_augmented, simTraj_all)
        
        prior = gamma_pdf(np.exp(est_sig2-log_Nobs),a=1.,b=0.1) + gamma_pdf(est_q,a=alpha_q,b=beta_q) + gamma_pdf(est_a,1,0.01) 
    
        if (lik-lik_A) + (prior-prior_A) + (h1+h2+h3) >= np.log(np.random.random()) or accept==1 and np.isfinite(lik):
            est_root_A = est_root
            est_ext_A = est_ext
            est_sig2_A = est_sig2
            est_q_A    = est_q
            est_a_A    = est_a
            gap_pr_A   = gap_pr
            lik_A      = lik
            prior_A    = prior
            x_augmented_A, simTraj_all_A = x_augmented, simTraj_all 
            accepted = 1
    
        if iteration % args.p == 0 and verbose:
            if iteration == 0:
                print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("it", "lik", "root", "ext", "sig2", "q_rate", "q_slope","q_gap"))
            print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (iteration, round(lik_A,2), round(est_root_A,2), round(est_ext_A,2), round(est_sig2_A,2), 
            round(est_q_A,5), round(est_a_A,5), round(gap_pr_A, 3)))
        
        if iteration % args.s == 0:
            if save_mcmc_samples:
                if run_simulations:
                    text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % \
                    ( iteration, lik_A+prior_A, lik_A, prior_A, Nobs, np.sum(x), age_oldest_obs_occ, age_youngest_obs_occ, true_root, true_ext, \
                    np.median(true_q), \
                    #true_sig2, DA_counts, age_oldest_obs_occ*(1+est_root_A), est_q_A, est_sig2_A)
                    true_sig2, DA_counts, est_root_A, est_ext_A, est_q_A, est_a_A, est_sig2_A, gap_pr_A)
                else:
                    text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % \
                    ( iteration, lik_A+prior_A, lik_A, prior_A, Nobs, np.sum(x), age_oldest_obs_occ, age_youngest_obs_occ,DA_counts, est_root_A,est_ext_A, est_q_A, est_a_A, est_sig2_A)
                logfile.writelines(text_str)
                logfile.flush()
            
            res[sampled_iteration,:] = np.array([lik_A, est_root_A, est_ext_A, est_q_A, est_sig2_A])
            sampled_iteration += 1
    return res

def simulate_data(rseed=0):
    if rseed > 0:
        np.random.seed(rseed)
    age_oldest_obs_occ = 0
    while age_oldest_obs_occ <= 0:    
        true_root = np.random.uniform(10,max_age)
        log_q_mean = -8.52 # 1/5000 mean Nfossil 
        log_q_std = 1

        logNobs = np.random.uniform(np.log(100),np.log(20000)) # np.log(50000.) 
        Nobs = np.rint(np.exp(logNobs))
        log_Nobs = np.log(Nobs)
        true_sig2 = np.random.uniform(10, 50)*Nobs #0.1
        Ntrue = sample_path_batch_discrete(time_bins = mid_points[mid_points<true_root], n_reps =100, sig2=true_sig2, y_start = Nobs,positive=0)
        Ntrue = np.rint(Ntrue)+1 # add +1 to make the bridge start at 1
        m = np.min(Ntrue,axis=1)
        print(m)
        Ntrue = Ntrue[m>=1][0]
        #Ntrue[Ntrue==0] = 1
        logNtrue = np.log(Ntrue)
        true_sig2 = np.log(true_sig2)
    
        # simulate fossil occs
        true_q = np.exp( np.random.normal(log_q_mean,log_q_std,len(mid_points)))[mid_points<true_root]
        if increasing_q_rates:
            true_q = np.random.choice(true_q,len(true_q),replace=0,p=(true_q**bias_exp)/np.sum((true_q**bias_exp)))    
            #true_q = np.sort(true_q)[::-1] # rate increases toward the recent

        true_q = true_q * np.random.binomial(1,1-freq_zero_preservation,len(true_q))
        true_q[true_q>0.1] = 0.1
        # preserved occs
        x = np.rint(Ntrue*true_q)[::-1] # order is temporarily reversed
        # remove first x values if they are == 0
        j,c=0,0
        for i in range(len(x)):
            if x[i]==0 and j==0:
                c+=1
            else:
                break

        x = x[c:][::-1]
        age_oldest_obs_occ = mid_points[len(x)-1]+0
    
        if verbose:
            print(x)
            print("true_root",true_root, "obs_root",age_oldest_obs_occ)
            print("true_q (log10)",np.log10(true_q+0.000000001), np.max(true_q)/np.min(true_q[true_q>0]))
            print( "Ntrue", Ntrue, "Nfossils",np.sum(x))
    
    return true_root, true_q, true_sig2, Nobs, age_oldest_obs_occ, x, log_Nobs, Ntrue

def simulate_extinct_clade(rseed=0):
    if rseed > 0:
        np.random.seed(rseed)
    age_oldest_obs_occ = 0
    age_youngest_obs_occ = 0
    while age_oldest_obs_occ <= 0:    
        true_root = np.random.uniform(100,max_age)
        true_ext = np.random.uniform(10,true_root*0.9)
        true_root_shifted = true_root - true_ext
        log_q_mean = -5 #-8.52 # 1/5000 mean Nfossil 
        log_q_std = 1
        
        Nobs = 0
        log_Nobs = 0
        true_sig2 = np.random.uniform(100, 500)
        indx_clade_life_span = np.array([i for i in range(len(mid_points)) if mid_points[i] < true_root and mid_points[i] > true_ext])
        
        Ntrue = sample_path_batch_discrete(time_bins=mid_points[mid_points<true_root_shifted],
                                           n_reps=100, 
                                           sig2=true_sig2, 
                                           y_start=Nobs,
                                           positive=0)
        Ntrue = np.rint(Ntrue)+1 # add +1 to make the bridge start at 1
        m = np.min(Ntrue,axis=1)
        try:
            # print(m)
            Ntrue = Ntrue[m>=1][0]
        
            # Shift div traj
            Ntrue = np.append(np.zeros(np.min(indx_clade_life_span)), Ntrue)
        
            #Ntrue[Ntrue==0] = 1
            logNtrue = 0
            true_sig2 = np.log(true_sig2)
        
            # simulate fossil occs
            true_q = np.exp( np.random.normal(log_q_mean,log_q_std,len(mid_points)))[0:len(Ntrue)]
            if increasing_q_rates:
                true_q = np.random.choice(true_q,len(true_q),replace=0,p=(true_q**bias_exp)/np.sum((true_q**bias_exp)))    
                #true_q = np.sort(true_q)[::-1] # rate increases toward the recent
        
            true_q = true_q * np.random.binomial(1,1-freq_zero_preservation,len(true_q))
            true_q[true_q>0.1] = 0.1
            # preserved occs
            x = np.rint(Ntrue*true_q)[::-1] # order is temporarily reversed
            # remove first x values if they are == 0
            j,c=0,0
            for i in range(len(x)):
                if x[i]==0 and j==0:
                    c+=1
                else:
                    break
        
            x = x[c:][::-1]
        
            age_oldest_obs_occ = mid_points[len(x)-1]+0
            if len(x):
                age_youngest_obs_occ = mid_points[np.min(np.where(x > 0))] 
            else:
                age_youngest_obs_occ = np.nan
        
            if verbose:
                print(x)
                print("true_root",true_root, "obs_root",age_oldest_obs_occ)
                print("true_ext",true_ext, "obs_ext",age_youngest_obs_occ)
                # print("true_q (log10)",np.log10(true_q+0.000000001), np.max(true_q)/np.min(true_q[true_q>0]))
                print( "Ntrue", Ntrue, "Nfossils",np.sum(x))
                print(indx_clade_life_span)
        except:
            pass
    
    return true_root, true_ext, true_q, true_sig2, Nobs, age_oldest_obs_occ, x, log_Nobs, Ntrue, age_youngest_obs_occ
  

if __name__ == '__main__':  
    if run_simulations and run_range_simulations == 0:
        save_summary = 1
        if save_summary:
            out_name = "%s/summary.txt" % (args.outpath)
            logfile = open(out_name, "w") 
            text_str = "iteration\tNobs\tNfossils\troot_true\troot_obs\tq_med_true\tsig2_true\troot_est\troot_M\troot_m\tq_est\tsig2_est"
            logfile.writelines(text_str)

        print("seed",seed)
        init_seed = seed + 0

        sim_number = 1
        counter = 0
        while sim_number <= n_simulations:
            counter += 1
            print(counter, sim_number)
            if counter > 10000:
                counter *= 0 
        
            print("simulating data...")
            if simulate_extinct:
                true_root, true_ext, true_q, true_sig2, Nobs, age_oldest_obs_occ, x, log_Nobs,Ntrue, age_youngest_obs_occ = simulate_extinct_clade(0)
                # indx_clade_life_span = np.array([i for i in range(len(mid_points)) if mid_points[i] < true_root and mid_points[i] > true_ext])
            else:
                true_root, true_q, true_sig2, Nobs, age_oldest_obs_occ, x, log_Nobs,Ntrue = simulate_data(seed+sim_number)
                age_youngest_obs_occ = 0
                true_ext = 0
        
            indx_clade_life_span = np.array([i for i in range(len(mid_points)) if mid_points[i] < true_root])
        
            if np.sum(x)< 1: 
                print("No fossils:",np.sum(x),"",age_oldest_obs_occ)
                seed_s = "%s%s" % (counter, init_seed) # change seed if it doesn't work
                seed = int(seed_s)
            else:
                if args.plot:
                    mid_points_temp = mid_points[indx_clade_life_span]
                    print("\n\nPLOTTING")
                    print(x)
                    print(indx_clade_life_span)
                    print(len(x), len(mid_points_temp), len(mid_points), len(Ntrue.T))
                    file_name = "%s/sim_data%s_%s.pdf" % (args.outpath,sim_number,seed)
                    fig = plt.figure(figsize=(12, 10))
                    plt.plot(mid_points[0:len(Ntrue)],Ntrue.T)
                
                
                
                    plt.plot(mid_points_temp[np.where(x > 0)],x[x > 0], 'ro')
                    for i in range(len(x)):
                        plt.text(mid_points_temp[i]-(1/150*true_root),-0.035*np.max(Ntrue), '%d' % (int(x[i])))
                
                    plt.plot(np.zeros(int(Nobs)),np.arange(Nobs), 'ro')
                
                    title = "n. extant species: %s   n. fossils: %s   $\sigma^{2} = 10^{%s}$   $q_{avg} = 10^{%s}$" % \
                    (int(Nobs), int(np.sum(x)), np.round(np.log10(true_sig2),2), np.round(np.log10(np.mean(true_q)),2))
                    plt.gca().set_title(title,  fontsize=16)
                    plt.xlabel('Time',  fontsize=14)
                    plt.ylabel('N. species',  fontsize=14)
                    plt.close()
                    plot_divtraj = matplotlib.backends.backend_pdf.PdfPages(file_name)        
                    plot_divtraj.savefig( fig )
                    plot_divtraj.close()
    
                print("replicate:", sim_number)
                print("N. fossils:",np.sum(x),"Obs age:", age_oldest_obs_occ)
                print("x vector:", x)
                print(len(x))            
            
                res=run_mcmc(age_oldest_obs_occ, age_youngest_obs_occ, x, log_Nobs, Nobs, sim_number)

                post_burnin_res = res[ int(res.shape[0]*0.2):, : ]

                par_est = np.mean(post_burnin_res, axis=0)
                root_est_hpd = calcHPD(post_burnin_res[:,1])

                if verbose:
                    print(par_est, root_est_hpd)

                if save_summary:
                    text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % \
                    ( sim_number, Nobs, np.sum(x), true_root, age_oldest_obs_occ, np.median(true_q), \
                    true_sig2, par_est[1], root_est_hpd[0], root_est_hpd[1], par_est[3], par_est[4])
                    logfile.writelines(text_str)
                    logfile.flush()
            
                sim_number += 1
            # quit()

    elif run_range_simulations:
        from bd_sim import *
        from fossil_sim import *
        print("seed",seed)
        init_seed = seed + 0
        
    
        #-- simulation parameters --#
        BIN_SIZE = 1
        mid_points = np.linspace(0,2*max_age,int(2*max_age/BIN_SIZE)+1)
        bin_size = np.abs(np.diff(mid_points)[0])
        
        max_true_root_age = 100
        min_true_root_age = 30
        root_age_range = np.array([max_true_root_age, min_true_root_age])
        n_sp_range = np.array([2000, 20000])
        avg_n_q_rate_shifts=100 # if =0 -> constant preservation
        rangeL = [0.1, 1]
        rangeM = [0.1, 1]
        q_range = np.array([0.0001, 0.01]) + q_offset
        q_log_mean_sd = np.array([np.log(0.01), 0.5])
        print_ltt = True
        # if DEBUG:
        #     print_ltt = True
        #---------------------------#
    
        sim_number = 1
        counter = 0
        while sim_number <= n_simulations:
            counter += 1
            print(counter, sim_number)
        
            ts, te = run_sim(#root_age=root_age,
                             root_r=root_age_range,    
                             rangeSP=n_sp_range,
                             rangeL=rangeL,
                             rangeM=rangeM,
                             print_ltt=print_ltt,
                             poiL = 10,
                             poiM = 10,
                            )
                    
            res = generate_bbb_data(ts, te,
                                    bin_size=bin_size,
                                    time_bins=mid_points,
                                    q_range=q_range,
                                    q_log_mean_sd=q_log_mean_sd,
                                    rate_shifts=mid_points,
                                    avg_n_q_rate_shifts=avg_n_q_rate_shifts, 
                                    freq_zero_preservation=freq_zero_preservation,
                                    debug=DEBUG)
            # {
            #         'ts': ts,
            #         'te': te,
            #         'n_bins': n_bins,
            #         'time_bins': time_bins,
            #         'range_through_traj': range_through_traj,
            #         'fossil_count': fossil_count,
            #         'n_extant': len(te[te == 0]),
            #         'oldest_occ': np.max(tbl),
            #         'fadlad_tbl': tbl,
            #         'youngest_occ': np.max(tbl),
            #         'true_range_through_traj': true_range_through_traj,
            #         'avg_q': np.mean(q_rates),
            #         'q_rates': q_rates
            #     }
            if DEBUG: 
                print('fossil_count', res['fossil_count'])
                print('range_through_traj', res['range_through_traj'])
                print(res['range_through_traj'] - res['fossil_count'])
        
            if args.plot:
                # mid_points_temp = res['time_bins']
                mid_points_temp = mid_points[mid_points <= np.max(res['ts'])]
                Ntrue = res['true_range_through_traj'][:len(mid_points_temp)]
                Nrange_through = res['range_through_traj'][:len(mid_points_temp)]
                Nobs = res['n_extant']
                x = res['fossil_count']

                file_name = "%s/sim_data_range_%s_%s.pdf" % (args.outpath,sim_number,seed)
                fig = plt.figure(figsize=(12, 10))
                if DEBUG:
                    print("\n\n", mid_points_temp.shape,mid_points_temp[0:(len(Ntrue)+3)].shape, len(Ntrue))
                plt.plot(mid_points_temp[0:len(Ntrue)],Ntrue.T)
                plt.plot(mid_points_temp[0:len(Nrange_through)],Nrange_through.T)
            
                plt.plot(mid_points_temp[np.where(x > 0)], x[x > 0], 'ro')
                for i in range(len(mid_points_temp)):
                    plt.text( mid_points_temp[i] - (1 / 150 * np.max(res['ts'])),
                                    -0.035 * np.max(Ntrue), '%d' % (int(x[i])))
            
                plt.plot(np.zeros(int(Nobs)), np.arange(Nobs), 'ro')
            
                title = "n. extant species: %s   n. fossils: %s " % (int(Nobs), int(np.sum(x)))
                plt.gca().set_title(title,  fontsize=16)
                plt.xlabel('Time',  fontsize=14)
                plt.ylabel('N. species',  fontsize=14)
                plt.close()
                plot_divtraj = matplotlib.backends.backend_pdf.PdfPages(file_name)        
                plot_divtraj.savefig( fig )
                plot_divtraj.close()
                
                
                tbl = pd.DataFrame(res['fadlad_tbl'])
                tax_id = pd.DataFrame([["taxon_%s" % i, res['n_extant']] for i in range(tbl.shape[0])])
                tbl2 = pd.concat((tax_id, tbl), axis=1)
                tbl2.columns = ["taxon", "n_extant", "fad", "lad", "n_occs"]
                tbl2.to_csv(file_name.replace(".pdf", ".txt"), index=False, sep="\t")
                
        
        
        
            if np.min(res['range_through_traj'] - res['fossil_count']) < 0:
                sys.exit("\nIncompatible bbb_condition!\n")
        
            """
            run_mcmc(age_oldest_obs_occ, age_youngest_obs_occ, x, log_Nobs, Nobs, sim_n = 0, DAbatch=DAbatch)
            """
            true_root = np.max(res['ts'])
            true_ext = np.min(res['te'])
            true_q = res['avg_q']
            true_sig2 = 0
            indx_clade_life_span = np.array([i for i in range(len(mid_points)) if mid_points[i] < true_root])
        
            f = res['fossil_count']
            max_obs_ind = np.max(np.where(f > 0)[0]) + 1
            age_oldest_obs_occ = res['oldest_occ']
            age_youngest_obs_occ = res['youngest_occ']
            n_singleton_taxa = np.sum(res['fadlad_tbl'][:,0] == res['fadlad_tbl'][:,1])
            n_sampled_taxa = res['fadlad_tbl'].shape[0]
            n_true_taxa = len(res['ts'])
            n_fossils = np.sum(res['fossil_count'])
            
            print("true_root",true_root, "obs_root", age_oldest_obs_occ)
            print("true_ext",true_ext, "obs_ext", age_youngest_obs_occ)
            print("n. fossils", n_fossils)
            print("n. sampled taxa", n_sampled_taxa, "\nn. true taxa", n_true_taxa)
            print("n. singleton taxa", n_singleton_taxa)
        
            # z = np.zeros(max_obs_ind)
            # x = z + 0
            # x[:len(z)] = 0
            x = res['fossil_count'][:max_obs_ind]
            # print("res['fossil_count'][:max_obs_ind]", x, res['fossil_count'])
            # bbb_condition = z + 0
            # bbb_condition[:len(res['fossil_count'])] = res['range_through_traj']
            bbb_condition = res['range_through_traj'][:max_obs_ind]

            if DEBUG:
                print(len(x))
                print(len(bbb_condition))
                print('Nobs', res['n_extant'])
                print("x", x)
                print("bbb_condition", bbb_condition)
                print("Nobs", res['n_extant'])
                print("age_oldest_obs_occ", age_oldest_obs_occ)
                print("age_oldest_obs_occ", age_oldest_obs_occ)
        
            bbb_res=run_mcmc(age_oldest_obs_occ=age_oldest_obs_occ, 
                             age_youngest_obs_occ=age_youngest_obs_occ, 
                             x=x, # fossil data (fad/lad)
                             log_Nobs=np.log(np.max([1, res['n_extant']])), 
                             Nobs=res['n_extant'], 
                             sim_n=sim_number,
                             bbb_condition=bbb_condition)
            
            #--  simulations summary  --#
            save_summary = 1
            if sim_number == 1:
                out_name = "%s/summary_bbbrange.txt" % (args.outpath)
                logfile = open(out_name, "w") 
                text_str = "iteration\tNobs\tNfossils\tNfossil_species\tNsingleton\tNtrue_species"
                text_str = text_str + "\troot_true\troot_obs\text_true\text_obs\tq_med_true\tsig2_true"
                text_str = text_str + "\troot_est\troot_m\troot_M"
                text_str = text_str + "\text_est\text_m\text_M"
                text_str = text_str + "\tq_est\tsig2_est"
                logfile.writelines(text_str)
            
            if save_summary:
                # [lik_A, est_root_A, est_ext_A, est_q_A, est_sig2_A]
                post_burnin_res = bbb_res[ int(bbb_res.shape[0]*0.2):, : ]
                par_est = np.mean(post_burnin_res, axis=0)
                root_est_hpd = calcHPD(post_burnin_res[:,1]) # clade age
                ext_est_hpd = calcHPD(post_burnin_res[:,2]) # extinction age

                if save_summary:
                    text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % \
                    ( 
                        sim_number, Nobs, n_fossils, n_sampled_taxa, n_singleton_taxa, n_true_taxa,
                        true_root, age_oldest_obs_occ, true_ext, age_youngest_obs_occ,
                        true_q, true_sig2, 
                        par_est[1], root_est_hpd[0], root_est_hpd[1], 
                        par_est[2], ext_est_hpd[0], ext_est_hpd[1], 
                        par_est[3], par_est[4]
                    )
                    logfile.writelines(text_str)
                    logfile.flush()
            #---------------------------#
            
            sim_number +=1
            
            
            
            

    else:
        ### EMPIRICAL ANALYSES (standard BBB model)
        if args.fadlad_data is None:
            data_file = args.fossil_data
            fossil_data = np.loadtxt(data_file,skiprows=1)
            taxa_names = np.array(next(open(data_file)).split())
            print(max_age, fossil_data[0,0])
            mid_points = np.linspace(fossil_data[0,0],max_age,int(max_age/fossil_data[0,0]))
            bin_size = np.abs(np.diff(mid_points)[0])
    
            counts_file = args.div_table
            diversity_table = np.genfromtxt(counts_file,dtype='str',skip_header=1)
    
            taxa_list = np.intersect1d(taxa_names, diversity_table[:,0])
    
            if np.max(args.clades) > 0:
                taxa_list = taxa_list[args.clades[0]:(args.clades[1]+1)]
    
            if args.clade_name != "":
                taxa_list = np.array([i for i in taxa_list if args.clade_name in i])
    
            print("Found", len(taxa_list), "clades:")
            print(taxa_list[0:5], "...")
    
            for taxon in taxa_list:
                print("\nParsing data...", taxon)
    
                Nobs = int(diversity_table[diversity_table[:,0]==taxon,1])

                taxon_indx = np.where(taxa_names==taxon)[0][0]
                x= fossil_data[:,taxon_indx]
    
                x = get_fossil_count_occs(x)
                print("mid_points", mid_points)
                age_oldest_obs_occ = mid_points[len(x)-1]
                age_youngest_obs_occ = np.min(mid_points[np.where(x > 0)[0]])
                Nfoss  = int(np.sum(x))
                if Nobs:
                    log_Nobs = np.log(Nobs)
                else:
                    log_Nobs = 0
    
                x_0 = 1
                n_samples = len(mid_points)-len(x)
                x_augmented = 0+x
        
                print(Nobs, x_augmented)
                if Nobs < x_augmented[0]:
                    e = "Modern diversity is lower than sampled diversity at most recent time bin. Consider using smaller timebins."
                    sys.exit(e)
        
                if np.sum(x)< 1: 
                    print("No fossils:",np.sum(x),age_oldest_obs_occ, age_youngest_obs_occ)
                else:
                    print("N. fossils:",np.sum(x), "N. extant species:", Nobs,
                          "\nAge oldest occurrence:",age_oldest_obs_occ, 
                          "Age youngest occurrence:",age_youngest_obs_occ)
                    res=run_mcmc(age_oldest_obs_occ, age_youngest_obs_occ, x, log_Nobs, Nobs, taxon)
            
        #------
        # run range model
        else:
            print("Running BBB-range model", args.fadlad_data)
            from bd_sim import *
            from fossil_sim import *
            print("seed",seed)
            init_seed = seed + 0
            tbl = pd.read_csv(args.fadlad_data, sep="\t")

            #-- simulation parameters --#
            BIN_SIZE = 1
            mid_points = np.linspace(0,2*max_age,int(2*max_age/BIN_SIZE)+1)
            bin_size = np.abs(np.diff(mid_points)[0])
            
            range_through_traj = getDT_equalbin(mid_points, tbl["fad"].to_numpy(), tbl["lad"].to_numpy())
            fossil_count = get_fossil_count(mid_points, tbl["fad"].to_numpy(), tbl["lad"].to_numpy())
            # print("np.sum(res['fossil_count'])", np.sum(fossil_count), fossil_count)
            
            
            if tbl["n_extant"][0] > 0:
                age_youngest_obs_occ = 0
            else:
                age_youngest_obs_occ = np.min(tbl["lad"])
           
            print("age oldest occ:", np.max(tbl["fad"]))
            print("age youngest occ:", age_youngest_obs_occ)
            print("present diversity:", tbl['n_extant'][0])
            if DEBUG:
                print("fossils:\n", fossil_count)
                print("range_through_traj:\n", range_through_traj)
            
            max_obs_ind = np.max(np.where(fossil_count > 0)[0]) + 1
            x = fossil_count[:max_obs_ind]
            # print("fossil_count[:max_obs_ind]", x, np.sum(x))
            bbb_condition = range_through_traj[:max_obs_ind]
            
            
            input_file_raw = os.path.basename(args.fadlad_data)
            clade_name = os.path.splitext(input_file_raw)[0]  # file name without extension
    
            bbb_res=run_mcmc(age_oldest_obs_occ=np.max(tbl["fad"]), 
                             age_youngest_obs_occ=age_youngest_obs_occ, 
                             x=x, # fossil data (fad/lad)
                             log_Nobs=np.log(np.max([1, tbl['n_extant'][0]])), 
                             Nobs=tbl['n_extant'][0], 
                             bbb_condition=bbb_condition,
                             sim_n=clade_name
                             )
    
    
    


# examples
"python3 rootBBB.py -sim 1 -seed 5962 -p 10"
"python3 rangeBBB.py -sim_range 1 -sim 100 -q_min 0 -q_var 1 -seed 1234"
