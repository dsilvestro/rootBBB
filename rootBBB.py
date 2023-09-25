from datetime import datetime
import sys
import argparse
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
import scipy.stats
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

p = argparse.ArgumentParser()
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
p.add_argument('-biased_q', type=int,   help='if 1 set increasing q through time', default = 0)
p.add_argument('-freq_q0',  type=float, help='frequency of 0-sampling rate', default = 0.1)
p.add_argument('-q_var',    type=int,   help='0) constant q 1) linearly increasing q', default = 0)
p.add_argument('-q_exp',    type=int,   help='magnitude of variation', default = 1)
p.add_argument('-clades',   type=int,   help='range of clade to be analyzed', default = [0,-1], nargs=2)
p.add_argument('-outpath',  type=str,   help='path output', default = ".")
p.add_argument('-clade_name', type=str,   help='', default = "")
p.add_argument('-f',        type=float, help='freq. DA runs with updates', default = 0.95)
p.add_argument('-out',      type=str,   help='add string to output', default = "")
p.add_argument('-nDA',      type=int,   help='DA samples', default = 1000)
p.add_argument('-DAbatch',  type=int,   help='DA batch size (if set to 0: auto-tune)', default = 0)
p.add_argument('-ws',       type=float, help='win sizes root, sig2, q', default = [10,1.25,1.25], nargs=3)
p.add_argument('-max_age',  type=int,   help='Max boundary of uniform prior on the root age', default = 300)
p.add_argument('-q_prior',  type=float,   help='shape and rate (default: 1.1, 1)', default = [1.1, 1], nargs=2)
p.add_argument('-a_prior',  type=float,   help='shape and rate (default: 1, 0.01)', default = [1, 0.01], nargs=2)
p.add_argument('-sig_prior',  type=float,   help='shape and rate (default: 1, 0.1)', default = [1, 0.1], nargs=2)
p.add_argument('-q_min',    type=float,   help='offset for q', default = 0)



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
[alpha_a, beta_a] = args.a_prior
[alpha_sig, beta_sig] = args.sig_prior


q_offset = args.q_min

# simulation settings
increasing_q_rates = args.biased_q
freq_zero_preservation = args.freq_q0
bias_exp = args.q_exp

simulate_extinct = args.sim_extinct


run_simulations = np.min([1,n_simulations])

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

def get_fossil_count(data):
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
    
    x_augmented = np.zeros(len(mid_points_shift))
    x_augmented[0:len(fossil_data)] = fossil_data+0    
    
    x_augmented = x_augmented[mid_points_shift > est_ext]
    mid_points_shift = mid_points_shift[mid_points_shift > est_ext]
    
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
    
def get_avg_likelihood(Nobs, fossil_data, est_root, est_sig2, est_q, est_a, x_augmented, simTraj_all):
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
        # print(x_augmented)
        # quit()
        
        sampling_prob_vec = np.ones(len(x_augmented))- np.exp(-q_vec)
        # print(np.log(sampling_prob_vec))
        lik_matrix = binomial_pmf(x_augmented,simTraj_all,sampling_prob_vec)
        # lik_avg = np.log(np.mean(np.exp( np.sum(lik_matrix,axis=1) )))
        # to avoid underflow:
        lik_i = np.sum(lik_matrix,axis=1)
        lik_max = np.max(lik_i)
        lik_avg = np.log(np.sum(np.exp( lik_i-lik_max ))) - np.log(simTraj_all.shape[0]) + lik_max
        #print( lik_avg,lik_max , len(lik_i[np.exp(lik_i)>0]), len(lik_i), n_samples*counter, counter )
        #print(lik_i)
        #quit()
        return lik_avg, len(lik_i[np.exp(lik_i-lik_max)>0])

def run_mcmc(age_oldest_obs_occ, age_youngest_obs_occ, x, log_Nobs, Nobs, sim_n = 0, DAbatch=DAbatch):
    lik_A = np.nan
    tries = 0
    # initialize MCMC
    print("\n\nInitializing the model...")
    while np.isnan(lik_A):
        est_a_A = 0.
        if alpha_a > 1:
            est_a_A = small_number
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
            x_augmented_A, simTraj_all_A = get_imputations(Nobs, x, est_root_A, est_ext_A, np.exp(est_sig2_A), 
                                                           n_samples=n_DA_samples,
                                                           DAbatch=DAbatch)
            if len(x_augmented_A)==0:
                # ie imputation didn't work
                lik_A = np.nan
            else:
                lik_A, DA_counts = get_avg_likelihood(Nobs, x, est_root_A, np.exp(est_sig2_A), est_q_A, est_a_A, x_augmented_A, simTraj_all_A)
                #prior_A = gamma_pdf(est_sig2_A,a=1.,b=1.)
                prior_A = gamma_pdf(np.exp(est_sig2_A-log_Nobs),a=alpha_sig,b=beta_sig) + gamma_pdf(est_q_A,a=alpha_q,b=beta_q) + gamma_pdf(est_a_A,a=alpha_a, b=beta_a) 
        elif tries <= 200:
            # print("Attempt 2...")
            # init root age
            #est_root_A = np.random.random()
            est_root_A = np.min([age_oldest_obs_occ*(1+np.random.uniform(0.05,0.25 )), max_age])
            # init sig2
            est_sig2_A = np.log(np.random.uniform(0, 5)*np.max([1, Nobs]))
            # init q_rate
            est_q_A = np.random.uniform(0.0005, 0.002)
            x_augmented_A, simTraj_all_A = get_imputations(Nobs, x, est_root_A, est_ext_A, np.exp(est_sig2_A), 
                                                           n_samples=n_DA_samples,
                                                           DAbatch=DAbatch)
                                                            
            if len(x_augmented_A)==0:
                # ie imputation didn't work
                lik_A = np.nan
            else:
                lik_A, DA_counts = get_avg_likelihood(Nobs, x, est_root_A, np.exp(est_sig2_A), est_q_A, est_a_A, x_augmented_A, simTraj_all_A)
                #prior_A = gamma_pdf(est_sig2_A,a=1.,b=1.)
                prior_A = gamma_pdf(np.exp(est_sig2_A-log_Nobs),a=alpha_sig,b=beta_sig) + gamma_pdf(est_q_A,a=alpha_q,b=beta_q) + gamma_pdf(est_a_A,a=alpha_a, b=beta_a) 
        elif tries <= 10000:
            # print("Attempt 3...")
            est_root_A =  np.min([age_oldest_obs_occ*(1+np.random.uniform(0.05,1 )), max_age])
            # init sig2
            est_sig2_A = np.log(np.random.uniform(0.1, 100)*np.max([1, Nobs]))
            # init q_rate
            est_q_A = np.random.uniform(0.0005, 0.1)

            x_augmented_A, simTraj_all_A = get_imputations(Nobs, x, est_root_A, est_ext_A, np.exp(est_sig2_A),
                                                           n_samples=n_DA_samples,
                                                           DAbatch=DAbatch)
            
            if len(x_augmented_A)==0:
                # ie imputation didn't work
                lik_A = np.nan
            else:
                lik_A, DA_counts = get_avg_likelihood(Nobs, x, est_root_A, np.exp(est_sig2_A), est_q_A, est_a_A, x_augmented_A, simTraj_all_A)
                #prior_A = gamma_pdf(est_sig2_A,a=1.,b=1.)
                prior_A = gamma_pdf(np.exp(est_sig2_A-log_Nobs),a=alpha_sig,b=beta_sig) + gamma_pdf(est_q_A,a=alpha_q,b=beta_q) + gamma_pdf(est_a_A,a=alpha_a, b=beta_a) 
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
            
        if run_simulations:
            out_name = "%s/mcmc_%s_%s_f%s%s%s.log" % (args.outpath, sim_n, seed, freq_par_updates, args.out, model_out)
            logfile = open(out_name, "w") 
            text_str = "iteration\tposterior\tlikelihood\tprior\tNobs\tNfossils\troot_obs\text_obs\troot_true\text_true\tq_med_true\tsig2_true\tDA_counts\troot_est\text_est\tq_est\ta_est\tsig2_est"
            logfile.writelines(text_str)
        else:
            out_name = "%s/%s_mcmc_%s_f%s%s%s.log" % (args.outpath, sim_n, seed, freq_par_updates, args.out, model_out)
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
        x_augmented, simTraj_all = x_augmented_A+0, simTraj_all_A+0
        h1,h2,h3 = 0,0,0
        accept = 0
        
        rr = np.random.random(3)
        if rr[0]< freq_par_updates: 
            update = 1
            accept = 0
            
        else: 
            x_augmented, simTraj_all = get_imputations(Nobs, x, est_root, est_ext, np.exp(est_sig2), 
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
            x_augmented, simTraj_all = get_imputations(Nobs, x, est_root, est_ext, np.exp(est_sig2), 
                                                       n_samples=n_DA_samples,
                                                       DAbatch=DAbatch)
            
            accept = 0
        
        
        
        lik, DA_counts = get_avg_likelihood(Nobs, x, est_root, np.exp(est_sig2), est_q, est_a, x_augmented, simTraj_all )
        
        prior = gamma_pdf(np.exp(est_sig2-log_Nobs),a=alpha_sig,b=beta_sig) + gamma_pdf(est_q,a=alpha_q,b=beta_q) + gamma_pdf(est_a,a=alpha_a, b=beta_a) 
    
        if (lik-lik_A) + (prior-prior_A) + (h1+h2+h3) >= np.log(np.random.random()) or accept==1 and np.isfinite(lik):
            est_root_A = est_root
            est_ext_A = est_ext
            est_sig2_A = est_sig2
            est_q_A    = est_q
            est_a_A    = est_a
            lik_A      = lik
            prior_A    = prior
            x_augmented_A, simTraj_all_A = x_augmented, simTraj_all 
            accepted = 1
    
        if iteration % args.p == 0 and verbose:
            if iteration == 0:
                print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("it", "lik", "root", "ext", "sig2", "q_rate", "q_slope"))
            print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (iteration, round(lik_A,2), round(est_root_A,2), round(est_ext_A,2), round(est_sig2_A,2), round(est_q_A,5), round(est_a_A,5)))
        
        if iteration % args.s == 0:
            if save_mcmc_samples:
                if run_simulations:
                    text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % \
                    ( iteration, lik_A+prior_A, lik_A, prior_A, Nobs, np.sum(x), age_oldest_obs_occ, age_youngest_obs_occ, true_root, true_ext, \
                    np.median(true_q), \
                    #true_sig2, DA_counts, age_oldest_obs_occ*(1+est_root_A), est_q_A, est_sig2_A)
                    true_sig2, DA_counts, est_root_A, est_ext_A, est_q_A, est_a_A, est_sig2_A)
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
    
if run_simulations:
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
            res=run_mcmc(age_oldest_obs_occ, age_youngest_obs_occ, x, log_Nobs, Nobs, sim_number)

            post_burnin_res = res[ int(res.shape[0]*0.2):, : ]

            par_est = np.mean(post_burnin_res, axis=0)
            root_est_hpd = calcHPD(post_burnin_res[:,1])

            if verbose:
                print(par_est, root_est_hpd)

            if save_summary:
                text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % \
                ( sim_number, Nobs, np.sum(x), true_root, age_oldest_obs_occ, np.median(true_q), \
                true_sig2, par_est[1], root_est_hpd[0], root_est_hpd[1], par_est[2], par_est[3])
                logfile.writelines(text_str)
                logfile.flush()
            
            sim_number += 1
        # quit()

else:
    ### EMPIRICAL ANALYSES
    data_file = args.fossil_data
    fossil_data = np.loadtxt(data_file,skiprows=1)
    taxa_names = np.array(next(open(data_file)).split())
    print(max_age, fossil_data[0,0])
    mid_points = np.linspace(fossil_data[0,0],max_age,int(max_age/fossil_data[0,0]))
    bin_size = np.abs(np.diff(mid_points)[0])
    
    counts_file = args.div_table
    diversity_table = np.genfromtxt(counts_file,dtype='str',skip_header=1)
    
    taxa_list = np.intersect1d(taxa_names, diversity_table[:,0])
    
    if np.min(args.clades) >= 0:
        taxa_list = taxa_list[args.clades[0]:(args.clades[1]+1)]
    
    if args.clade_name != "":
        taxa_list = np.array([i for i in taxa_list if args.clade_name in i])
    
    if len(taxa_list)==1: 
        print("Found 1 clade:")
    else: 
        print("Found", len(taxa_list), "clades:")
    print(taxa_list[0:5], "...")
    
    for taxon in taxa_list:
        print("\nParsing data...", taxon)
    
        Nobs = int(diversity_table[diversity_table[:,0]==taxon,1])

        taxon_indx = np.where(taxa_names==taxon)[0][0]
        x= fossil_data[:,taxon_indx]
    
        x = get_fossil_count(x)
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
            
        
"python3 rootBBB.py -sim 1 -seed 5962 -p 10"