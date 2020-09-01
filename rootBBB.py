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
p.add_argument('-biased_q', type=int,   help='if 1 set increasing q through time', default = 0)
p.add_argument('-freq_q0',  type=float, help='frequency of 0-sampling rate', default = 0.1)
p.add_argument('-clades',   type=int,   help='range of clade to be analyzed', default = [0,0], nargs=2)
p.add_argument('-outpath',  type=str,   help='path output', default = ".")
p.add_argument('-clade_name', type=str,   help='', default = "")
p.add_argument('-f',        type=float, help='freq. DA runs with updates', default = 0.95)
p.add_argument('-out',      type=str,   help='add string to output', default = "")
p.add_argument('-nDA',      type=int,   help='DA samples', default = 1000)
p.add_argument('-DAbatch',  type=int,   help='DA batch size', default = 1000)
p.add_argument('-ws',       type=float, help='win sizes root, sig2, q', default = [10,1.25,1.25], nargs=3)


print("""

Root age estimator using a Bayesian Brownian bridge.

""")


args = p.parse_args()

if args.seed== -1:
	seed =np.random.randint(1000,9999)
else:
	seed =args.seed
np.random.seed(seed)


n_simulations = args.sim
save_mcmc_samples = 1
mid_points = np.linspace(0,500,int(500/2.5)+1)
n_DA_samples = args.nDA
sim_loglinear = 0
freq_par_updates = args.f
verbose = args.verbose
DAbatch = args.DAbatch

# simulation settings
increasing_q_rates = args.biased_q
freq_zero_preservation = args.freq_q0



run_simulations = np.min([1,n_simulations])


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
	#	return pmf
	#else: 
	#	return approx_log_binomiam_pmf(x,n,p)

def normal_pdf(x,n,p):
	return scipy.stats.norm.logpdf(x, n, p, loc=0)

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
	
def get_imputations(log_Nobs, Nobs, fossil_data, est_root, est_sig2, est_q, n_samples=1000):
	time_bins = mid_points[mid_points<est_root]
	simTraj_all = np.zeros((n_samples,len(time_bins)))
	j=0
	counter = 0
	while j <= n_samples:
	#if 2>1:
		# simulate expected trajectories
		if sim_loglinear:
			simTraj = sample_path_batch_discrete(time_bins, n_reps =DAbatch, sig2=est_sig2, y_start = log_Nobs)
			x_augmented = np.zeros(simTraj.shape[1])
			x_augmented[0:len(fossil_data)] = fossil_data+0	
			m = np.min(np.exp(simTraj)-x_augmented,axis=1)
		else:
			simTraj = sample_path_batch_discrete(time_bins, n_reps =DAbatch, sig2=est_sig2, y_start = Nobs)
			simTraj = np.rint(simTraj)+1
			x_augmented = np.zeros(simTraj.shape[1])
			x_augmented[0:len(fossil_data)] = fossil_data+0	
			m = np.min(simTraj-x_augmented,axis=1)
        		
		#remove incompatible trajectories
		simTraj = simTraj[m>=0]
		#remove a priori incompatible trajectories
		m = np.min(simTraj,axis=1)
		if sim_loglinear:
			simTraj = simTraj[m>=0]
		else:
			simTraj = simTraj[m>=1]
		valid_samples = simTraj.shape[0]
		simTraj_all[j:np.min([n_samples, j+valid_samples]), :] = simTraj[0:np.min([valid_samples, n_samples-j]), :]
		
		j += valid_samples
		counter +=1
		if counter > 100:
			#print("reach max limit, found:",j)
			return [], 0
	return x_augmented, simTraj_all
	
def get_avg_likelihood(log_Nobs, Nobs, fossil_data, est_root, est_sig2, est_q, x_augmented, simTraj_all):
	if len(x_augmented)==0:
		return -np.inf, 0
	else:
		sampling_prob_vec = np.ones(len(x_augmented))- np.exp(-est_q)
		if sim_loglinear:
			lik_matrix = binomial_pmf(x_augmented,np.exp(simTraj_all),sampling_prob_vec)
		else:
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

def run_mcmc(age_oldest_obs_occ, x, log_Nobs, Nobs, sim_n = 0):
	lik_A = np.nan
	tries = 0
	while np.isnan(lik_A):
		if tries <= 100:
			# init root age
			#est_root_A =  np.random.random()
			est_root_A =  age_oldest_obs_occ*(1+np.random.uniform(0.05,0.25 ))
			# init sig2
			if sim_loglinear:
				est_sig2_A = np.random.uniform(0.01, 0.2)
			else:
				est_sig2_A = np.log(np.random.uniform(10, 50)*Nobs)
			# init q_rate
			est_q_A = np.random.uniform(0.0005, 0.002)
			x_augmented_A, simTraj_all_A = get_imputations(log_Nobs, Nobs, x, est_root_A, est_sig2_A, est_q_A, n_samples=n_DA_samples)
			if len(x_augmented_A)==0:
				# ie imputation didn't work
				lik_A = np.nan
			else:
				lik_A, DA_counts =          get_avg_likelihood(log_Nobs, Nobs, x, est_root_A, est_sig2_A, est_q_A, x_augmented_A, simTraj_all_A)
				#prior_A = gamma_pdf(est_sig2_A,a=1.,b=1.)
				prior_A = gamma_pdf(np.exp(est_sig2_A-log_Nobs),a=1.,b=.1) + gamma_pdf(est_q_A,a=1.1,b=1)
		elif tries <= 1000:
			est_root_A =  age_oldest_obs_occ*(1+np.random.uniform(0.05,1 ))
			# init sig2
			if sim_loglinear:
				est_sig2_A = np.random.uniform(0.01, 0.2)
			else:
				est_sig2_A = np.log(np.random.uniform(1, 100)*Nobs)
			# init q_rate
			est_q_A = np.random.uniform(0.0005, 0.002)

			x_augmented_A, simTraj_all_A = get_imputations(log_Nobs, Nobs, x, est_root_A, est_sig2_A, est_q_A, n_samples=n_DA_samples)
			if len(x_augmented_A)==0:
				# ie imputation didn't work
				lik_A = np.nan
			else:
				lik_A, DA_counts =          get_avg_likelihood(log_Nobs, Nobs, x, est_root_A, est_sig2_A, est_q_A, x_augmented_A, simTraj_all_A)
				#prior_A = gamma_pdf(est_sig2_A,a=1.,b=1.)
				prior_A = gamma_pdf(np.exp(est_sig2_A-log_Nobs),a=1.,b=.1) + gamma_pdf(est_q_A,a=1.1,b=1)
		else:
			sys.exit("Failed to initialize model.")
		tries+=1
		
	print("Running MCMC...")
	if save_mcmc_samples:
		if run_simulations:
			out_name = "%s/mcmc_%s_%s_f%s%s.log" % (args.outpath, sim_n, seed, freq_par_updates, args.out)
			logfile = open(out_name, "w") 
			text_str = "iteration\tposterior\tlikelihood\tprior\tNobs\tNfossils\troot_obs\troot_true\tq_med_true\tsig2_true\tDA_counts\troot_est\tq_est\tsig2_est"
			logfile.writelines(text_str)
		else:
			out_name = "%s/%s_mcmc_%s_f%s%s.log" % (args.outpath, sim_n, seed, freq_par_updates, args.out)
			print(out_name)
			logfile = open(out_name, "w") 
			text_str = "iteration\tposterior\tlikelihood\tprior\tNobs\tNfossils\troot_obs\tDA_counts\troot_est\tq_est\tsig2_est"
			logfile.writelines(text_str)
			
	res = np.zeros((int(args.n/args.s), 4))
	if verbose: print(res.shape)
	sampled_iteration=0
	
	for iteration in range(args.n):
		accepted = 0
		#start = datetime.now()
		est_root,est_sig2,est_q = est_root_A+0,est_sig2_A+0,est_q_A+0
		x_augmented, simTraj_all = x_augmented_A+0, simTraj_all_A+0
		h1,h2,h3 = 0,0,0
		accept = 0
		
		rr =np.random.random(3)
		if rr[0]< freq_par_updates: 
			update = 1
			accept = 0
			
		else: 
			x_augmented, simTraj_all = get_imputations(log_Nobs, Nobs, x, est_root, np.exp(est_sig2), est_q, n_samples=n_DA_samples)
			update = 0
			accept = 1
		
		if update:
			if rr[2]< 0.7:
				est_root, h1 = update_normal(est_root_A , m=age_oldest_obs_occ, M=300, d=args.ws[0])
			if rr[2]> 0.5:
				if sim_loglinear:
					est_sig2, h2 = update_multiplier(est_sig2_A ,d=1.05)
				else:
					est_sig2EXP, h2 = update_multiplier(np.exp(est_sig2_A),args.ws[1])
					est_sig2 = np.log(est_sig2EXP)
					est_q   , h3 = update_multiplier(est_q_A ,d=args.ws[2])
			x_augmented, simTraj_all = get_imputations(log_Nobs, Nobs, x, est_root, np.exp(est_sig2), est_q, n_samples=n_DA_samples)
			accept = 0
		
		
		
		if sim_loglinear:
			lik, DA_counts = get_avg_likelihood(log_Nobs, Nobs, x, est_root, est_sig2, est_q, x_augmented, simTraj_all )
		else:
			lik, DA_counts = get_avg_likelihood(log_Nobs, Nobs, x, est_root, np.exp(est_sig2), est_q, x_augmented, simTraj_all )
		
		prior = gamma_pdf(np.exp(est_sig2-log_Nobs),a=1.,b=0.1) + gamma_pdf(est_q,a=1.1,b=1) 
	
		if (lik-lik_A) + (prior-prior_A) + (h1+h2+h3) >= np.log(np.random.random()) or accept==1 and np.isfinite(lik):
			est_root_A = est_root
			est_sig2_A = est_sig2
			est_q_A    = est_q
			lik_A      = lik
			prior_A    = prior
			x_augmented_A, simTraj_all_A = x_augmented, simTraj_all 
			accepted = 1
	
		if iteration % args.p == 0 and verbose:
			if iteration == 0:
				print("%s\t%s\t%s\t%s\t%s" % ("it", "lik", "root", "sig2", "q_rate"))
			print("%s\t%s\t%s\t%s\t%s" % (iteration, round(lik_A,2), round(est_root_A,2), round(est_sig2_A,2), round(est_q_A,5)))
		
		if iteration % args.s == 0:
			if save_mcmc_samples:
				if run_simulations:
					text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % \
					( iteration, lik_A+prior_A, lik_A, prior_A, Nobs, np.sum(x), age_oldest_obs_occ, true_root, np.median(true_q), \
					#true_sig2, DA_counts, age_oldest_obs_occ*(1+est_root_A), est_q_A, est_sig2_A)
					true_sig2, DA_counts, est_root_A, est_q_A, est_sig2_A)
				else:
					text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % \
					( iteration, lik_A+prior_A, lik_A, prior_A, Nobs, np.sum(x), age_oldest_obs_occ,DA_counts, est_root_A, est_q_A, est_sig2_A)
				logfile.writelines(text_str)
				logfile.flush()
			
			res[sampled_iteration,:] = np.array([lik_A, est_root_A, est_q_A, est_sig2_A])
			sampled_iteration += 1
	return res

def simulate_data(rseed=0):
	if rseed > 0:
		np.random.seed(rseed)
		
	true_root = np.random.uniform(10,180)
	log_q_mean = -8.52 # 1/5000 mean Nfossil 
	log_q_std = 1

	logNobs = np.random.uniform(np.log(100),np.log(20000)) # np.log(50000.) 
	Nobs = np.rint(np.exp(logNobs))
	log_Nobs = np.log(Nobs)
	if sim_loglinear:
		true_sig2 = np.random.uniform(0.01,0.25) #0.1
		logNtrue = sample_path_batch_discrete(time_bins = mid_points[mid_points<true_root], n_reps =100, sig2=true_sig2, y_start = log_Nobs,positive=0)
		m = np.min(logNtrue,axis=1)
		logNtrue = logNtrue[m>=0][0]
		Ntrue = np.rint(np.exp(logNtrue))
	else:
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
		true_q = np.random.choice(true_q,len(true_q),replace=0,p=true_q/np.sum(true_q))

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

if run_simulations:
	save_summary = 1
	if save_summary:
		out_name = "%s/summary.txt" % (args.outpath)
		logfile = open(out_name, "w") 
		text_str = "iteration\tNobs\tNfossils\troot_true\troot_obs\tq_med_true\tsig2_true\troot_est\troot_M\troot_m\tq_est\tsig2_est"
		logfile.writelines(text_str)

	print("seed",seed)

	sim_number = 1
	counter = 0
	while sim_number <= n_simulations:
		counter += 1
		print("simulating data...")
		true_root, true_q, true_sig2, Nobs, age_oldest_obs_occ, x, log_Nobs,Ntrue = simulate_data(seed+sim_number)
	
		if np.sum(x)< 1: 
			print("No fossils:",np.sum(x),"",age_oldest_obs_occ)
			seed_s = "%s%s" % (counter, seed) # change seed if it doesn't work
			seed = int(seed_s)
		else:
			if args.plot:
				file_name = "%s/sim_data%s_%s.pdf" % (args.outpath,sim_number,seed)
				fig = plt.figure(figsize=(12, 10))
				plt.plot(mid_points[mid_points<true_root],Ntrue.T)
				mid_points_temp = mid_points[mid_points<true_root]
				print(len(x), len(mid_points_temp), len(mid_points))
				
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
			res=run_mcmc(age_oldest_obs_occ, x, log_Nobs, Nobs, sim_number)

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

else:
	### EMPIRICAL ANALYSES
	data_file = args.fossil_data
	fossil_data = np.loadtxt(data_file,skiprows=1)
	taxa_names = np.array(next(open(data_file)).split())
	
	max_age = 300
	mid_points = np.linspace(fossil_data[0,0],max_age,int(max_age/fossil_data[0,0]))
	
	counts_file = args.div_table
	diversity_table = np.genfromtxt(counts_file,dtype='str',skip_header=1)
	
	taxa_list = np.intersect1d(taxa_names, diversity_table[:,0])
	
	if np.max(args.clades) > 0:
		taxa_list = taxa_list[args.clades[0]:(args.clades[1]+1)]
	
	if args.clade_name != "":
		taxa_list = np.array([i for i in taxa_list if args.clade_name in i])
	
	print("Analyzing", len(taxa_list), "clades:")
	print(taxa_list)
	
	for taxon in taxa_list:
		print("parsing data...", taxon)
	
		Nobs = int(diversity_table[diversity_table[:,0]==taxon,1])

		taxon_indx = np.where(taxa_names==taxon)[0][0]
		x= fossil_data[:,taxon_indx]
	
		x = get_fossil_count(x)
		age_oldest_obs_occ = mid_points[len(x)-1]
		Nfoss  = int(np.sum(x))
		log_Nobs = np.log(Nobs)
	
		x_0 = 1
		n_samples = len(mid_points)-len(x)
		x_augmented = 0+x

		if np.sum(x)< 1: 
			print("No fossils:",np.sum(x),age_oldest_obs_occ)
		else:
			print("N. fossils:",np.sum(x),age_oldest_obs_occ)
			res=run_mcmc(age_oldest_obs_occ, x, log_Nobs, Nobs, taxon)
		
