import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import time

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.random.normal(loc = 0,scale = 1,size = 10000)
y = np.random.normal(loc = 0,scale = 1,size = 10000)

hist, xedges, yedges = np.histogram2d(x, y, bins=100, range=[[-5, 5], [-5, 5]])
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
np.random.rand(2,)
plt.show()'''

times = 0
create = 0
dimension = 1
S = 1

def create_sigma(dimension):
    #create a random covariance matrix
    distribution_sigma = np.random.rand(dimension,dimension)
    distribution_sigma = np.triu(distribution_sigma)
    distribution_sigma += np.transpose(distribution_sigma) - np.diag(distribution_sigma.diagonal())
    return distribution_sigma

while create == 0:
    distribution_sigma = create_sigma(dimension)
    try:
        np.linalg.inv(distribution_sigma)
        create = 1
        if np.linalg.det(distribution_sigma) <= 0:
            create = 0
        
    except:
        times += 1
        print(str(times) + ' times is used to create sigma')

standard_mu = np.zeros((dimension,))
standard_sigma = np.identity(dimension)
width = 2
sample_set = [100,250,1000,5000,50000]
distribution_sigma = S ** 2 * standard_sigma
distribution_mu = standard_mu

def gauss_pdf(z,mu,sigma):
    #calculate the value of probability density of variable z
    dim = z.shape[0]
    p = ((1 / (2 * np.pi)) ** (dim / 2)) * (1 / np.sqrt(np.linalg.det(sigma))) * np.exp(- 1 / 2 * np.dot(np.dot(np.transpose(z - mu),np.linalg.inv(sigma)),z - mu))
    return p

def one_d_gauss(z,mu,sigma):
    #it is used to calculate the margin probability of the joint distribution
    p = (1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(- 1 / 2 * (z - mu) ** 2 / sigma ** 2))
    return p


def accept(A):
    u = random.uniform(0,1)
    if u <= A:
        return 1
    else:
        return 0

def proposal_dis(z_old,method = 'uniform', width = 2):
    dim = z_old.shape[0]
    if method == 'uniform':
        z_new = z_old + 2 * width * np.random.rand(dim,) - width * np.ones((dim,))
        return z_new

def evaluate(sigma,mu,counts,bins):
    kl = 0
    i = 0
    delta_x = (bins[-1] - bins[0]) / (len(bins) - 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    for count in counts:
        x_bin = bin_centers[i]
        i += 1
        if count == 0:
            continue
        p_i = one_d_gauss(x_bin,mu,sigma)
        kl +=  count * np.log(count / p_i)
        
    
    kl =  delta_x * kl
    return kl


def M_chain_sample_gauss(sample_total,mu,sigma,width,burnin_rate):
    #sample main function
    
    #get the dimension 
    dim = mu.shape[0]

    #generate initial value
    z_0 = np.random.rand(dim,)
    z_set = [z_0]

    z_current = z_set[-1]

    #calculate burn-in counts
    burn_in_sample = int(burnin_rate * sample_total)

    with tqdm(total = burn_in_sample) as pbar:
        accept_point = 1
        total_point = 1
        pbar.set_description('Burn in period')
        pbar.update(1)

        while len(z_set) < burn_in_sample:
            total_point += 1
            z_new = proposal_dis(z_current,width = width)
            pz_current = gauss_pdf(z_current,mu,sigma)
            pz_new = gauss_pdf(z_new,mu,sigma)
            A = min(1,pz_new / pz_current)
            if accept(A) == 1:
                accept_point += 1
                z_current = z_new
                z_set.append(z_new)
                pbar.update(1)
            else:
                z_set.append(z_current)
                pbar.update(1)
            rate = round(accept_point / total_point * 100 , 2)
            pbar.set_postfix(Accept_rate = str(rate) + '%')

    z_set = [z_current]
    
    with tqdm(total = sample_total) as pbar:
        accept_point = 1
        total_point = 1
        pbar.set_description('Width = ' + str(width) + ',Burn in rate = ' + str(burnin_rate) + ',Processing')
        pbar.update(1)

        while len(z_set) < sample_total:
            total_point += 1
            z_new = proposal_dis(z_current,width = width)
            pz_current = gauss_pdf(z_current,mu,sigma)
            pz_new = gauss_pdf(z_new,mu,sigma)
            A = min(1,pz_new / pz_current)
            if accept(A) == 1:
                accept_point += 1
                z_current = z_new
                z_set.append(z_new)
                pbar.update(1)
            else:
                z_set.append(z_current)
                pbar.update(1)
            rate = round(accept_point / total_point , 2)
            show_rate = round(accept_point / total_point * 100 , 2)
            pbar.set_postfix(Accept_rate = str(show_rate) + '%')

    return np.array(z_set),rate

sample_total = 60000
width_list = [1]
burn_list = [0]

acc_rate_set = []
time_set = []
data_record = []

for burn_in_rate in burn_list:
    for width in width_list:
        #calculate the time
        start = time.time()

        #main function
        z_set,acc_rate = M_chain_sample_gauss(sample_total,distribution_mu,distribution_sigma,width,burn_in_rate)

        end = time.time()
        running_time = end - start
        
        KL_set = []
        #draw picture of margin probabilities
        for i in range(dimension):
            slice_sample = z_set[:,i]
            plt.figure()
            x = np.linspace(distribution_mu[i] - 4 * np.sqrt(distribution_sigma[i,i]),distribution_mu[i] + 4 * np.sqrt(distribution_sigma[i,i]),1000)
            y = one_d_gauss(x,distribution_mu[i],np.sqrt(distribution_sigma[i,i]))
            text_x = distribution_mu[i] - 4 * np.sqrt(distribution_sigma[i,i])
            text_y = max(y) * 1.0
            plt.text(text_x,text_y,'Burn_rate = ' + str(burn_in_rate) + ',Width = ' + str(width) + ',mu = ' + str(round(distribution_mu[i],2)) + ',sigma = ' + str(round(np.sqrt(distribution_sigma[i,i]),2)))
            plt.plot(x,y)
            plt.hist(slice_sample,bins = 50,density=True)
            plt.title('Standard normal distribution sampling of ' + str(sample_total)+ ' points')
            plt.xlabel('x')
            plt.ylabel('counts')
            plt.savefig('./p_u_3d/s_' + str(S) + '/MCMC_' +str(dimension) + 'd_v2_' + str(sample_total) +'_slice_' + str(i) +'_width_'+ str(width) + '_burn_' + str(burn_in_rate) + '.png')
            plt.close()
            counts,bins,patches = plt.hist(slice_sample,bins = 1000,density=True)
            KL_set.append(evaluate(np.sqrt(distribution_sigma[i,i]),distribution_mu[i],counts,bins))
        acc_rate_set.append(acc_rate)
        time_set.append(running_time)
        data_record.append(np.array([burn_in_rate,width,running_time,acc_rate,KL_set[0],KL_set[1],KL_set[2]]))

np.savetxt('./p_u_3d/s_' + str(S) + '/data.txt',data_record,fmt = '%.02f')






    