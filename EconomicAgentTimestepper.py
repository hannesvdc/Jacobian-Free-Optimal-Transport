import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

def event_timestepper_numpy(x : np.ndarray, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N):
    buyers = 0
    sellers = 0
    num_step = 4
    dt_small = dt/num_step
    egdt = np.exp(-gamma*dt_small)
    egdt2 = (1-egdt)/(gamma*dt_small)
    for _ in range(num_step):
        Good_news=np.random.poisson(dt_small*vpc,N)
        Bad_news=np.random.poisson(dt_small*vmc,N)
        Infplus=eplus*Good_news
        Infminus=eminus*Bad_news
        Inftotal=Infplus+Infminus
        x_new=x*egdt+Inftotal*egdt2
        Flag = ((x + Infplus) >= 1).astype(int) + ((x + Infminus) <= -1).astype(int) #can it be written better?
        indFlag=np.where(Flag>0)
        lindFlag=np.size(indFlag)
        for iia in range(lindFlag): #can it be written better?
            ia=indFlag[0][iia] 
            #Needs special handling, could have gone outside  (-1,1)
            Gnia=Good_news[ia]
            Bnia=Bad_news[ia]
            timeb = np.random.uniform(0,dt_small,Gnia)       #Times of buys
            times = np.random.uniform(0,dt_small,Bnia)      # and sells
            type_bs = np.append(np.ones(Gnia), -np.ones(Bnia))
            timebs=np.append(timeb, times)
            idx = np.argsort(timebs)
            timebs = np.sort(timebs)     #Sorted times
            typebs_sort = type_bs[idx]             #directions
            lindx=np.size(idx)
            timebsdiff = timebs - np.append(np.zeros(1), timebs[range(lindx-1)])  #Time differences
            x_temp = x[ia]
            for jja in range(lindx):
                if (typebs_sort[jja] > 0):           #Positive news
                    x_temp = x_temp*np.exp(-gamma*(timebsdiff[jja])) + eplus
                    if (x_temp >= 1) :
                        x_temp = 0
                        buyers = buyers + 1
                else:                        #negative news
                    x_temp = x_temp*np.exp(-gamma*(timebsdiff[jja])) + eminus
                    if (x_temp <= -1):
                        x_temp = 0
                        sellers = sellers + 1
            x_new[ia] = x_temp*np.exp(-gamma*(dt_small - timebs[lindx-1]))
        x = x_new
        # print('First number is', np.size(Ibuy), 'second number is', np.size(Isell))
        # input()
    buyrate = buyers / dt / N
    sellrate = sellers / dt / N
    
    vpc = vplus + g * buyrate
    vmc = vminus + g * sellrate
    return x, vpc, vmc

def event_timestepper_torch(x : torch.Tensor, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N):
    buyers = 0
    sellers = 0
    num_step = 4
    dt_small = dt / num_step
    egdt = math.exp(-gamma * dt_small)
    egdt2 = (1 - egdt) / (gamma * dt_small)

    for _ in range(num_step):
        Good_news = torch.poisson(torch.full((N,), dt_small * vpc)).to(torch.int64)
        Bad_news = torch.poisson(torch.full((N,), dt_small * vmc)).to(torch.int64)
        Infplus = eplus * Good_news
        Infminus = eminus * Bad_news
        Inftotal = Infplus + Infminus
        x_new = x * egdt + Inftotal * egdt2

        Flag = ((x + Infplus) >= 1).int() + ((x + Infminus) <= -1).int()
        indFlag = torch.nonzero(Flag > 0, as_tuple=True)
        lindFlag = indFlag[0].numel()

        for iia in range(lindFlag):
            ia = indFlag[0][iia].item()
            Gnia = Good_news[ia].item()
            Bnia = Bad_news[ia].item()

            timeb = torch.rand(size=(Gnia,)) * dt_small if Gnia > 0 else torch.tensor([])
            times = torch.rand(size=(Bnia,)) * dt_small if Bnia > 0 else torch.tensor([])
            type_bs = torch.cat((torch.ones(Gnia), -torch.ones(Bnia)))
            timebs = torch.cat((timeb, times))

            idx = torch.argsort(timebs)
            timebs = timebs[idx]
            typebs_sort = type_bs[idx]
            lindx = idx.numel()

            timebsdiff = timebs - torch.cat((torch.tensor([0.0]), timebs[:-1]))
            x_temp = x[ia].item()
            for jja in range(lindx):
                if typebs_sort[jja] > 0:
                    x_temp = x_temp * torch.exp(-gamma * timebsdiff[jja]) + eplus
                    if x_temp >= 1:
                        x_temp = 0
                        buyers += 1
                else:
                    x_temp = x_temp * torch.exp(-gamma * timebsdiff[jja]) + eminus
                    if x_temp <= -1:
                        x_temp = 0
                        sellers += 1
            x_new[ia] = x_temp * torch.exp(-gamma * (dt_small - timebs[-1]))

        x = x_new

    buyrate = buyers / dt / N
    sellrate = sellers / dt / N
    vpc = vplus + g * buyrate
    vmc = vminus + g * sellrate
    return x, vpc, vmc

def evolveAgentsNumpy(x0, k, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N):
    x = x0
    thres = 0.3
    for i in range(k):
        if x.mean() > thres:
            break
        x, vpc, vmc = event_timestepper_numpy(x, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N)
        print('t =', (i+1)*dt)
    return x

def evolveAgentsTorch(x0 : torch.Tensor, k, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N, verbose=False):
    x = torch.clone(x0)
    thres = 0.3
    for i in range(k):
        if x.mean() > thres:
            break
        x, vpc, vmc = event_timestepper_torch(x, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N,)
        if verbose:
            print('t =', (i+1)*dt)
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        dest='type',
        help="Numpy or Torch"
    )
    args = parser.parse_args()

    # Simulation parameters
    N = 50000
    eplus = 0.075
    eminus = -0.072
    vplus = 20
    vminus = 20
    gamma = 1
    g = 38.0
    vpc = vplus
    vmc = vminus

    # Time evolution up to T = 100s
    T = 100.0
    dt = 0.25
    k = int(T / dt)

    # Normal initial distribution
    if args.type == 'numpy':
        x0 = np.random.normal(0,0.1,N)
        x0[x0 <= -1.0] = 0.0
        x0[x0 >=  1.0] = 0.0
        x = evolveAgentsNumpy(x0, k, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N)
    else:
        x0 = torch.normal(0.0, 0.1, (N,))
        x0[x0 <= -1.0] = 0.0
        x0[x0 >=  1.0] = 0.0
        x = evolveAgentsTorch(x0, k, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N)
        x = x.numpy()

    # Plot the histogram
    plt.hist(x, bins=int(math.sqrt(N)), density=True, label=rf"$T =${T}")
    plt.xlabel('Agents')
    plt.legend()
    plt.show()