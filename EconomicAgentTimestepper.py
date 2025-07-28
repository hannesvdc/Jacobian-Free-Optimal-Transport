import math
import numpy as np
import matplotlib.pyplot as plt

def event_timestepper(x, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N):
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

def evolveAgents(x0, k, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N):
    x = x0
    thres = 0.3
    print('t =', 0.0)
    for i in range(k):
        if x.mean() > thres:
            break
        x, vpc, vmc = event_timestepper(x, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N)
        print('t =', (i+1)*dt)
    return x

if __name__ == '__main__':
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
    x0 = np.random.normal(0,0.1,N)
    x0[x0 <= -1.0] = 0.0
    x0[x0 >=  1.0] = 0.0
    x = evolveAgents(x0, k, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N)

    # Plot the histogram
    plt.hist(x, bins=int(math.sqrt(N)), density=True, label=rf"$T =${T}")
    plt.xlabel('Agents')
    plt.legend()
    plt.show()