#!/usr/bin/env python
# coding: utf-8

# # Implementationer från 2. Brownsk rörelse
# Här är implementationer från sektion 2 i Higham om Brownsk rörelse. Koden är given som Matlab-kod och här översatt till Python.

# In[1]:


import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math as math
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[31]:


T = 1
N = 500
dt = T/N
t = np.linspace(dt, 1, N)

dW = math.sqrt(dt) * np.matlib.randn(2,N)
np.insert(dW, 0, 0)
np.insert(t, 0, 0)
W = np.cumsum(dW, axis=1) # Cumulative sum over the columns for each row

plt.plot(t, np.transpose(W)[:,], ls="--")
plt.xlabel("Diskretiserad tid")
plt.ylabel("W(t)")
# plt.savefig("Browian_motion.pdf")
plt.show()


# In[32]:


T = 1
N = 500
dt = T/N
t = np.linspace(dt, 1, N)

M = 1000
dW = math.sqrt(dt) * np.matlib.randn(M,N)
W = np.cumsum(dW, axis=1) # Cumulative sum over the columns for each row
U = np.exp(np.matlib.repmat(t, M, 1) + 0.5*W)
U_mean = np.mean(U, axis=0)

np.insert(t, 0, 0)
np.insert(U_mean, 0, 1)
np.insert(U, 0, 1, axis=1)

plt.plot(t, np.transpose(U)[:, 1:5], "r--", linewidth=1)
plt.plot(t, np.transpose(U_mean)) #Plot the average of M simulations
plt.xlabel("Diskretiserad tid")
plt.ylabel("$F(W(t))$",rotation=0, labelpad=10)
# plt.savefig("Brownian_function.pdf")

plt.show()


# # Implementation från 4. Euler-Maruyama

# In[30]:


lmbda = 2
mu = 1
X_zero = 1
T = 1
N = 2**8
dt = 1/N 
t1 = np.linspace(0, T, N) # Discretized time

dW = np.sqrt(dt) * np.matlib.randn(1,N) # Borwnian increments
W = np.cumsum(dW) # Discretized brownian path

X_true = X_zero * np.exp((lmbda - 0.5 * mu**2) * t1 + mu * W) # Exact solution for the SDE

np.insert(t1, 0, 0)
np.insert(X_true, 0, X_zero)
plt.plot(t1, np.transpose(X_true), "m", label="Exact solution")

## Euler-Maruyama ##
R = 2
Dt = R * dt # New time increment 
L = int(N/R)
t2 = np.linspace(0, T, L) # Discretized time
X_em = np.zeros((L,1)) # Allocation for speed
X_temp = X_zero
for j in range(0,L):
    W_inc = dW[0, R * (j-1): R * j].sum()
    X_temp = X_temp + Dt * lmbda * X_temp + mu * X_temp * W_inc
    X_em[j] = X_temp # Approximation for X(t_j)

plt.plot(t2, X_em, "g--o", label="Euler-Maruyama", mfc='none')

plt.legend(loc="best")
plt.show()


# # Implementation från sektion 5. Konvergens

# In[19]:


## Test av stark konvergens av EM ##

lmbda = 2
mu = 1
X_zero = 1
T = 1
N = 2**9
dt = T/N

M = 1000

X_err = np.zeros((5,M))
for s in range(0,M): #Each discrete Browninan path, lets us have info on pathwise level
    dW = np.sqrt(dt) * np.matlib.randn(1,N)
    W = np.cumsum(dW) # Diskret Brownsk väg
    X_true = X_zero * np.exp((lmbda - 0.5 * mu**2) + mu * W[0,-1])
    for p in range(1,6): #Each step size
        R = 2**(p-1)
        Dt = R*dt 
        L = int( N/R )# L Euler steg av längd Dt = R*dt
        X_temp = X_zero
        
        for j in range(1,L): #Each step in the path
            W_inc = dW[0, R * (j-1): R * j].sum()
            X_temp = X_temp + Dt * lmbda * X_temp + mu * X_temp * W_inc #EM
        X_err[p-1,s] = abs(X_temp - X_true) # Fel i slutpunkt

Dt_vals = dt * (2**np.linspace(0,4,5))


# In[27]:


## Test av svag konvergens av EM

lmbda = 2
mu = 1
X_zero = 1
T = 1

M = 500000

X_em = np.zeros((5,1)) # No pathwise info
for p in range(0,5): #Each step size
    Dt = 2**(p-10)
    L = int( T/Dt )
    X_temp = X_zero * np.ones((M,1))
    for j in range(1,L): #Each step in the path
        W_inc = np.sqrt(Dt) * np.matlib.randn(M,1)
        X_temp = X_temp + Dt * lmbda * X_temp + mu * np.multiply(X_temp,W_inc)
    X_em[p] = np.mean(X_temp)    
X_err_w = abs(X_em - np.exp(lmbda))

Dt_vals_w = 2**(np.linspace(1,5,5)-10)


# In[29]:


## Plot för analys av konvergens
plt.figure(figsize=(11,5))

plt.subplot(1,2,1)
plt.loglog(Dt_vals, np.mean(X_err, axis=1), "b*-", lw=2)
plt.loglog(Dt_vals, Dt_vals**(0.5), "r--", lw=2)

plt.xlim([1/1000, 1/10])
plt.ylim([1/1000, 10])
plt.xlabel("$\Delta t$")
plt.ylabel("$\mathbb{E}|X_N-X(T)|$", labelpad=0)
# plt.savefig("EM_strong.pdf")

plt.subplot(1,2,2)
plt.loglog(Dt_vals_w, X_err_w, "b*-", lw=2)
plt.loglog(Dt_vals_w, Dt_vals_w, "r--", lw=2)

plt.xlim([1/1000, 1/10])
plt.ylim([1/1000, 10])
plt.xlabel("$\Delta t$")
plt.ylabel("$|\mathbb{E}X_N-\mathbb{E}X(T)|$", labelpad=0)
# plt.savefig("EM_weak.pdf")
# plt.savefig("EM_conv.pdf")
plt.show()


# # Simulation från sista stycket i sektion 5

# In[33]:


lmbda = 2
mu = 1
X_zero = 1
T = 1
N = 2**8
dt = 1/N 
t1 = np.linspace(0, T, N) # Discretized time

frac_array = np.zeros(1000)
for k in range(0,1000):
    dW = np.sqrt(dt) * np.matlib.randn(1,N) # Borwnian increments
    W = np.cumsum(dW) # Discretized brownian path

    X_true = X_zero * np.exp((lmbda - 0.5 * mu**2) * t1 + mu * W) # Exact solution for the SDE

    np.insert(t1, 0, 0)
    np.insert(X_true, 0, X_zero)

    ## Euler-Maruyama ##
    R = 1
    Dt = R * dt # New time increment 
    L = int(N/R)
    t2 = np.linspace(0, T, L) # Discretized time
    X_em = np.zeros((L,1)) # Allocation for speed
    X_temp = X_zero
    for j in range(0,L):
        W_inc = dW[0, R * (j-1): R * j].sum()
        X_temp = X_temp + Dt * lmbda * X_temp + mu * X_temp * W_inc
        X_em[j] = X_temp # Approximation for X(t_j)

    X_err = abs(np.subtract(np.transpose(X_true), X_em))

    
    
    num = 0
    for i in range(X_err.size):
        if X_err[i] < Dt**(1/4):
            num = num + 1
    frac = num / X_err.size
    frac_array[k] = frac

print("För en given simulation beräknas andelen fel som är under tröskeln. Detta görs för 1000 vägar och ett medelvärde tas över alla kvoter.")    
print("Det resulterande medelvärdet är:")
print(np.mean(frac_array))

print("Nedan är ett histogram i vilket kvoterna för de 1000 vägarna sorterats")
plt.hist(frac_array, 50)

plt.show()


# In[ ]:




