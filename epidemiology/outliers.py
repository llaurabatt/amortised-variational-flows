#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%

data_path = '/home/llaurabat/variationalmetaposterior/examples/epidemiology_new/data/' 

df = pd.read_csv(data_path + 'epidemiology_plummer.csv')

# %%
fig, ax = plt.subplots(2,2, figsize=(10,7))

ax[0,0].bar(np.arange(df['nhpv'].shape[0]), df['Npop']*np.exp(df['nhpv']/df['Npart']))
ax[0,0].set_title(fr'T_i * exp(Z_i/N_i)')

ax1_2 = ax[1,0].twinx()
ax[1,0].bar(np.arange(df['nhpv'].shape[0]), np.exp(df['nhpv']/df['Npart']))
ax[1,0].set_title(fr'exp(Z_i/N_i)')

ax1_2.plot(df['Npop'], color='red', label='T_i')
ax1_2.legend()

ax[0,1].bar(np.arange(df['nhpv'].shape[0]), df['nhpv']/df['Npart'])
ax[0,1].set_title(fr'Z_i/N_i')

ax[1,1].bar(np.arange(df['ncases'].shape[0]), df['ncases']/df['Npop'])
ax[1,1].set_title(fr'Y_i/T_i')
plt.savefig(data_path + 'outliers.png')

plt.show()
