import torch 
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_data(n = -1):
    # Load Diagnostics.xlsx
    diagnostics = pd.read_excel("data/Diagnostics.xlsx")
    
    # Load Conditions.xlsx
    conditions = pd.read_excel("data/RhythmNames.xlsx")
    n_cond = conditions["Acronym Name"].size
    cond_map = { conditions["Acronym Name"][i].strip() : i for i in range(n_cond) }
    
    # add SA
    cond_map['SA'] = n_cond 
    n_cond += 1
    
    # Load Examples from files.
    num_examples = diagnostics.shape[0] if (n == -1) else n
    
    X = np.zeros((5000, num_examples, 12))
    Y = np.zeros((num_examples, n_cond))

    print("Loading", num_examples, "Examples.")
    for i, file in tqdm(enumerate((Path.cwd() / "data/ECGDataDenoised/").glob("*.csv")), total = num_examples-1):
        
        # get name of file.
        fname = str(file.name).split('.')[0]
        
        # get row number in Diagnostics.xlsx for this file name.
        j = diagnostics.index[diagnostics["FileName"] == fname].item()
        
        # make one-hot vector for condition names.
        for b in diagnostics["Rhythm"][j].split(' '):
            if b != "NONE":
                k = cond_map[b]
                Y[i,k] = 1
        
#         data = pd.read_csv(file, header=None).to_numpy()
#         if data.shape[0] < 5000:
#             continue 
            
        X[:, i, :] = pd.read_csv(file, header=None).to_numpy()
        
        if i >= num_examples - 1:
            break
#     print("Done! Loaded {0} examples.".format())

    Y = torch.from_numpy(Y).type(torch.FloatTensor)
    X = torch.from_numpy(X).type(torch.FloatTensor)
    
    return X, Y

def plot_ecg(y, n_leads = 12):

    fig, axs = plt.subplots(n_leads, figsize=(8, 2))
    plt.xlabel("Seconds")
    plt.ylabel("Microvolts")
    plt.xlim([0, 10])
    
    x_len = y.shape[0]
    x = np.linspace(0, 10, x_len)
    
    if n_leads == 1:
        sig = axs.plot(x, y[:,0].numpy(), color='black')
    else:
        for i in range(n_leads):
            print(y.shape)
            axs[i].plot(x, y[:,i].numpy(), color='black')
    
    plt.show()
    
    return plt, axs, x, y[0,:].numpy()