import numpy as np
from collections import Counter
import copy
from tqdm import tqdm

def timeVariationMatrixBuilding(timestamp_matrix,mask_matrix):
    Delta_time=[]
    for k in tqdm(range(timestamp_matrix.shape[0])):
        all_=[]
        for j in range(timestamp_matrix.shape[1]):
            tempo=[]
            for p in range(timestamp_matrix.shape[2]):
                if j==0:
                    # at t=1 in the paper
                    tempo.append(0)
                else:
                    #If m^k_(t-1)=0 (in the paper)
                    if mask_matrix[k][j-1][p]==0:
                        tempo.append(timestamp_matrix[k][j][p]-timestamp_matrix[k][j-1][p]+all_[j-1][p])
                    else:
                        tempo.append(timestamp_matrix[k][j][p]-timestamp_matrix[k][j-1][p])
            all_.append(tempo)
        Delta_time.append(all_)
    Delta_time=np.array(Delta_time) 
    return Delta_time

def maskImputation(percentage,original_mask):
    imputer_mask=copy.deepcopy(original_mask)
    distribution_mask=Counter(original_mask.reshape(-1))
    print(f'Total number of true observations {distribution_mask[1.]}')
    percentage_drop=(distribution_mask[1.]*percentage)//100
    print(f'Number of observed values used for imputation/padding {percentage_drop}')
    
    
    
    imputer_mask=imputer_mask.reshape(-1)
    indexes=np.where(imputer_mask==1)
    indexes_drop=np.random.choice(indexes[0], size=percentage_drop)
    imputer_mask[indexes_drop]=0
    
    return imputer_mask.reshape(original_mask.shape[0],original_mask.shape[1],original_mask.shape[2])