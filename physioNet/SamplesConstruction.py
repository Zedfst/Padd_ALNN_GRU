#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pandas as pd
import numpy as np
# from datetime import datetime
# from tqdm import tqdm
from Utils import *


# In[ ]:


#Get all file names
path = '../../data/physionet/set-a'
csv_files = glob.glob(path + "/*.txt")


# In[ ]:


#Outcomes
outcomes=pd.read_csv(path +'/Outcomes.csv')
outcomes.head()


# In[ ]:


#Load in csv format and concat all files
data = (pd.read_csv(file) for file in csv_files)
data   = pd.concat(data, ignore_index=True)
data.head()


# In[ ]:


#Itinial features
#['RecordID' 'Age' 'Gender' 'Height' 'ICUType'] are not include in the study
print(f'Features {data.Parameter.unique()}')


# In[ ]:


univariate_length=data.groupby(['Parameter','RecordID'])['Value'].count().values
univariate_length=int(np.max(univariate_length))
print(f'# of observations per univariate time series {univariate_length}')
no_missing_values=len(data.loc[data.Value==-1,:])
print(f'# of missing values {no_missing_values}')


# In[ ]:


#Create mask that determines whether a value is missing.
data['Mask']=data.Value.apply(lambda x: 1 if x>=0 else 0)
data.head(2)


# In[ ]:


univariate_length,list_lengths=getMaxLength(data)
print(f'# Number of obsevation per univaraite {univariate_length}')


# In[ ]:


average_length=int(np.max(list_lengths))
print(f'Average length per univaraite {average_length}')


# In[ ]:


#Samples construction
#target_name define the name of the downstream task.
#See outcomes dataframe above. 
VALUES,TIMESTAMPS,MASKS,PADDINGS,OUTCOMES=samplesConstructorPhysionet(
    data,outcomes,univariate_length,average_length,target_name='In-hospital_death',mean_length=False)


# In[ ]:


path='data/'
#Save the samples
np.save(path+'/Values',VALUES)
np.save(path+'/Timestamps',TIMESTAMPS)
np.save(path+'/Masks',MASKS)
np.save(path+'/Padds',PADDINGS)
np.save(path+'/Targets',OUTCOMES)

