#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import copy
import tensorflow as tf
from sklearn.metrics import average_precision_score,roc_curve,auc
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from utils import *
from model import PaddALNNGRU
import time as tm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,roc_curve,auc,f1_score,confusion_matrix


# In[ ]:


#Load inputs
data_set='physioNet'
path=f'{data_set}/data'

VALUES=np.load(path+'/Values.npy')
TIMESTAMPS=np.load(path+'/Timestamps.npy')
MASKS=np.load(path+'/Masks.npy')
PADDINGS=np.load(path+'/Padds.npy')
OUTCOMES=np.load(path+'/Targets.npy')


# In[ ]:


#Build the Delta time matrix
Delta_Time=timeVariationMatrixBuilding(TIMESTAMPS,MASKS)
print(Delta_Time.shape)


# In[ ]:


print(f'Shape of matrix values {VALUES.shape}')
print(f'Shape of matrix timestamps {TIMESTAMPS.shape}')
print(f'Shape of matrix mask {MASKS.shape}')
print(f'Shape of matrix mask {Delta_Time.shape}')
print(f'Shape of matrix labels {OUTCOMES.shape}')


# In[ ]:


tf.random.set_seed(1234)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=2, mode='min')
kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=14)
loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
aucs,aucpr=[],[]

#For mimic-3 set as follows: epochs=40; batch_size=500
#For physioNet set as follows: epochs=30; batch_size=200

epochs=30
batch_size=200


for idx,(train_index, test_index) in enumerate(kfold.split(VALUES,OUTCOMES)):
    paddalnngru=PaddALNNGRU(no_features=VALUES.shape[2],max_timestamp=48,is_timestamps_generated=True,
                            loss_regularizer=0.5,alnn_dropout_rate=0.1,
                            padd_gru_units=50,padd_gru_dropout=0.7,is_imputation=True)

    paddalnngru.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    TRAINING_SAMPLES=[VALUES[train_index],TIMESTAMPS[train_index],MASKS[train_index],Delta_Time[train_index],PADDINGS[train_index]]
    start= tm.perf_counter()
    paddalnngru.fit(TRAINING_SAMPLES,
             OUTCOMES[train_index],
             epochs=epochs,
             callbacks=[callback],
             batch_size=batch_size,
             verbose=1)
    finish=tm.perf_counter()
    print(f"Training time {round(finish-start,2)},second(s)")


    TESTING_SETS=[VALUES[test_index],TIMESTAMPS[test_index],MASKS[test_index],Delta_Time[test_index],PADDINGS[test_index]]

    start= tm.perf_counter()
    loss_test, accuracy_test = paddalnngru.evaluate(TESTING_SETS,OUTCOMES[test_index],verbose=1)
    finish=tm.perf_counter()
    print(f"Testing time {round(finish-start,2)},second(s)")
    y_probas = paddalnngru.predict(TESTING_SETS).ravel()
    print(y_probas.shape)
    print(OUTCOMES[test_index].shape)
    fpr,tpr,thresholds=roc_curve(OUTCOMES[test_index],y_probas)
    aucs.append(auc(fpr,tpr))
    print('AUC->',auc(fpr,tpr))
    auprc_ = average_precision_score(OUTCOMES[test_index], y_probas)
    aucpr.append(auprc_)
    print('AUPRC->', auprc_)
    print('\n')

print(f'AUC: mean{np.round(np.mean(np.array(aucs)),3)},std{np.round(np.std(np.array(aucs)),3)}')
print(f'AUPRC: mean{np.round(np.mean(np.array(aucpr)),3)},std{np.round(np.std(np.array(aucpr)),3)}')
print('\n')


# In[ ]:




