import numpy as np
from tqdm import tqdm

def getMaxLength(df,view_list_parameters=False):
    max_length=0
    list_lengths=[]
    list_recordids=df.RecordID.unique()
    list_parametrs=df.Parameter.unique()
    excluded_parameters=['RecordID','Age','Gender','Height','ICUType']
    list_parametrs=np.setdiff1d(list_parametrs, excluded_parameters)
    
    if view_list_parameters:
        print(list_parametrs)

    VALUES,TIMESTAMPS,MASKS,PADDINGS,OUTCOMES,OUTCOMES_2,PATIENT_ID=[],[],[],[],[],[],[]
    for recordid in tqdm(list_recordids):
        sub_df=df.loc[df.RecordID==recordid,:].sort_values(by='Time') 
        tempo_values,tempo_timestamps,tempo_masks,tempo_paddings=[],[],[],[]
        for parameter in list_parametrs:
            sub_sub_df=sub_df.loc[sub_df.Parameter==parameter,:]
            timestamps=sub_sub_df.Time.values
            timestamps=np.round(np.array(list(map(lambda x: int(x.split(':')[0])+(int(x.split(':')[1])/60),timestamps))),1)
            unique_timestamps=np.unique(timestamps)
            if len(unique_timestamps)>max_length:
                max_length=len(unique_timestamps)
            list_lengths.append(len(unique_timestamps))
    return max_length,list_lengths


def samplesConstructorPhysionet(df,outcomes,univariate_length,average_length,target_name,view_list_parameters=False,mean_length=False):
    list_recordids=df.RecordID.unique()
    no_samples=df.RecordID.nunique()
    list_parametrs=df.Parameter.unique()
    excluded_parameters=['RecordID','Age','Gender','Height','ICUType']
    list_parametrs=np.setdiff1d(list_parametrs, excluded_parameters)
    
    if view_list_parameters:
        print(f'List of parameters: {list_parametrs}')

    VALUES,TIMESTAMPS,MASKS,PADDINGS,OUTCOMES,OUTCOMES_2,PATIENT_ID=[],[],[],[],[],[],[]
    OUTCOMES=np.zeros((no_samples,1))
    
    if mean_length:
        VALUES=np.zeros((no_samples,average_length,len(list_parametrs)))
        TIMESTAMPS=np.zeros((no_samples,average_length,len(list_parametrs)))
        MASKS=np.zeros((no_samples,average_length,len(list_parametrs)))
        PADDINGS=np.zeros((no_samples,average_length,len(list_parametrs)))
    else:
        VALUES=np.zeros((no_samples,univariate_length,len(list_parametrs)))
        TIMESTAMPS=np.zeros((no_samples,univariate_length,len(list_parametrs)))
        MASKS=np.zeros((no_samples,univariate_length,len(list_parametrs)))
        PADDINGS=np.zeros((no_samples,univariate_length,len(list_parametrs)))
        average_length=univariate_length
        
    for idx,recordid in tqdm(enumerate(list_recordids),total=len(list_recordids)):
        sub_df=df.loc[df.RecordID==recordid,:].sort_values(by='Time') 
        tempo_values,tempo_timestamps,tempo_masks,tempo_paddings=[],[],[],[]
        for idx2,parameter in enumerate(list_parametrs):
            sub_sub_df=sub_df.loc[sub_df.Parameter==parameter,:]
            values=sub_sub_df.Value.values

            timestamps=sub_sub_df.Time.values
            masks=sub_sub_df.Mask.values
            timestamps=np.round(np.array(list(map(lambda x: int(x.split(':')[0])+(int(x.split(':')[1])/60),timestamps))),2)
            
            values_candidate,masks_candidate,padds_candidate,timestamps_candidate=[],[],[],[]

            #If there are no timestamp values available
            if len(timestamps)==0:
                values_candidate.append(0)
                padds_candidate.append(1)
                masks_candidate.append(0)
                timestamps_candidate.append(0)
            else:
                unique_timestamps=np.unique(timestamps)
                for u_t in unique_timestamps:
                    index_timestamps=np.where(timestamps==u_t)
                    value_candidate=values[index_timestamps]
                    value_candidate=value_candidate[value_candidate!=-1]
                    padd_candidate=np.ones_like(value_candidate)
                    if len(value_candidate)==0:
                        value_candidate=0
                        mask_candidate=0
                    else:
                        value_candidate=np.mean(value_candidate)
                        mask_candidate=1
                    timestamps_candidate.append(u_t)

                    values_candidate.append(value_candidate)
                    padds_candidate=np.ones_like(values_candidate)
                    masks_candidate.append(mask_candidate)
            
            #Padding if the univariate has less observations than exepected.
            #This padded values are not considered in the calculation.
            #They are just use to have inputs with uniform shape.
            if len(values_candidate)<univariate_length:
                values_candidate=np.pad(values_candidate,(0,univariate_length-len(values_candidate)),'edge')
                masks_candidate=np.pad(masks_candidate,(0,univariate_length-len(masks_candidate)),'edge')
                padds_candidate=np.pad(padds_candidate,(0,univariate_length-len(padds_candidate)),'constant',constant_values=0)
                timestamps_candidate=np.pad(timestamps_candidate,(0,univariate_length-len(timestamps_candidate)),'edge')
            else:
                values_candidate=values_candidate[:univariate_length]
                masks_candidate=masks_candidate[:univariate_length]
                padds_candidate=padds_candidate[:univariate_length]
                timestamps_candidate=timestamps_candidate[:univariate_length]
                
            VALUES[idx,:,idx2]=values_candidate[:average_length]
            TIMESTAMPS[idx,:,idx2]=timestamps_candidate[:average_length]
            MASKS[idx,:,idx2]=masks_candidate[:average_length]
            PADDINGS[idx,:,idx2]=padds_candidate[:average_length]
            
        outcome=outcomes.loc[outcomes.RecordID==recordid,:][target_name].values
        OUTCOMES[idx,:]=outcome

    return VALUES,TIMESTAMPS,MASKS,PADDINGS,OUTCOMES