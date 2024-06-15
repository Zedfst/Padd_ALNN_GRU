import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

class ALNNLayer(tf.keras.layers.Layer):

    def __init__(self,init_time,time_interval_reference_time_point,max_timestamp,
                 no_reference_time_point,alnn_dropout_rate,type_distance):
        

        super(ALNNLayer, self).__init__()
        self.init_time = init_time
        self.time_interval_reference_time_point = time_interval_reference_time_point
        self.max_timestamp=max_timestamp
        self.no_reference_time_point=no_reference_time_point
        self.type_distance=type_distance
        self.alnn_dropout_rate=alnn_dropout_rate
        
        
                

#         if((self.prior_hours%self.time_space)!=0):
#             raise Exception(f'{self.time_space}  must be a multiple of {self.prior_hours}.')

        #Reference time points vector
        self.ref_time_point_vector=np.linspace(self.init_time,self.max_timestamp,self.no_reference_time_point)
        self.ref_time_point_vector=self.ref_time_point_vector.reshape(self.no_reference_time_point,1,1)

        self.dropout_1=layers.Dropout(self.alnn_dropout_rate)
        self.dropout_2=layers.Dropout(self.alnn_dropout_rate)
        self.normalize=layers.Normalization()


    def build(self, input_shape):

        self.axis_2=input_shape[0][1]
        self.axis_3=input_shape[0][2]

        self.alpha = self.add_weight(shape=(self.no_reference_time_point,1,1),
                                 initializer='glorot_uniform',
                                 name='alpha',
                                 dtype='float32',
                                 trainable=True)

        self.w_v = self.add_weight(shape=(self.no_reference_time_point,self.axis_2,input_shape[0][2]),
                                 initializer='glorot_uniform',
                                 name='w_inyensity',
                                 dtype='float32',
                                 trainable=True)

        self.w_t = self.add_weight(shape=(self.no_reference_time_point,self.axis_2,input_shape[0][2],5),
                                 initializer='glorot_uniform',
                                 name='w_tempo',
                                 dtype='float32',
                                 trainable=True)

        self.b_v= self.add_weight(shape=(self.no_reference_time_point,1,input_shape[0][2]),
                                 initializer='glorot_uniform',
                                 name='bias_intensity',
                                 dtype='float32',
                                 trainable=True)

        self.b_t = self.add_weight(shape=(self.no_reference_time_point,self.axis_2, self.axis_3,1),
                                 initializer='glorot_uniform',
                                 name='bias_tempo',
                                 dtype='float32',
                                 trainable=True)

    def call(self, inputs,training=None):
        self.X=inputs[0]#values
        self.T=inputs[1]#timestamps
        self.M=inputs[2]#masks
        self.DT=inputs[3]#delta times
        self.P=inputs[4]#delta times


        #Dupliction with respect to the number of reference time points
        self.x=tf.tile(self.X[:,None,:,:],[1,self.no_reference_time_point,1,1])
        self.t=tf.tile(self.T[:,None,:,:],[1,self.no_reference_time_point,1,1])
        self.m=tf.tile(self.M[:,None,:,:],[1,self.no_reference_time_point,1,1])
        self.dt=tf.tile(self.DT[:,None,:,:],[1,self.no_reference_time_point,1,1])
        self.p=tf.tile(self.P[:,None,:,:],[1,self.no_reference_time_point,1,1])
        

        if(self.type_distance=="abs"):
            self.diastance=tf.abs(self.t-tf.cast(self.ref_time_point_vector,tf.float32))
        else:
            self.diastance=tf.square(self.t-tf.cast(self.ref_time_point_vector,tf.float32))

        self.kernel=tf.exp(-tf.cast(tf.nn.relu(self.alpha),tf.float32)*self.diastance)
        #time lag intensity
        self.intensity=tf.nn.relu(self.x*self.kernel)


        self.x_s=tf.reshape(self.x,[-1,self.no_reference_time_point,self.axis_2, self.axis_3,1])
        self.dt=tf.reshape(self.dt,[-1,self.no_reference_time_point,self.axis_2, self.axis_3,1])
        self.intensity_s=tf.reshape(self.intensity,[-1,self.no_reference_time_point,self.axis_2, self.axis_3,1])
        self.m_s=tf.reshape(self.m,[-1,self.no_reference_time_point,self.axis_2, self.axis_3,1])
        self.p=tf.reshape(self.p,[-1,self.no_reference_time_point,self.axis_2, self.axis_3,1])

        
        if training:
            #Value-level extraction
            self.lattent_x=self.dropout_1(tf.nn.relu(tf.reduce_sum(self.w_t*tf.concat([self.x_s,self.intensity_s,self.m_s,self.dt,self.p],4)+self.b_t,4)),training=training)
            #Feature-level aggregation
            self.lattent_x=self.dropout_2(tf.nn.relu(tf.reduce_sum(self.w_v*self.lattent_x + self.b_v,2)),training=training)
        else:
            #Value-level extraction
            self.lattent_x=tf.nn.relu(tf.reduce_sum(self.w_t*tf.concat([self.x_s,self.intensity_s,self.m_s,self.dt,self.p],4)+self.b_t,4))
            #Feature-level aggregation
            self.lattent_x=tf.nn.relu(tf.reduce_sum(self.w_v*self.lattent_x + self.b_v,2))


        return self.lattent_x #pseudo-aligned latent values

    def get_config(self):
        config = super(ALNNLayer, self).get_config()
        config.update({"init_time": self.init_time})
        config.update({"prior_hours": self.prior_hours})
        config.update({"time_space": self.time_space})
        onfig.update({"type_distance": self.type_distance})
        onfig.update({"alnn_dropout_rate": self.alnn_dropout_rate})
        return config
        

class PaddGRULayer(keras.layers.Layer):
    def __init__(self,max_timestamp,padd_gru_units,
                 no_features,padd_gru_dropout=0.0,
                 is_timestamps_generated=True,**kwargs):
        
        self.padd_gru_units=padd_gru_units
        self.padd_gru_dropout=padd_gru_dropout
        self.no_features=no_features
        self.is_timestamps_generated=is_timestamps_generated
        self.max_timestamp=max_timestamp
        
    
        
        self.state_size = [tf.TensorShape([self.padd_gru_units]),tf.TensorShape([self.no_features]),tf.TensorShape([self.no_features])]
        self.dropout=layers.Dropout(self.padd_gru_dropout)
        super(PaddGRULayer,self).__init__(**kwargs)

    def build(self, input_shapes):
        self.w_regression = self.add_weight(
            shape=(self.padd_gru_units,self.no_features), initializer="uniform", name="w_regression",trainable=True
        )
        self.b_regression = self.add_weight(
            shape=(1,), initializer="zero", name="b_regression",trainable=True
        )
        
        self.w_controleur = self.add_weight(
            shape=(self.padd_gru_units,self.no_features), 
            initializer="uniform", 
            name="w_controleur",trainable=True
        )
        self.b_controleur = self.add_weight(
            shape=(self.no_features,),
            initializer="zero", name="b_controleur",trainable=True
        )
        
        self.w_delta = self.add_weight(
            shape=(self.no_features,self.padd_gru_units), initializer="glorot_uniform", name="w_delta",trainable=True
        )
        self.b_delta = self.add_weight(
            shape=(1,), initializer="zero", name="b_delta",trainable=True
        )
        
        self.w_value_mask = self.add_weight(
            shape=(self.no_features*2, self.padd_gru_units), initializer="glorot_uniform", name="w_value_mask",trainable=True
        )
        self.b_value_mask = self.add_weight(
            shape=(1,), initializer="zero", name="b_value_mask",trainable=True
        )
        
        self.w_corr = self.add_weight(
            shape=(self.no_features, self.no_features), initializer="glorot_uniform", name="weights_correlation",trainable=True
        )
        self.b_corr = self.add_weight(
            shape=(self.no_features, ), initializer="zero", name="bias_correlation",trainable=True
        )
        
        self.w_concat = self.add_weight(
            shape=(self.no_features+self.padd_gru_units,self.no_features), initializer="glorot_uniform", name="w_concat",trainable=True
        )
        self.b_concat = self.add_weight(
            shape=(1,), initializer="zero", name="b_concat",trainable=True
        )
        self.u_update = self.add_weight(
            shape=(self.padd_gru_units, self.padd_gru_units), initializer="glorot_uniform", name="u_update",trainable=True
        )
        
        self.w_update = self.add_weight(
            shape=(self.no_features*2, self.padd_gru_units), initializer="glorot_uniform", name="w_update",trainable=True
        )
        self.b_update = self.add_weight(
            shape=(1,), initializer="zero", name="b_update",trainable=True
        )
        self.u_reset = self.add_weight(
            shape=(self.padd_gru_units, self.padd_gru_units), initializer="glorot_uniform", name="u_reset",trainable=True
        )
        
        self.w_reset = self.add_weight(
            shape=(self.no_features*2, self.padd_gru_units), initializer="glorot_uniform", name="w_reset",trainable=True
        )
        self.b_reset = self.add_weight(
            shape=(1,), initializer="zero", name="b_reset",trainable=True
        )

    def call(self,inputs,states,training=None):
        
        h_t_1=states[0]
        h_t_1=self.dropout(h_t_1,training=training)
        prvious_timestamps=states[1]
        prvious_delta=states[2]
        
        time_intervals=inputs[:,:self.no_features]
        x_current=inputs[:,self.no_features:self.no_features*2]
        imputer_mask=inputs[:,self.no_features*2:self.no_features*3]
        delta_time=inputs[:,self.no_features*3:self.no_features*4]
        padding=inputs[:,self.no_features*4:self.no_features*5]


        x_regressor=tf.matmul(h_t_1,self.w_regression)+self.b_regression
        if self.is_timestamps_generated:
            self.time_variation=tf.abs(h_t_1@self.w_controleur+self.b_controleur)
            time_intervals=padding*time_intervals+ (1.-padding)*(prvious_timestamps+self.time_variation)
            time_intervals=tf.where(tf.less_equal(time_intervals,self.max_timestamp),time_intervals,prvious_timestamps)
            delta_generated=tf.where(tf.less_equal(prvious_timestamps+self.time_variation,self.max_timestamp),self.time_variation,prvious_delta)
            
            delta_time=padding*delta_time+ (1.-padding)*(delta_generated)
            

        x_current=imputer_mask*x_current + (1.-imputer_mask)*x_regressor
        decay=tf.exp(-tf.nn.relu(delta_time@self.w_delta+self.b_delta))
        z_corr=tf.matmul(x_current,(1.-tf.eye(x_current.shape[1]))*self.w_corr)+self.b_corr
        beta_t=tf.nn.sigmoid(tf.matmul(tf.concat([imputer_mask,decay],1),self.w_concat)+self.b_concat)
        c_t=beta_t*z_corr+(1.-beta_t)*x_regressor
        c=imputer_mask*x_current+(1.-imputer_mask)*c_t
        
        z_update=tf.nn.sigmoid(tf.matmul(decay*h_t_1,self.u_update) + tf.matmul(tf.concat([c,imputer_mask],1),self.w_update)+self.b_update)
        r_reset=tf.nn.sigmoid(tf.matmul(decay*h_t_1,self.u_reset) + tf.matmul(tf.concat([c,imputer_mask],1),self.w_reset)+self.b_reset)
        h_hat=tf.nn.tanh(tf.matmul(r_reset*decay*h_t_1,self.u_update) + tf.matmul(tf.concat([c,imputer_mask],1),self.w_value_mask)+self.b_value_mask)
        h_t=(1.-z_update)*h_t_1 + z_update*h_hat
        

        
            
        if self.is_timestamps_generated:
            return [x_regressor,z_corr,c,time_intervals,delta_time],[h_t,time_intervals,delta_time]
        else:
            return [x_regression,z_corr,c],[h_t,time_intervals,delta_time]



    def get_config(self):
        config = super(PaddGRULayer, self).get_config()
        return config
    
class ALNN(keras.Model):
    
    def __init__(self,init_time,max_timestamp,
                 time_interval_reference_time_point,
                 alnn_dropout_rate,type_distance):
        
      
        
        super(ALNN, self).__init__()
        self.init_time=init_time
        self.max_timestamp=max_timestamp
        self.alnn_dropout_rate=alnn_dropout_rate
        self.type_distance=type_distance
        self.time_interval_reference_time_point=time_interval_reference_time_point
        self.no_reference_time_point=self.max_timestamp*self.time_interval_reference_time_point+1
        
        
        
        self.alnnlayer=ALNNLayer(self.init_time,
                       self.time_interval_reference_time_point,
                       self.max_timestamp,
                       self.no_reference_time_point,
                       self.alnn_dropout_rate,
                       self.type_distance)

    def call(self, inputs,training=None):
        self.values=tf.cast(inputs[0],tf.float32)
        self.timestamps=tf.cast(inputs[1],tf.float32)
        self.masks=tf.cast(inputs[2],tf.float32)
        self.timevariations=tf.cast(inputs[3],tf.float32)
        self.paddings=tf.cast(inputs[4],tf.float32)
        
        self.pseudo_aligned=self.alnnlayer([self.values,self.timestamps,self.masks,self.timevariations,self.paddings])
                      
        
        return self.pseudo_aligned
    
    def get_config(self):
        config = super(ALNN, self).get_config()
        config.update({"max_time": self.max_time})
        config.update({"init_time": self.init_time})
        config.update({"time_interval": self.time_interval})
        config.update({"type_of_distance": self.type_distance})
        config.update({"gru_unit": self.gru_unit})
        config.update({"gru_dropout": self.gru_dropout})
        config.update({"pseudo_latent_dropout": self.pseudo_latent_dropout})
        return config
    
class PaddALNNGRU(keras.Model):
    
    def __init__(self,          
                 no_features,
                 alnn_dropout_rate=0.1,
                 time_interval_reference_time_point=1,
                 type_distance="abs",
                 init_time=0,
                 loss_regularizer=1.,
                 max_timestamp=48.,
                 is_timestamps_generated=True,
                 padd_gru_units=50,
                 padd_gru_dropout=0.0,
                 is_imputation=True,
                 is_alignment=True,
                 gru_units=168,
                 gru_dropout=0.0,
                 ):
        
        
        super(PaddALNNGRU, self).__init__()
        self.padd_gru_units=padd_gru_units
        self.no_features=no_features
        self.padd_gru_dropout=padd_gru_dropout
        self.is_timestamps_generated=is_timestamps_generated
        self.is_imputation=is_imputation
        self.loss_regularizer=loss_regularizer
        self.is_alignment=is_alignment
        self.init_time=init_time
        self.max_timestamp=max_timestamp
        self.alnn_dropout_rate=alnn_dropout_rate
        self.time_interval_reference_time_point=time_interval_reference_time_point
        self.type_distance=type_distance
        self.gru_units=gru_units
        self.gru_dropout=gru_dropout
        
        
        self.paddgrulayer=PaddGRULayer(self.max_timestamp,self.padd_gru_units,
                                  self.no_features,
                                  self.padd_gru_dropout,
                                  self.is_timestamps_generated)
        self.paddgru=layers.RNN(self.paddgrulayer,return_sequences=True)
        self.alnn=ALNN(self.init_time,self.max_timestamp,
                       self.time_interval_reference_time_point,
                       self.alnn_dropout_rate,self.type_distance)
                       
        self.gru=layers.GRU(self.gru_units,dropout=gru_dropout,return_sequences=False)
        self.dense1=layers.Dense(100)
        self.drop1=layers.Dropout(0.2)
        self.classifier=layers.Dense(1,activation="sigmoid",name="classifier")
        
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()


        
    def call(self, inputs,training=None):
        values=tf.cast(inputs[0],tf.float32)#tensor of values
        time_stamps=tf.cast(inputs[1],tf.float32)#tensor of timestamps
        masks=tf.cast(inputs[2],tf.float32)# tensor of masks
        delta=tf.cast(inputs[3],tf.float32)# tensor of time variation
        paddings=tf.cast(inputs[4],tf.float32)# tensor of padding indicator
        
        
        #Data-driven imputation with padd-gru
        if self.is_imputation:
            feaures_embedded=self.paddgru(tf.concat([time_stamps,values,masks,delta,paddings],2))
            
            if training:
                self.add_loss(self.loss_regularizer*self.mae((masks)*values,(masks)*feaures_embedded[0]))
                self.add_loss(self.loss_regularizer*self.mae((masks)*values,(masks)*feaures_embedded[1]))
                self.add_loss(self.loss_regularizer*self.mae((masks)*values,(masks)*feaures_embedded[2]))
                if self.is_timestamps_generated:
                    self.add_loss(self.loss_regularizer*self.mae((paddings)*delta,(paddings)*feaures_embedded[4]))
        
            if self.is_timestamps_generated:
                #Update the timestamps matrix
                time_stamps=paddings*time_stamps + (1.-paddings)*feaures_embedded[3]
                #Update the time variation matrix
                delta=paddings*delta + (1.-paddings)*feaures_embedded[4] 
             
            #Update the value matrix
            feaures_embedded_=masks*values+(1.-masks)*feaures_embedded[2]
            
            if self.is_alignment:
                #Alignment
                feaures_embedded_=self.alnn([feaures_embedded_,time_stamps,masks,delta,paddings])
            else:
                feaures_embedded_=self.gru(feaures_embedded_)
        else:
            #Alignment
            feaures_embedded_=self.alnn([values,time_stamps,masks,delta,paddings])
           
        #GRU+Classifier
        feaures_embedded_=self.gru(feaures_embedded_)
        feaures_embedded_=self.dense1(feaures_embedded_)
        feaures_embedded_=self.drop1(feaures_embedded_,training=training)
        feaures_embedded_=self.classifier(feaures_embedded_)
        return feaures_embedded_
    
    def get_config(self):
        config = super(PaddALNNGRU, self).get_config()
        config.update({"no_features": self.no_features})
        config.update({"alnn_dropout_rate": self.alnn_dropout_rate})
        config.update({"time_interval_reference_time_point": self.time_interval_reference_time_point})
        config.update({"type_distance": self.type_distance})
        config.update({"init_time": self.init_time})
        config.update({"loss_regularizer": self.loss_regularizer})
        config.update({"max_timestamp": self.max_timestamp})
        config.update({"is_timestamps_generated": self.is_timestamps_generated})
        config.update({"padd_gru_units": self.padd_gru_units})
        config.update({"padd_gru_dropout": self.padd_gru_dropout})
        config.update({"is_imputation": self.is_imputation})
        config.update({"is_alignment": self.is_alignment})
        config.update({"gru_units": self.gru_units})
        config.update({"gru_dropout": self.gru_dropout})
        return config 