# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import random

from sklearn.metrics import mean_squared_error as mse

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

# +
### READ DATA ###

df = pd.read_csv('Aggregated-2013-2017.csv')
df['date_time'] = pd.to_datetime(df['date_time'])
df.drop_duplicates('date_time', inplace=True)
df.set_index('date_time', inplace=True)

print(df.shape)
df.head()
# drop_indices = np.random.choice(df.index, 4000, replace=False)
# df =  df.drop(drop_indices)
# print(df.shape)

# +
### INSERT MISSING DATES ###

df = df.reindex(pd.date_range(df.head(1).index[0], df.tail(1).index[0], freq='H'))

df.shape

# +
### PLOT TRAFFIC SAMPLE ###

df.use.tail(2000).plot(figsize=(18,5))
plt.ylabel('Electricity Use')

# +
### PLOT MISSING VALUES OVER TIME ###

plt.figure(figsize=(18,5))
sns.heatmap(df[['use']].isna().T, cbar=False, cmap='plasma', 
            xticklabels=False, yticklabels=['Electricity Use'])
plt.xticks(range(0,len(df), 24*180), list(df.index.year[::24*180]))
np.set_printoptions(False)

# +
### FILL MISSING VALUES ###

df = df[df.index.year.isin([2015,2016,2017])].copy()

df = pd.concat([df.select_dtypes(include=['object']).fillna(method='backfill'),
                df.select_dtypes(include=['float']).interpolate()], axis=1)

df.shape

# +
### PLOT TRAFFIC DISTRIBUTION IN EACH MONTH ###

plt.figure(figsize=(9,5))
sns.boxplot(x=df.index.month, y=df.use, palette='plasma')

plt.ylabel('Electricity Use'); plt.xlabel('month')

# +
### PLOT TRAFFIC DISTRIBUTION IN EACH WEEKDAY ###

plt.figure(figsize=(9,5))
sns.boxplot(x=df.index.weekday, y=df.use, palette='plasma')

plt.ylabel('Electricity Use'); plt.xlabel('weekday')

# +
### PLOT TRAFFIC DISTRIBUTION IN EACH HOUR ###

plt.figure(figsize=(9,5))
sns.boxplot(x=df.index.hour, y=df.use, palette='plasma')

plt.ylabel('Electricity Use'); plt.xlabel('hour')

# +
### NUMERICAL ENCODE CATEGORICAL COLUMNS ###

map_col = dict()

X = df.select_dtypes(include=['object']).copy()

for i,cat in enumerate(X):
    X[cat] = df[cat].factorize()[0]
    map_col[cat] = i
i=-1 if X.empty else i
X['month'] = df.index.month;  i += 1;  map_col['month'] = i
X['weekday'] = df.index.weekday;  i += 1;  map_col['weekday'] = i
X['hour'] = df.index.hour;  i += 1;  map_col['hour'] = i
X.shape


# +
### UTILITY FUNCTION FOR 3D SEQUENCE GENERATION ###

def gen_seq(id_df, seq_length, seq_cols):

    data_matrix =  id_df[seq_cols]
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length, 1), range(seq_length, num_elements, 1)):
        
        yield data_matrix[stop-sequence_length:stop].values.reshape((-1,len(seq_cols)))


# +
### GENERATE 3D SEQUENCES ###

sequence_length = 24*7

sequence_input = []
sequence_target = []

for seq in gen_seq(X, sequence_length, X.columns):
    sequence_input.append(seq)
    
for seq in gen_seq(df, sequence_length, ['use']):
    sequence_target.append(seq)
    
sequence_input = np.asarray(sequence_input)
sequence_target = np.asarray(sequence_target)

sequence_input.shape, sequence_target.shape


# +
### UTILITY FUNCTION TO INSERT RANDOM MISSING INTERVALS ###

def drop_fill_pieces(sequence_input, sequence_target, missing_len, missing_val=np.nan, size=0.2):
    
    sequence_input = np.copy(sequence_input)
    sequence_target = np.copy(sequence_target)
    
    _id_seq = np.random.choice(range(len(sequence_target)), int(len(sequence_target)*size), replace=False)
    _id_time = np.random.randint(0,sequence_length-missing_len, int(len(sequence_target)*size))
    
    for i,t in zip(_id_seq, _id_time):
        sequence_input[i, t:t+missing_len, 
                       []] = -1
        sequence_target[i, t:t+missing_len, :] = missing_val
        
    sequence_input[:,:, 
                   []] += 1
    
    return sequence_input, sequence_target


# +
### INSERT MISSING INTERVALS AT RANDOM ###

np.random.seed(33)

missing_len = 24
sequence_input, sequence_target_drop = drop_fill_pieces(sequence_input, sequence_target,
                                                        missing_len=missing_len, size=0.6)

sequence_input.shape, sequence_target_drop.shape

# +
### TRAIN TEST SPLIT ###

train_size = 0.8

sequence_input_train = sequence_input[:int(len(sequence_input)*train_size)]
sequence_input_test = sequence_input[int(len(sequence_input)*train_size):]
print(sequence_input_train.shape, sequence_input_test.shape)

sequence_target_train = sequence_target[:int(len(sequence_target)*train_size)]
sequence_target_test = sequence_target[int(len(sequence_target)*train_size):]
print(sequence_target_train.shape, sequence_target_test.shape)

sequence_target_drop_train = sequence_target_drop[:int(len(sequence_target_drop)*train_size)]
sequence_target_drop_test = sequence_target_drop[int(len(sequence_target_drop)*train_size):]
print(sequence_target_drop_train.shape, sequence_target_drop_test.shape)


# +
### UTILITY CLASS FOR SEQUENCES SCALING ###

class Scaler1D:
    
    def fit(self, X):
        self.mean = np.nanmean(np.asarray(X).ravel())
        self.std = np.nanstd(np.asarray(X).ravel())
        return self
        
    def transform(self, X):
        return (X - self.mean)/self.std
    
    def inverse_transform(self, X):
        return (X*self.std) + self.mean


# +
### SCALE SEQUENCES AND MASK NANs ###

scaler_target = Scaler1D().fit(sequence_target_train)

sequence_target_train = scaler_target.transform(sequence_target_train)
sequence_target_test = scaler_target.transform(sequence_target_test)

sequence_target_drop_train = scaler_target.transform(sequence_target_drop_train)
sequence_target_drop_test = scaler_target.transform(sequence_target_drop_test)

mask_value = -999.
sequence_target_drop_train[np.isnan(sequence_target_drop_train)] = mask_value
sequence_target_drop_test[np.isnan(sequence_target_drop_test)] = mask_value

# +
### UTILITY FUNCTIONS FOR VAE CREATION ###

latent_dim = 2

def sampling(args):
    
    z_mean, z_log_sigma = args
    batch_size = tf.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

def vae_loss(inp, original, out, z_log_sigma, z_mean):
    
    reconstruction = K.mean(K.square(original - out)) * sequence_length
    kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))

    return reconstruction + kl

def get_model():
    
    ### encoder ###
    
    inp = Input(shape=(sequence_length, 1))
    inp_original = Input(shape=(sequence_length, 1))
    
    cat_inp = []
    cat_emb = []
    for cat,i in map_col.items():
        inp_c = Input(shape=(sequence_length,))
        if cat in ['holiday', 'weather_main', 'weather_description']:
            emb = Embedding(X[cat].max()+2, 6)(inp_c)
        else:
            emb = Embedding(X[cat].max()+1, 6)(inp_c)
        cat_inp.append(inp_c)
        cat_emb.append(emb)
    
    concat = Concatenate()(cat_emb + [inp])
    enc = LSTM(64)(concat)
    
    z = Dense(32, activation="relu")(enc)
        
    z_mean = Dense(latent_dim)(z)
    z_log_sigma = Dense(latent_dim)(z)
            
    encoder = Model(cat_inp + [inp], [z_mean, z_log_sigma])
    
    ### decoder ###
    
    inp_z = Input(shape=(latent_dim,))

    dec = RepeatVector(sequence_length)(inp_z)
    dec = Concatenate()([dec] + cat_emb)
    dec = LSTM(64, return_sequences=True)(dec)
    
    out = TimeDistributed(Dense(1))(dec)
    
    decoder = Model([inp_z] + cat_inp, out)   
    
    ### encoder + decoder ###
    
    z_mean, z_log_sigma = encoder(cat_inp + [inp])
    z = Lambda(sampling)([z_mean, z_log_sigma])
    pred = decoder([z] + cat_inp)
    
    vae = Model(cat_inp + [inp, inp_original], pred)
    vae.add_loss(vae_loss(inp, inp_original, pred, z_log_sigma, z_mean))
    vae.compile(loss=None, optimizer=Adam(lr=1e-3))
    
    return vae, encoder, decoder


# +
tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)


es = EarlyStopping(patience=10, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)
vae, enc, dec = get_model()
vae.fit([sequence_input_train[:,:,i] for cat,i in map_col.items()] + [sequence_target_drop_train, sequence_target_train], 
         batch_size=128, epochs=100, validation_split=0.2, shuffle=False, callbacks=[es])

# +
### COMPUTE RECONSTRUCTION ###

vae = Model(vae.input[:-1], vae.output)

reconstruc_train = scaler_target.inverse_transform(
    vae.predict([sequence_input_train[:,:,i] for cat,i in map_col.items()] + [sequence_target_drop_train]))
reconstruc_test = scaler_target.inverse_transform(
    vae.predict([sequence_input_test[:,:,i] for cat,i in map_col.items()] + [sequence_target_drop_test]))

reconstruc_train.shape, reconstruc_test.shape

# +
### PLOT REAL vs RECONSTRUCTION ###

id_seq = 100

seq = np.copy(sequence_target_drop_test[id_seq])
seq[seq == mask_value] = np.nan
seq = scaler_target.inverse_transform(seq)

plt.figure(figsize=(9,5))
plt.plot(reconstruc_test[id_seq], label='reconstructed', c='red')
plt.plot(seq, c='blue', label='original', alpha=0.6)
plt.legend()

# +
### PLOT REAL vs RECONSTRUCTION ###

id_seq = 800

seq = np.copy(sequence_target_drop_test[id_seq])
seq[seq == mask_value] = np.nan
seq = scaler_target.inverse_transform(seq)

plt.figure(figsize=(9,5))
plt.plot(reconstruc_test[id_seq], label='reconstructed', c='red')
plt.plot(seq, c='blue', label='original', alpha=0.6)
plt.legend()

# +
### COMPUTE PERFORMANCES ON TRAIN ###

mask = (sequence_target_drop_train == mask_value)

print('reconstruction error on entire sequences:',
    mse(np.squeeze(reconstruc_train, -1), np.squeeze(sequence_target_train, -1), squared=False))
print('reconstruction error on missing sequences:',
    mse(reconstruc_train[mask].reshape(-1,missing_len), sequence_target_train[mask].reshape(-1,missing_len), squared=False))

# +
### COMPUTE PERFORMANCES ON TEST ###

mask = (sequence_target_drop_test == mask_value)

print('reconstruction error on entire sequences:',
    mse(np.squeeze(reconstruc_test, -1), np.squeeze(sequence_target_test, -1), squared=False))
print('reconstruction error on missing sequences:',
    mse(reconstruc_test[mask].reshape(-1,missing_len), sequence_target_test[mask].reshape(-1,missing_len), squared=False))

# +
### GET LATENT REPRESENTATION ON TRAIN DATA ###

enc_pred, _ = enc.predict([sequence_input_train[:,:,i] for cat,i in map_col.items()] + [sequence_target_drop_train])
enc_pred.shape

# +
### PLOT LATENT REPRESENTATION ###

for cat,i in map_col.items():
    plt.scatter(enc_pred[:,0], enc_pred[:,1], c=sequence_input_train[:,sequence_length//2,i], cmap='plasma')
    plt.title(cat); plt.show()

# +
### GENERATE RANDOM PERMUTATION ###

np.random.seed(33)

id_seq = 3333

_X = np.random.normal(enc_pred[id_seq,0], 3, 10)
_Y = np.random.normal(enc_pred[id_seq,1], 3, 10)
_cat_input = [sequence_input_train[[id_seq],:,i] for cat,i in map_col.items()]

# +
### PLOT RANDOM PERMUTATION ###

plt.figure(figsize=(9,5))
        
for x in _X:
    for y in _Y:
        dec_pred = dec.predict([np.asarray([[x,y]])] + _cat_input)
        plt.plot(scaler_target.inverse_transform(dec_pred[0]), c='orange', alpha=0.6)
plt.plot(scaler_target.inverse_transform(sequence_target_train[id_seq]), c='blue')
