import tensorflow
from tensorflow import keras
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
import numpy as np
from tensorflow.keras.optimizers import Adam
import gc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import keras_tuner as kt
from pyDRMetrics.pyDRMetrics import *

random_state=0
np.random.seed(random_state)

def get_trained_autoencoder(input_data, validation_split, random_state,
                            batch_size=[16, 32, 64, 128],
                            learning_rate=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
                            epochs=200, logdir="autoencoder",
                            project_name="tune_model", es_patience=5):
  """
  Get the trained model with tuned hyperparameters

  Inputs:
    input_data: image input of size (num of samples, width, height, bands);
                reshaped & imputed input of convert_tiffs_to_image_dataset
    validation_split: split ratio for model.fit()
    random_state: random state
    batch_size: list of batch sizes to check
    learning_rate: list of learning rates to check
    epochs: max epochs for training; specify value slightly higher than expected convergence
    logdir: directory for logging
    project_name: name of project
    es_patience: patience for early stopping

  Outputs: trained keras sequential autoencoder model
  """
  input_dims = input_data.shape

  class HyperModel(kt.HyperModel):
    def build(self, hp):
      
      model = tensorflow.keras.Sequential()
      
      model.add(keras.Input(shape=input_dims[1:]))
      model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'))
      model.add(MaxPooling2D((2, 2), padding='same', name='mp1'))
      model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2'))
      model.add(MaxPooling2D((2, 2), padding='same', name='mp2'))
      model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv3'))
      model.add(MaxPooling2D((2, 2), padding='same', name='Encoder_Output'))
      
      model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv4'))
      model.add(UpSampling2D((2, 2), name='us1'))
      model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv5'))
      model.add(UpSampling2D((2, 2), name='us2'))
      model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv6'))
      model.add(UpSampling2D((2, 2), name='us3'))
      model.add(Conv2D(input_dims[-1], (3, 3), activation=None, padding='same', name='Decoder_Output'))
      
      hp_learning_rate = hp.Choice('learning_rate', values=learning_rate)
      
      model.compile(
          optimizer=Adam(learning_rate=hp_learning_rate),
          loss=keras.losses.MeanSquaredError()
          )
      print(model.summary())
      return model
    
    def fit(self, hp, model, *args, **kwargs):
      return model.fit(*args,
                     batch_size=hp.Choice("batch_size", batch_size),
                     epochs=epochs,
                     validation_split=validation_split,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=es_patience), TensorBoard("/tmp/tb_logs")]
                     )
  
  tuner = kt.Hyperband(HyperModel(),
                     objective='val_loss',
                     factor=5,
                     max_epochs=epochs,
                     overwrite=True,
                     directory=logdir,
                     project_name=project_name,
                     seed=random_state)
  
  tuner.search(input_data, input_data)

  best_hps = tuner.get_best_hyperparameters()[0]
  print(f"""
  Best LR: {best_hps.get('learning_rate')}
  Best Batch Size: {best_hps.get('batch_size')}.
  """)
  model = tuner.hypermodel.build(best_hps)
  history = model.fit(input_data, input_data, epochs=epochs, validation_split=validation_split)
  
  val_loss_per_epoch = history.history['val_loss']
  best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
  print('Best epoch: ', str(best_epoch))

  tuned_model = tuner.hypermodel.build(best_hps)
  tuned_model.fit(input_data, input_data, epochs=best_epoch)

  return tuned_model

def check_autoencoder_reconstruction(trained_autoencoder, input_data):
  
  """
  Plot reconstructed images to check performance of autoencoder

  Inputs:
    trained_autoencoder: output of get_trained_autoencoder() or saved keras model
    input_data: image input of size (num of samples, width, height, bands);
                reshaped & imputed input of convert_tiffs_to_image_dataset

  Outputs: None
  """
  decoded_imgs = trained_autoencoder.predict(input_data)
  n = 5
  plt.figure(figsize=(20, 8))
  for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    band = np.random.randint(0, input_data.shape[-1])
    sample = np.random.randint(0, input_data.shape[0])
    plt.imshow(input_data[sample,:,:,band].reshape(input_data.shape[1], input_data.shape[1]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Actual Sample:"+str(sample)+", Band:"+str(band))

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[sample,:,:,band].reshape(input_data.shape[1], input_data.shape[1]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Reconstructed Sample:"+str(sample)+", Band:"+str(band))
  
  plt.show()

def get_encoded_features(trained_autoencoder, input_data, get_encoding_eval=False):

  """
  Get encoded features from trained autoencoder

  Inputs:
    trained_autoencoder: output of get_trained_autoencoder() or saved keras model
    input_data: image input of size (num of samples, width, height, bands);
                reshaped & imputed input of convert_tiffs_to_image_dataset
    get_encoding_eval: bool variable to print metrics of encoded features

  Outputs: 2D encoded feature array of size (num of samples, original width*original_height/8 )
  """
  intermediate_model = tensorflow.keras.models.Model(inputs=trained_autoencoder.input,
                                                     outputs=trained_autoencoder.get_layer('Encoder_Output').output)
  intermediate_prediction = intermediate_model.predict(input_data)

  if get_encoding_eval:
    drm = DRMetrics(input_data.reshape(input_data.shape[0], -1), intermediate_prediction.reshape(intermediate_prediction.shape[0], -1))
    print("Residual Variance: ", str(drm.Vrs))
    print("AUC Trustworthiness: ", str(drm.AUC_T))
    print("AUC Continuity: ", str(drm.AUC_C))

  return intermediate_prediction