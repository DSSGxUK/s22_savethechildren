DL steps
======================================

.. _autoencoder:

The principal deep learning avenue we explored for feature engineering was the use of autoencoding.
That is, to incorporate more information about the distribution of e.g. population or other bands from
the GeoTIFFs in our dataset, rather than simply mean aggregated values within each hexagon, we want to
reduce the dimension from that of the full image about a hexagon (e.g. 16x16 or 32x32) down to a chosen dimension,
while losing as little information as possible overall (such that we may reconstruct the image).

Autoencoders are one option for doing so, where we train an architecture to recover an image after reducing
(in the 'encoder' stage) to a small number of dimensions, by then 'decoding'. After the network is trained, the
decoder may then be removed, and the encoded representations of the input 'images' (i.e. stacked bands of all GeoTIFFs
around the chosen location) used as a low-dimensional representation.

Comparing the quality of this representation to other conventional approaches such as PCA and UMAP using pyDRMetrics,
we find that the autoencoder performs significantly better, as we might hope, even when trained on only a small subset
of a country.

The simple architecture used is as follows:

.. code-block:: console
    Model: "autoencoder-v1"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #
    =================================================================
    conv1 (Conv2D)              (None, 16, 16, 32)        17600
    mp1 (MaxPooling2D)          (None, 8, 8, 32)          0
    conv2 (Conv2D)              (None, 8, 8, 16)          4624
    mp2 (MaxPooling2D)          (None, 4, 4, 16)          0
    conv3 (Conv2D)              (None, 4, 4, 8)           1160
    mp3 (MaxPooling2D)          (None, 2, 2, 8)           0
    conv4 (Conv2D)              (None, 2, 2, 8)           584
    Encoder_Output (MaxPooling2  (None, 2, 2, 8)          0
    D)
    conv5 (Conv2D)              (None, 2, 2, 8)           584
    us1 (UpSampling2D)          (None, 2, 2, 8)           0
    conv6 (Conv2D)              (None, 2, 2, 8)           584
    us2 (UpSampling2D)          (None, 4, 4, 8)           0
    conv7 (Conv2D)              (None, 4, 4, 16)          1168
    us3 (UpSampling2D)          (None, 8, 8, 16)          0
    conv8 (Conv2D)              (None, 8, 8, 32)          4640
    us4 (UpSampling2D)          (None, 16, 16, 32)        0
    Decoder_Output (Conv2D)     (None, 16, 16, 61)        17629
    =================================================================
    Total params: 48,573
    Trainable params: 48,573
    Non-trainable params: 0
    _________________________________________________________________


.. _transfer learning:

An alternative approach would be more conventional feature extraction / transfer learning - for instance,
from `FB RWI`_ paper:

    We use a 50-layer resnet50 network (36), where pre-training is similar to Mahajan et. al.(32). This network is
    trained on 3.5 billion public Instagram images (several orders of magnitude larger than the original Imagenet
    dataset) to predict corresponding hashstags. We extract the 2048-dimensional vector from the penultimate layer of
    the pre-trained network, without fine-tuningthe network weights. The satellite imagery has a native resolution of
    0.58 meters/pixel. We downsample these images to 9.375m/pixel resolution by averaging each 16x16 block. The
    downsampled images are segmented into 2.4km squares, then passed through the neural network. For each satellite
    image, we do a forward-pass through the network to extract the 2048 nodes on the second-to-last layer. We then
    apply PCA to this 2048-dimensional object and extract the first 100 components. The PCA eigenvectors are
    computed from images in the training dataset (i.e., the images from the 56 countries with household surveys)

So basically just using Pytorch built-in pretrained resnet50 network.

Options to extend:

- Our idea that we may trial in future is to freeze the resnet, and then simply add trainable layers before and after before
  performing end-to-end training. After this is completed, depending on performance the outputs prior to the new final layer
  could be used as input features.

.. _FB RWI: https://arxiv.org/pdf/2104.07761.pdf
