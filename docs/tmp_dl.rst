DL section comments
======================================

From FB RWI paper:

    We use a 50-layer resnet50 network (36), where pre-training is similar to Mahajan et. al.(32). This network is
    trained on 3.5 billion public Instagram images (several orders of magnitude larger than the original Imagenet
    dataset) to predict corresponding hashstags. We extract the 2048-dimensional vector from the penultimate layer of
    the pre-trained network, without fine-tuningthe network weights. The satellite imagery has a native resolution of
    0.58 meters/pixel. We downsample these images to 9.375m/pixel resolution by averaging each 16x16 block. The
    downsampled images are segmented into 2.4km squares, then passed through the neural network. For each satellite
    image, we do a forward-pass through the network to extract the 2048 nodes on the second-to-last layer. We then
    apply PCA to this 2048-dimensional object and extract the first 100 components. The PCA eigenvectors are
    computed from images in the training dataset (i.e., the images from the 56 countries with household surveys)

So just using pytorch built-in pretrained resnet50 network.

Options to extend:

- Blah
