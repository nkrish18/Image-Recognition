# Image Classification with CNN

## Objective

The objective of this project is to build an image recognition model that classifies a set of images into their corresponding categories. Each of these categories corresponds to a specific type of clothing. The aim is to use convolutional neural networks and other machine learning algorithms and optimizers after cleaning and pre-processing to culminate in the most accurate model for the job.

## Dataset description

The dataset comprises of Zalando's article images consisting of a set of 60,000 training images and 10,000 testing images. Each example is a 28x28 image, associated with a label from 10 classes. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels, and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.

These images are converted to a more standard format with 32x32 pixels by adding padding.


## Preprocessing

As in any machine learning problem, initial data cleaning and pre-processing is integral in achieving the most optimal and accurate results. 

It is possible that certain images could be crooked or deformed. Deskewing is done to overcome this. Since the skewed image is an affine transformation on the original, the center of mass of the image and the covariance matrix of pixel intensities are calculated to find the offset and angle of skew respectively and then it is skewed back to the original image. 

A ‘Non-local Means Denoising algorithm’ is then used to remove noise from the images. It averages out the pixel values from similar neighbouring windows from the affected pixel and replaces it. 

In order to avoid rapid changes in pixel intensity across an image, ‘Gaussian blurring’ or smoothening is done on the images for better edge detection when constructing neural networks.

Finally, the data is normalized and scaled. Since the maximum value of the pixel intensity is 255, each of the data points is scaled by a factor of 255 so that all the pixels have a value between 0 and 1. This helps algorithms converge at a much faster rate and improves performance.

Since neural networks require class labels to be in ‘One hot encoding’, the training and testing labels are converted into this format. As there are 10 different types of classes here, each label is converted into a 1-D array with 10 binary values. For instance, if the first value of the array is 1 and the rest are 0, then the class label corresponds to a shoe.


## Modelling and Classification

###Convolutional Neural Network

#### Image Augmentation

Though 60,000 training samples is a relatively large amount, deep learning methods generally require much more training data to perform at their best. Data augmentation is one way of increasing training samples by using the existing data by artificially creating images by various processing methods. The various augmentation methods implemented on this dataset are rotations, zoom, shears, width shifts and horizontal flips. This step is also necessary to increase the robustness of the model and control overfitting.


### Architecture

The ‘keras’ package with the tensorflow backend was used for building this convolutional network. A total of five convolution 2D layers are used here.

The first convolution layer consists of 32 filters, each of size (5x5) with a ‘relu’ activation layer. A relatively large filter size is used here because, being the first layer, it is used to identify the more general features like curves and edges. This is followed by a batch normalization layer to minimize the covariance shift and speed up the training process. The second and third convolution layers consist of 32 kernels, each of size (3x3). In order to preserve the maximum amount of information, padding is added to the output layers. These are also followed by batch normalization layers. A max pooling layer with a (2x2) kernel is added to reduce the number of learning parameters and computational costs. Additionally, a dropout layer is also added after the third convolution layer to help control overfitting. The fourth and fifth convolution layers each have 64 filters of size (2x2). The size of filters have been reduced in order to identify more detailed and intricate features in the final layers. Both these layers are followed by batch normalization and dropout layers. The final convolution layer is also followed by another maxpooling layer.

The output from the final maxpooling layer is flattened and normalized again before going to the fully connected layers for classification. Two of these layers are used, each with 128 neurons along with dropout and normalization layers. The outputs are then finally sent to a dense softmax layer for converting them to probabilities for each class.


### Optimizer

The ‘Adam’ optimizer which is an extension of SGD is used here. The initial learning rate is set to 0.01. A learning rate reducer is used as callback to reduce rate by a factor of 0.8 if there is no improvement after three continuous epochs, upto a minimum learning rate of 0.0001. The loss function used is ‘categorical crossentropy’ for 50 epochs.



## Other Models

In order to appreciate the benefits and capability of deep learning methods, the random forest and SVM algorithms were also fitted on the same dataset inorder to compare performance.

For Support Vector Machines, In order to deal with high dimensionality, Principal Component Analysis (PCA) is used.




