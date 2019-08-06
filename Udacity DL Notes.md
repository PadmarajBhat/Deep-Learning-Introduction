#### Notes from the chapter: Going Ahead with CNNs and rest
* When we say that image of size X * Y * Z then X and Y values are intuitive because the image we see is 2d and hence we can have a picture of X & Y. Now Z has to pictured as a layer of size X * Y and number of layers depends on the value of Z. Usually, Z =3 corresponding to RGB. In Other words, There are 3 layers, one for red with X*Y values, one for green with X*Y values and one for Blue with X*Y. Now, these are stacked on each other like a cube. Now when a ray of light  passes through all the 3 layers, then the compound effect of lights results in the beautiful colored picture.
* since we have 3 layers in the input, kernel is also 3 layers but the output is 2d. i.e. convoluted image is 2d but like the conventional approach is to have multiple kernel, here also we can have multiple 3 layers kernel because of which we will have nd kernel output where n is number of kernels.
* when it comes to max pooling, the same formula applies. We would have one layer per convoluted layer and hence multilayer max pooled layer. Note, usually strides and max pooling is applied so that max pooling keeps the important and strides reduce the size of the image.
* https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c01_dogs_vs_cats_without_augmentation.ipynb for quick code to train cat and dog color images
  * This is an example of how the over fitting can happen and the way it is identified is through 2 plots. One for the accuracy comparison between training and validation and other plot is to compare the loss between the training and testing.

* Image Augumentation: Synthesis of the images when training data is low in volume (we see that the model overfit for various combination of the CNN)
  * we can vertical or horizontal flip the images, random zoom and chop off the image with no charasteristic features.
    * similar to SMOTE, synthetic minority oversampling technique, where algorithm tries to put an additional point in the cluster to increase the number of samples of a minority class.

* https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42 talks about 6 different techniques to overfit the model.
  * get more data
  * augument the data
  * drop out
  * l1 and l2 regularization on feature weights
    * l1 penalizes the absolute value of the weights
      * can identify the important features by making the non important feature's weight as 0
      * less computation efficient (why?)
    * l2 penalizes the squared value of the weights 
      * better computational efficient
      * thumb rule is to first apply l2 and see if it improves the accuracy and if not then use l1 to penalize the non important
   * early termination: overfitting occurs when we focus on increasing accuracy but we can stop the epoch to a preset accuracy level. This may be done by having extra epochs and then a plot of training and validation plot and then in the second iteration of training stop at the position noted at the last iteration.
   * Simplify the model by providing the hyper parameter values with again prior experience of the dataset modelling.
   
##### Transfer Learning:
* it is all about transferring the learning from one model to the other. As the model learns the importance of the features through the training, transfer learning focuses on transferring the weights to new model.
 * note that the source of the transfer learning cannot be used as is because the purpose of the reference model may be different. for e.g. it might have been to detect the different number of classes. We also need to identify the right reference model because a car classification model cannot be used for flower class prediction. However, if the reference model is generic enough to detect car and flower can be used to fine tune new set of flower or car model prediction.
 
 * clearly, the last layer might not be usable for current use. For e.g. there need not have to be thousand types for class predictions, we might need only two. And thus using the same model might not result in the needed class prediction. We may have to chop of the last layer or usually chopped model pre exist. When we add a new layer, we have to retrain the model with available dataset for pointing it to right classes.
 * Nice visualization of CNNs for image classification : https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b
