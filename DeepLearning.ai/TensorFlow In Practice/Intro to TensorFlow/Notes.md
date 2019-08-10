##### Notes on Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
* Machine learning in a sentence: data + answer go into model and model predicts the rules mapping data and answers.
* executed the colab work to check if the increase in neuron for the first layer makes any difference. 
* flatten layer at the input is the must if we are not using cnn but if we have cnn then conv2d can take 3d input.
* flatten layer will be used in cnn as well at the end of convolutions. This will reduce the number of nuerons requirement without compromising the image recognition accuracy.
* normalization required so that loss function or optimizer would not land on high values.
* quiet obvious but even then the number of classses in the neural network should be equal to the final dense layer shape.
  * usually relu layer is used for all the activations of the neurons
    * activation : outcome of neuron processing goes through a function (relu / softmax) and the result determines that the neuron has some value for next layer or not. Hence, "activation". With the abundandant neurons in the neural network, there is high chance of some neurons not contributing to the final outcome.
      * neuro processing: summation of (input features * weight ) + bias
      
* *Max Pooling* actually takes the best of the window(which is usually of size 2x2). Hence, keeps the important value and shrinks the size.
  * a 148 x 148 x 64 convolution output will be shrinked to 74x74x64 with no additional parameters.
    * there are no parameters because it only a max function on the window
    * the size os half as it is a window 2x2 and hence reduces the size to half in both hieght and width.

* *tf.feature_column* helps us  to map the categorical values to tensor/keras layers (https://www.tensorflow.org/guide/feature_columns)
