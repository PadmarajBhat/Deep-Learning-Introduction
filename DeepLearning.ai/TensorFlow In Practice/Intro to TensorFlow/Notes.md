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
      
* **Max Pooling** actually takes the best of the window(which is usually of size 2x2). Hence, keeps the important value and shrinks the size.
  * a 148 x 148 x 64 convolution output will be shrinked to 74x74x64 with no additional parameters.
    * there are no parameters because it only a max function on the window
    * the size os half as it is a window 2x2 and hence reduces the size to half in both hieght and width.

* **tf.feature_column** helps us  to map the categorical values to tensor/keras layers (https://www.tensorflow.org/guide/feature_columns)

  * indicator column :  is equivalent to one hot encoding : 1 for the categorical value and 0 for others but what if the millions or billions categorical values.
  * embedding column : compresses the number of features in case of many categorical values. the values inside the vector represenation is learnt during the training processes and the number of features to represent a categorical feature depends on the formula 
    * number of categorical to the power of 0.25

* ##### when is densefeature used?
* ##### how do we feed the pandas to tf pipeline?
  * Ans: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/keras/feature_columns.ipynb#scrollTo=dxQloQ9jOoXL

* ##### what is tensorboard? How it is used ? should it be used at all ?
* ##### what is distributed training in tensorflow?
  ```
  stratergy = tf.distribute.MirrorStratergy()
  stratergy.scope():
    model = Sequential([....])
    model.compile(..)
  ```
  * this actually duplicates the model in to gpu and share the parameters during training and hence it is linearly scalable as in more the number of gpu more faster is the training.
  * ##### should the data be distributed before building model ? or TF variable automatically scales/sliced to different worker nodes?
  * ##### is it a replacement to spark or rapid + dask ?
  * ##### Does automatically recognizes the underlying cluster manager like yarn or mesos and completely abstract the configuration requirements ? Does it also provide the parameter facilities to override default configuration ?
      * The author of the video at Google I/O at 15th minute said that it is one machine with multiple GPU in it is what the above code achieves parallelism. https://www.youtube.com/watch?v=lEljKc9ZtU8&list=PLQY2H8rRoyvy2_vtWvCpQWM9GJXNTa5rV&index=2
      
* ##### rapids.ai vs tensorflow distributed. is it a right set of comparison ? are these to alternatives ?
* ##### How to save the model in tensorflow?
  * model.save("path and file name", save_format = 'tf'
  * tf.keras.models.load_model("path\to\file")

* **tensorflow serving**: 
  * is it only to load the model to tensorflow based server and ui has access to it grpc ?
  
* **tensorflow lite**:
  * is it only for **loading** the model and not to **build** the model on the mobile device ?
  
* **tensorflow JS**:
 * is it to load the model in the ui application to immediately load the model and predict online ?

* 
