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
    * ```tf.data.Dataset.from_tensor_slices((df.values, target.values))``` 
     *  This most often as the panda reades both features and labels
       * alternatively, you can have ```tf.data.Dataset.from_tensor_slices((df[df.columns[:-1]].values, df[df.columns[-1]].values))``` 
       * if you dont want intrim variable
    or
    * ```tf.data.Dataset.from_tensor_slices((dict(df), labels))```
      * here you convert the pandas to dictionary and then hypothetically you have seperate labels variable
      
    * In either cases , from_tensor_slices, is the magic function.

* ##### what is tensorboard? How it is used ? should it be used at all ?
  * First of all, you need not have to use it at all the times a model is built. You can just scroll through the output of the training output and make a guess manually if the loss is decreasing or increasing. However, it might get tricky if the epochs are > 50 then you would need a plot. Here you can use **history** can be used to plot the loss and accuracy
    ```
    history = model.fit(...)
    print( history.history.keys())
    ....plot code.... #https://keras.io/visualization/
    ```
    * If you want to get away with those head aches you can use tensorboard. I used it to see
       * keras layer in graph
       * training and validation prediction scores over epochs 
       * loss per epochs
    * To Kick start with loading
      ```!pip install -q tf-nightly-2.0-preview
      # Load the TensorBoard notebook extension
      %load_ext tensorboard```
     * Define the logs to read the statistics from
      ```
       logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
       tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
      ```
     * Finally, launch it post training:
      ```
      %tensorboard --logdir logs
      ```
     * Few more control through notebook 
      ```
      from tensorboard import notebook
      notebook.list() # View open TensorBoard instances
      # Control TensorBoard display. If no port is provided, 
      # the most recently launched TensorBoard is used
      notebook.display(port=6006, height=1000) 
      ```
      
      * You can also plug in couple of lines in the code and can have good visualization for the hyper parameter tuning.
     		* https://www.youtube.com/watch?v=xM8sO33x_OU 8 mins video for more details.
		
* ##### what is distributed training in tensorflow?
![TF Arch](https://github.com/PadmarajBhat/Deep-Learning-Introduction/blob/master/TF%20DataFlow%20Pipeline.PNG)
  ```
  stratergy = tf.distribute.MirrorStratergy()
  stratergy.scope():
    model = Sequential([....])
    model.compile(..)
  ```
  * this actually duplicates the model in to gpu and share the parameters during training and hence it is linearly scalable as in more the number of gpu more faster is the training.
  * https://www.tensorflow.org/guide/distribute_strategy indicates that the distributed can also be multiple system with multiple gpu
  
  
  * ##### should the data be distributed before building model ? or TF variable automatically scales/sliced to different worker nodes?
  * ##### is it a replacement to spark or rapid + dask ?
    * No, it complements spark. https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/hadoop.md indicates that tf can read the HDFS file. This concludes that it has its presence in hadoop. However, it need not use spark architecture to assign tasks to worker node. Instead, itself manages the parameter update from the worker node. It is unlike the spark echo system, where a job is written with single source of input and single node approach where as spark makes the seemless access to single source input of the distributed data and does the heavy lifting of assigning the same computation on different worker node for the data they have or assigned with.
    * What I meant above is that the training is not a job to spark echo system. Or it does, the TF might internally initiate a job submit and then the task could be just to compute the gradient to update the parameter and pipe the another batch to workers. That way tensorflow can leverage the spark api and keep its focus on ml stuff.
    * https://www.tensorflow.org/guide/extend/architecture : talks about the graph computation being distributed to workers. It talks basically about the master and worker fundamental tasks and flow of graph (or sub graph) computation between the two.
    
    * tf.distribute.experimental.MultiWorkerMirroredStrategy : this replicates the same model in different workers and gets the parameter update in the sync fashion.
    
  * ##### Does automatically recognizes the underlying cluster manager like yarn or mesos and completely abstract the configuration requirements ? Does it also provide the parameter facilities to override default configuration ?
      * The author of the video at Google I/O at 15th minute said that it is one machine with multiple GPU in it is what the above code achieves parallelism. https://www.youtube.com/watch?v=lEljKc9ZtU8&list=PLQY2H8rRoyvy2_vtWvCpQWM9GJXNTa5rV&index=2
      
* ##### rapids.ai vs tensorflow distributed. is it a right set of comparison ? are these to alternatives ?
* ##### How to save the model in tensorflow?
  * model.save("path and file name", save_format = 'tf'
  * tf.keras.models.load_model("path\to\file")

* **tensorflow serving**: 
  * is it only to load the model to tensorflow based server and ui has access to it through grpc ?
    * it can give both API or grpc access
      * anyone can access the service with valid parameter and would get the prediction output. Auto scaled and versioning mechanism is also provided.
  
* **tensorflow lite**:
  * is it only for **loading** the model and not to **build** the model on the mobile device ?
  
* **tensorflow JS**:
 * is it to load the model in the ui application to immediately load the model and predict online ?

* TensorFlow is a library written in C++ and gets compiled at backend to create a AutoGraph. It can also be accelerated.
  ```
  @tf.function
  ```
  This one liner has to be placed before to the function definition which has tf code to get accelerated.
  
 * tf older version can be migrated to tf 2.0 with simple script run : https://medium.com/tensorflow/upgrading-your-code-to-tensorflow-2-0-f72c3a4d83b5

* There can be unknown biases:
  * there are issues like if all the images are of Indian Hindu wedding, even if the accuracy is 90% . The model is only 10% accurate for he world dataset 
  * Similar example would be in translating where language model can be biased to detect the musculine and feminine words.
  * The solution is only to look the data in various angles, geographical data, open data to be not biased to one source, test early and test often.
  * Note that the data keeps changing, there cannot be same pattern of interaction with the system.
* What is the difference between federated and distributed learning?

* **SWIFT**: TF team promotes swift programming language for tf coding for following reasons-
	* Interoperability with other languages
		* you can import the python (any) libraries like matplotlib or numpy
		* you can import c libraries 
	* typed language: early detection of errors, compiled for faster execution
	* efficient compiler for low latency and auto graph detection
	* oriented for reasearchers to work to play around with different layer definitiona and optimization experiments.

##### Questions (Inerviewing myself):
* Importing through tf.data : https://www.youtube.com/watch?v=oFFbKogYdfc
	* the above video tutorial indicates ``` tf.contrib.data.CsvDataset``` to use for the csv read. However, the latest version indicates, that we have to user keras util function to load csv: https://www.tensorflow.org/beta/tutorials/load_data/csv
	```
	TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
	train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
	```
		importing various types of data	
			HDFS
			file read
			csv
			image processing
			any pipes? like steams
			can we save back the data post processing
				batching

* what are the different tensorflow data input processing capabilities?
	* https://www.youtube.com/watch?v=-nTe44WT0ZI talks about the dataset operations. i.e. once data is is provided as dataset what are operation that can be done on it
		* map : apply series of preprocessing stpes 
			* ** num_parallel_calls** : for high throughput
		* shuffle and batch : most common operation to remove any bias inference through the order of data in data set is to shuffle. Batch is for both splitting the huge data set into smaller subset (to fit in the physical device capacity ) for one iteration of the neural network and also to give balanced target/label dataset for each iteration
		* prefetch: to avoid cpu/gpu starving for the input post training
		
	what else is there in the tensorflow	
		distributed processing
		saving models
		building NN models through keras
		tensorboard

	can spark dataframe be used along with tensor_slicing (like that of panda)
		incremental learning / pausing the learning
		batch normalization
		transfer learning
		what is debuggin model ?
			watching variables ?
		any inbuilt plotting like plotting?
		hyperparameter tuning ?
			for estimators only ?
transfer learning for word processing


 
