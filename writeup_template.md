# **Traffic Sign Recognition Project**

The goals / steps of this project were the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/frequencyTrainingSet.png "Distribution Samples Training Set"
[image2]: ./results/frequencyValidationSet.png "Distribution Samples Validation Set"
[image3]: ./results/frequencyTestSet.png "Distribution Samples Test Set"
[image4]: ./results/exampleSampleImage.png "Example of Traffic Sign"
[image5]: ./results/exampleAugmentedImage.png "Example of Augmented Image"
[image6]: ./data/newImages/bumpyRoads.jpeg "Bumpy road sign"
[image7]: ./data/newImages/img1s.jpeg "Turn right ahead sign"
[image8]: ./data/newImages/img2s.jpeg "Stop sign"
[image9]: ./data/newImages/img4s.jpeg "No entry sign"
[image10]: ./data/newImages/speedlimits.jpeg "Speed limit 70 sign"

---

### Data Set Summary & Exploration

The first step for the project was to load the data. The data came already with the train, validation and test set separatly.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set was 34799 images.
* The size of the validation set was 4410 images.
* The size of test set was 12630 images.
* The shape of a traffic sign image was (32x32x3)
* The number of unique classes/labels in the data set was 43.

After this first analysis, I analysed the number of samples per class and checked if the distribution was uniform. After printing the number of samples per class (see notebook) it is clear that the number of samples per class was not distributed uniformally. For example, looking at the training set, the class #41 had only 210 samples, and there were other classes with over 2000 samples.

Looking at the number of samples per class in the training set:

![alt text][image1]

For the validation set:

![alt text][image2]

And for the test set:

![alt text][image3]

The three sets (training, validation and test) had a similar distribution. However, as stated before, it is clear that one of the problems was that the number of samples per class were not distributed unifomally across classes.

An example of the images of the class number 13:

![alt text][image4]

### Data Generation (data augmentation)

Since for training, the more data you have, the better, I implemented some code that augment existing data *from the trainign set* (rotate, translate and zoom existing images) so that the model could see images in different positions and orientations. I threfore generated new images from existing images, and I used the ImageDataGenerator from Keras where you can introduce which image augmentation operations you want to use to generate the new images.

The final image augmentations operations (e.g. rotation, etc.) were chosen based on (1) what can make sense to do (e.g. flipping horizontally the image was discarded as then the signs with text would appear with the text reversed, which can be unexpected in the real world), and (2) the validation accuracy obtained after adding these new images on the model.

The modifications I used to augment the images were as follows: rotation of 15 degrees, zoom of 20% and translation hozizontally and vertically of 30% of the width and height of the image. These transformations would still generate images that you can expect to see in the real world.

I also tried different approaches for augmenting the images: (1) change the number of augmented images to generate, (2) calculate the number of images to augment based on the already existing samples per class to try to even the number of samples per class.

The best results obtained were with the approach 2. However, I also observed that generating too many augmented images the model started producing worse results that with fewer augmented images. One of the possible reasons for this could be is that augmented images might have high correlation between them.

So, I decided to define the number of samples per class to 700, so classes in the training set that had fewer number of samples than 700, I generated the required number of augmented samples until reaching the 700 samples. Those classes that already had more than 700 samples, I would not add new images. With this procedure I tried to even even a bit the number of samples per class in the whole set but without introducing too much correlation between the images.

The total number of augmented images (in the training set) were: 9081.

An example of an augmented image is:

![alt text][image5]

### Design and Test a Model Architecture

As a first step, I decided to convert the image to grayscale as color images might introduce additional information on the model that might not be too much relevant for training the model to classify traffic signs. After converting to grayscale, I also normalized the images: [0-255] - 128) / 128. I converted to grayscale and normalized the images to the three sets (training, validation and test sets).

For the training data, I also shuffled the data as I wanted to prevent an ordering effect of the data on the training of the network.

#### Arquitecture

I started by using the LeNet arquitecture, and pre-processed the images by converting them to grayscale and normalizing the images. With 10 epochs and learning rate of 0.001, and batch size of 128 I obtained a 0.872 validation accuracy.

The first modifcations were tunning the hyperparameters. Some results obtained were as follows (without augmenting the images in the training set):


| Learning rate         		|     Epochs	        					| Batch size | *Validation accuracy*
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:|:---------------------------------------------:|
| 0.001         		| 30   							| 128 | 0.92 |
|0.001   | 100   | 128    | 0.93x  |
|0.01   | 100   | 128   | 0.05   |
|0.0005   | 200    | 128   | 0.93x   |
|0.001   | 100   | 256  | 0.92   |
|0.001   | 300   | 64   | 0.94   |

After playing with some hyperparameters, seemed that the best results were obtained with epoch = 300, and smaller batch size (64).

Afterwards I decided to start adding the augmented samples in the training set. Some observations I made: (1) only augmenting the samples with translation did not help in increasing the validation accuracy (maybe because this can generate more correlation between images), (2) changing the brightness did not help in increasing the validation accuracy, (3) augmenting all the classes similarly, i.e. adding the same number of images to each class did not help in increasing the validation accuracy (I tried to add additional 3000 samples in each class, I did get < 0.9 in validation accuracy).

After doing some tests I decided I would only use the following operations when augmenting the images: rotation, translation and zooming.

Based on point 3 mentioned beforehand, I decided that I might need to balance the data. That is why I decided that the number of augmented images per class should depend on the already existing number of samples per class. First I tried with making that each class has a minimum of 1500 images, and I got a validation accuracy of 0.909. A lower value such as 600 minimum images per class, I got better results: validation accuracy of 0.938. That means that I needed to be careful on the number of augmented images (adding too many augmented images can worsed the model during the training).

 After doing a few tests, and changing the hyperparameters, and the number of minimum images per class, I realised was hard to get a validation accuracy over 0.94. At that point I decided to try other techniques to improve the model, such as the regularization.

I implemented a dropout regularization technique by putting a dropout layer after each activation function. First I set a dropout with a probability of keeping the neuron of 0.5, but then I realised that a higher probability such as 0.7 I obtained better validation accuracy (I obtained a validation accuracy of 0.95 with a probability of 0.7). Note that the dropout was for the training stage. For the evaluation stage, I set the probability to 1.0 (no dropout).

To summarise, the decisions and observations made to obtain the final model were as follows:
- Not all type of augmentation operations helped in improve the model. For example, altering the brightness did not help in improving the model.
- It is good to use image augmentation in order to even the number of samples per class, but it is important not too use it too much because this can cause some problems in the training.
- Dropout really helps in improving the model so it was added to the existing model arquitecture, and that the probability of keeping neurons can be important. This helped prevent overfitting.

The final model arquitecture implement was the following one:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| DROPOUT   | 0.7 probability   |
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x6      									|
| RELU					|												|
| DROPOUT   | 0.7 probability   |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| Outputs: 400        									|
| Fully connected		| Input: 400, Output: 120        									|
| RELU					|												|
| DROPOUT   | 0.7 probability   |
| Fully connected		| Input: 120, Output: 84        									|
| RELU					|												|
| DROPOUT   | 0.7 probability   |
| Fully connected	(logits)	| Input: 84, Output: 43        									|


And the parameters used for this model were (after doing set of testings described beforehand):
- Epochs: 100.
- Learning rate: 0.001.
- Batch size: 64.
- Probability dropout: 0.7.
- Augmenting images: rotation range=15,zoom_range=0.2, width_shift_range=0.3, height_shift_range=0.3.
- Minimum number of samples per class: 700 (used for augmenting the images).
- Use Adam optimizer in order to minimise the loss function.

The final model results were:
* validation set accuracy of 0.95.
* test set accuracy of 0.934.

The test accuracy of 0.934 shows that the model is working well with images that have not seen before.


### Test the model on new images

I have chosen five traffic signs I found on the web:
Bumpy road sign:
![alt text][image6]
Turn right ahead sign:
![alt text][image7]
Stop sign:
![alt text][image8]
No entry sign:
![alt text][image9]
Speed limit 70 sign:
![alt text][image10]

The quality that probably might cause more difficulty in classiy is the text as there is not many signs that have text in it. So probably the Stop sign might be the more difficult to classify.

The predictions for each of the images were as follows:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Turn right ahead sign     			| Turn right ahead sign 										|
| Stop sign					| Speed limit 50 km/h											|
| Bumpy road sign      		| Bumpy road sign   									|
| No entry sign	      		| No entry sign					 				|
| Speed limit 70 sign			| Speed limit 70 sign      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This gives a lower percentage compared to the test set (0.934). This could be expected as I only tested with 5 new images and only getting one prediction wrong, this would decrease  significantly the accuracy on the new images.

Looking at the softmax 5 top probabilities for each prediction (see notebook), we can see that the model was quite sure about the last three images (bumpy road sign, no entry sign and speed limit 70 sign), with a probability of >0.99. Although the turn right ahead was predicted correctly, the probability was quite close (0.387) with the second highest probability, which is the Stop sign (0.312). Interestingly, the stop sign was predicted incorrectly, and the this sign was not in the first five top probabilities.

The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .387         			| Turn right ahead sign   									|
| .62     				| Stop sign 										|
| .99					| Bumpy road sign											|
| .99	      			| No entry sign					 				|
| .99				    | Speed limit 70 sign      							|
