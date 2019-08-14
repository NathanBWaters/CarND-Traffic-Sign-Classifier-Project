----------
# **Traffic Sign Recognition** 

### Data Set Summary & Exploration

#### 1. Overview

I used the numpy library to calculate summary statistics of the traffic
signs data set:

Train shape:  (34799, 32, 32, 3)

Number of training examples = 34799

Number of testing examples = 12630

Image data shape = (32, 32, 3)

Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many of the training images fall into each category

![vis](https://raw.githubusercontent.com/NathanBWaters/CarND-Traffic-Sign-Classifier-Project/master/vis.png.png)

### Design and Test a Model Architecture

#### 1. Preprocessing
Preprocessing simply entailed normalizing the images.  Originally, I didn't grayscale the images because I believed that color is important information for these images in examples where the shape is blurred or skewed.  In practice, the gray-scaled image performed better. 

I tried using two different libraries for image augmentation: 
1) https://github.com/aleju/imgaug
2) https://github.com/mdbloice/Augmentor

I failed to get imgaug working on my PC due to it using a library called Shapely which wouldn't build.  When using Augmentor, it significantly increase training times and hurt the validation accuracy score.  Therefore, I abandoned the two techniques since I was already achieving greater than 93% accuracy without it.

I originally used the normalization technique of centering by 126 and dividing by 126.  However, dividing the numbers by 255.0 also normalized the pixel values between 0 and 1 and allowed me to visualize the images better. 
 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


| Layer         		|  Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input | 32x32x1 Gray-scale image | 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x30 	|
| RELU	 |		 |
| Batch Normalization |     |
| Max pooling	      	| 2x2 stride,  outputs 14x14x30 |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x30     |
| RELU	 |		 |
| Batch Normalization |     |
| Max pooling	      	| 2x2 stride,  outputs 5x5x60 |
| Flatten | outputs 1 x 1500 |
| Dropout|  |
| Fully connected		| output 1x200  |
| Fully connected		| output 1x120  |
| Fully connected		| output 1x43  |
| Softmax	 |   |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:

| Hyperparameter | Value
|:---------------------:|:---------------------------------------------:| 
| EPOCHS | 100
| BATCH_SIZE | 128
| learning rate| 0.001

I used the Adam Optimizer.


#### 4.  Approach for achieving >93% accuracy

My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 95.9%
* test set accuracy of 93.8%

I started off with the classic LeNet architecture that we first tried in the MNIST dataset.  One of the first problems I noticed with the architecture is that that the convolution depth of the network was too low for the new number of classes.  I increased the depth and this improved the accuracy from 85% to 91%.   I increased the depths to absurd numbers like 100 and 300.  These were overkill and only gave marginally improved scores over depths such as 20 and 30.  I also was unable to push such a large model to GitHub.

I then added Batch Normalization to normalize the output coming out of each Max-Pool layer.  This also improved the performance of the model from 91% to 93.4% accuracy.

Finally, I added a Dropout layer after the flatten layer.  During training I used a 0.5 probability for keeping the weights and in validation used a 1.0 probability. This brought the accuracy up to 96%.

### Test a Model on New Images

#### 1. Visualizing New Images

Here are German traffic signs that I found on the web:

![stop](https://raw.githubusercontent.com/NathanBWaters/CarND-Traffic-Sign-Classifier-Project/master/german_signs/sign10_stop_id_14.png)

This stop sign should be straightforward for the model to predict.

![priority_road](https://raw.githubusercontent.com/NathanBWaters/CarND-Traffic-Sign-Classifier-Project/master/german_signs/sign2_priority_road_id_12.png)

The fact that another sign is in the background could confuse the model because it's not expecting a circle for a priority sign.

![roadwork](https://raw.githubusercontent.com/NathanBWaters/CarND-Traffic-Sign-Classifier-Project/master/german_signs/sign4_roadwork_id_25.png)

One concern about the above image is that the 30 speed limit is also in the background which could easily confuse the network

![turn right](https://raw.githubusercontent.com/NathanBWaters/CarND-Traffic-Sign-Classifier-Project/master/german_signs/sign_turn_right_id_33.png)

Very straightforward

![priority_road](https://raw.githubusercontent.com/NathanBWaters/CarND-Traffic-Sign-Classifier-Project/master/german_signs/sign9_keep_right_id_38.png)

This should be straightforward but the edges from the watermark might confuse the model.  Most likely those lines will be lost when the images is downsized to (32x32x1)

![caution](https://raw.githubusercontent.com/NathanBWaters/CarND-Traffic-Sign-Classifier-Project/master/german_signs/sign1_general_caution_id_18.png)

The watermark might cause an extra difficulty.

#### 2. Model's Predictions on New Signs

Here are the results of the prediction:

![caution](https://raw.githubusercontent.com/NathanBWaters/CarND-Traffic-Sign-Classifier-Project/master/probs.png.png)

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.  I'm surprised with how confident the model is with each of its predictions.  I'm only showing the top three, there really isn't even a need to show the top two since the first category got a majority of the confidence score in the softmax output.  I'm a bit disappointed that the Road Work image did not have any confidence that the 30km/h sign was in the image considering there is a 30km/h sign in the background.
