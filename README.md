# Neural_Network_Charity_Analysis


An exercise in Neural Networks and Deep Learning Models using TensorFlow and Pandas libraries in Python to preprocess datasets and create a predictive binary classifier.

## Overview

The purpose of this analysis was to explore and implement neural networks using `TensorFlow` in `Python`. Neural networks is an advanced form of `Machine Learning` that can recognize patterns and features in the dataset. Neural networks are modeled after the human brain and contain layers of neurons that can perform individual computations.

![neural](https://github.com/Adpetfem83/Neural_Network_Charity_Analysis/blob/main/Images/Neural_Network_Image.png..png)

Throughout this module, we learned:

* How to build a basic neural network
* Preprocess/prepare the datasets
* Create a training and testing set
* Measure model accuracy
* Add additional neurons and hidden layers to optimize the model
* Select the best model to use for our dataset

**AlphabetSoup**, a philanthropic foundation is requesting for a mathematical, data-driven solution that will help determine which organizations are worth donating to and which ones are considered "high-risk". In the past, not every donation AlphabetSoup has made has been impactful as there have been applicants that will receive funding and then disappear. **Beks**, a data scientist for AlphabetSoup is tasked with analyzing the impact of each donation and vet the recipients to determine if the company's money will be used effectively. In order to accomplish this request, we are tasked with helping Beks create a binary classifier that will predict whether an organization will be successful with their funding. We utilize `Deep Learning Neural Networks` to evaluate the input data and produce clear decision making results.


## Results

The CSV file contains more than 34,000 organizations that have received funding from Alphabet Soup in the past. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively
* 

### Data Preprocessing

To start with, data needs to be preprocessed in order to compile, train and evaluate the neural network model. For the ***Data Preprocessing*** portion:

* **EIN** and **NAME** columns were removed during the preprocessing stage as these columns added no value.
* We also binned **APPLICATION_TYPE** and categorized any unique values with less that 500 records as "Other"  
* **IS_SUCCESSFUL** column was the target variable.
* The remaining 43 variables were added as the features (i.e. STATUS, ASK_AMT, APPLICATION TYPE, etc.)

### Compiling, Training and Evaluating the Model

After the data was preprocessed, we used the following parameters to ***compile, train, and evaluate the model***:

* The initial model had a total of 5,961 parameters as a result of 43 inputs with 2 hidden layers and 1 output layer. 
  * The first hidden layer had 70 neurons.  
  * The second hidden layer had 40 neurons.
  * The output layer had 1 neuron. 
  * Both the first and second hidden layers were activated using `RELU - Rectified Linear Unit` function. The output layer was activated using the `Sigmoid` function. 

* The target performance for the accuracy rate should be greater than 75%, however, the model that was created only achieved an accuracy rate of 72.56% and loss percentage was 57.40%.
![origin](https://github.com/Adpetfem83/Neural_Network_Charity_Analysis/blob/main/Images/Neural_Optimization.png)
![origin](https://github.com/Adpetfem83/Neural_Network_Charity_Analysis/blob/main/Images/AlphabetSoupCharity.png)

### Attempts to Optimize and Improve the Accuracy Rate

Three additional attempts were made to increase the model's performance by changing features, adding/subtracting neurons and epochs. The results did not show much improvement. 

  * **Optimization Attempt #1:**
    * Binned **INCOME_AMT** column
    * Created 6,641 total parameters, an increase of 680 from the original of 5,961
    * Increased neurons to 80 for the first hidden layer and maintained 40 for the second hidden layer
    * Accuracy almost remained the same from 72.56% to 72.50% with a decrease of 0.08%
    * Loss was reduced by 0.12% from 57.40% to 57.33%
![results](https://github.com/Adpetfem83/Neural_Network_Charity_Analysis/blob/main/Images/Neural_Optimization_1.png)
![results](https://github.com/Adpetfem83/Neural_Network_Charity_Analysis/blob/main/Images/AlphabetSoupCharity_Opt_1.png)
    
  * **Optimization Attempt #2:**
    *  Removed **ORGANIZATION** column
    *  Binned **INCOME_AMT** column
    *  Removed **SPECIAL_CONSIDERATIONS_Y** column from features as it is redundant to **SPECIAL_CONSIDERATIONS_N**
    *  Increased neurons to 100 for the first hidden layer and 50 for the second hidden layer
    *  Created 8,801 total parameters, an increase of 2,840 from the original of 5,961
    *  Accuracy decreased by 0.11% from 72.56% to 72.30%
    *  Loss decreased by 0.77% from original 57.40% to 56.96% 

![results](https://github.com/Adpetfem83/Neural_Network_Charity_Analysis/blob/main/Images/Neural_Optimization_2.png)
![results](https://github.com/Adpetfem83/Neural_Network_Charity_Analysis/blob/main/Images/AlphabetSoupCharity_Opt_2.png)

    
  * **Optimization Attempt #3:**
    *  Binned **INCOME_AMT** and **AFFILIATION** column
    *  Removed **SPECIAL_CONSIDERATIONS_Y** column from features as it is redundant to **SPECIAL_CONSIDERATIONS_N**
    *  Increased neurons to 125 for the first hidden layer and 50 for the second hidden layer
    *  Created 11,101 total parameters, an increase of 5,140 from the original of 5,961
    *  Accuracy increased 0.11% from 72.56% to 72.64%
    *  Loss increased by 1.23% from original 57.40 to 58.11%

![results](https://github.com/Adpetfem83/Neural_Network_Charity_Analysis/blob/main/Images/Neural_Optimization_3.png)
![results](https://github.com/Adpetfem83/Neural_Network_Charity_Analysis/blob/main/Images/AlphabetSoupCharity_Opt_3.png)

## Summary

In summary, the model and various optimizations did not help to achieve the desired result greater than 75%. With the variations of increasing the epochs, removing variables, adding multiple hidden layer (done offline in Optimization attempt #4) and/or increasing/decreasing the neurons, the changes were minimal and negligible. In reviewing other `Machine Learning` algorithms, the results did not prove to be any better. For example, `Random Forest Classifier` had a predictive accuracy rate of 70.80% which is a 2.11% decrease from the accuracy rate of the `Deep Learning Neural Network` model (72.33%). 

In conclusion, Neural Networks are very intricate and would require experience through trial and error or many iterations to identify the perfect configuration to work with this dataset.

## Resources
* **Software:** Python 3.7.9, Anaconda 4.9.2 and Jupyter Notebooks 6.1.4
* **Libraries:** Scikit-learn, Pandas, TensorFlow, Keras
