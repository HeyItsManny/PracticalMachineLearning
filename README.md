---
title: "Practical Machine Learning - Weight Lifting Execution"
author: "Manny Ruiz"
date: "October 13, 2019"
output: 
  html_document: 
    keep_md: yes
---


#### Introduction 
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively.
In this exercise participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
We will use accelerometer data to predict if a participant did the exercise correctly or incorrectly. Classe A represents correct exercise form and classe B,C,D,E represent common mistakes. 
More information on the Human Activity Recognition project is available from the website http://groupware.les.inf.puc-rio.br/har 


#### Analysis
In order to predict the exercise method, we load a training dataset into a data frame. Standard data cleansing is performed by removing unneeded columns, columns with near zero variance and columns with no data. For testing accuracy we partition the training dataset at 75% training and 25% testing and use 4 fold Cross Validation.
Three models were trained.  1. Gradient Boosting, 2. Naive Bayes and 3. Random Forest. The trained models were then used to preduct outcome using the paritioned testing set. The output accuracy of the prediction test were then compared to select the model with the highest accuracy.


#### Conclusion
Results from confusionMatrix showed that random forest is the best model at predicting the execution manner in which participants performed the dumbell exercises.

----
----



#### Data Staging

```r
#Load Libraries and Set Random Sample Seed

library(caret)
set.seed(1015)

# Given Dataset 
# Source Data: http://groupware.les.inf.puc-rio.br/har
# Training Data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
# Test Data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

#Load staging data
training_get <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")

# A review of the data shows many devices (columns) with no data. 
# We will remove these columns from our dataset

training_na <- training_get[,(colSums(is.na(training_get)) == 0)]
# cleanTest <-test[,(colSums(is.na(test)) == 0)]

# The first five columns are unneeded for our modeling. we will remove these columns 
training_rm <- training_na[,-c(1:5)]

# We will remove any predictors with near zero variance
training_nz <- training_rm[, -nearZeroVar(training_rm)] 

# To finalize our dataset we will partition the training data at 75 25 to create a test dataset for cross validation 

inTrain <- createDataPartition(y=training_nz$classe, p=0.75, list=FALSE)
training <- training_nz[inTrain, ]
testing  <- training_nz[-inTrain, ]

dim(training); dim(testing)
```

```
## [1] 14718    54
```

```
## [1] 4904   54
```

```r
# the training dataset contains 14718 observations and the testing dataset contains 4904 observations. 
```

----

#### Data Modeling

```r
# Before we begin to train our model we setup parallel processing 
library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

# I will test 3 models 1. Gradient Boosting, 2. Naive Bayes and 3. Random Forest

model_gbm <- train(classe ~., method = "gbm", data = training, trControl = trainControl(method = "cv", number = 4))
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2395
##      2        1.4559             nan     0.1000    0.1677
##      3        1.3499             nan     0.1000    0.1308
##      4        1.2684             nan     0.1000    0.0988
##      5        1.2055             nan     0.1000    0.0989
##      6        1.1440             nan     0.1000    0.0883
##      7        1.0903             nan     0.1000    0.0731
##      8        1.0445             nan     0.1000    0.0807
##      9        0.9958             nan     0.1000    0.0550
##     10        0.9613             nan     0.1000    0.0461
##     20        0.7095             nan     0.1000    0.0326
##     40        0.4597             nan     0.1000    0.0165
##     60        0.3276             nan     0.1000    0.0058
##     80        0.2504             nan     0.1000    0.0051
##    100        0.1958             nan     0.1000    0.0039
##    120        0.1565             nan     0.1000    0.0040
##    140        0.1242             nan     0.1000    0.0013
##    150        0.1122             nan     0.1000    0.0011
```

```r
model_nb  <- train(classe ~., method = "nb",  data = training, trControl = trainControl(method = "cv", number = 4))

model_rf  <- train(classe ~., method = "rf",  data = training, trControl = trainControl(method = "cv", number = 4))

# We now predict against the test set

test_gbm <- predict(model_gbm, testing)
test_nb  <- predict(model_nb,  testing)
test_rf  <- predict(model_rf,  testing)

#Stop Cluster
stopCluster(cl)

cm_gbm <- confusionMatrix(test_gbm, testing$classe)
cm_nb  <- confusionMatrix(test_nb,  testing$classe)
cm_rf  <- confusionMatrix(test_rf,  testing$classe)

cm_gbm$overall['Accuracy']; cm_nb$overall['Accuracy']; cm_rf$overall['Accuracy']
```

```
##  Accuracy 
## 0.9885808
```

```
##  Accuracy 
## 0.7561175
```

```
##  Accuracy 
## 0.9985726
```

```r
#Lets also print the OOB estimated error for Random Forest
model_rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.21%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4183    1    0    0    1 0.0004778973
## B    8 2837    2    1    0 0.0038623596
## C    0    4 2562    1    0 0.0019477990
## D    0    0    9 2402    1 0.0041459370
## E    0    0    0    3 2703 0.0011086475
```

The confusion matrix results show random forest is most accurate with Accuracy of 99.85%  
Gradient Boosting has an accuracy of 98.85%.
Trailing is Naive Bayes with accuracy of 75.61%

Random Forest final Model shows an OOB estimate of error rate of .21%

----

We now execute the winning model againt the source data test set. The Course Project Prediction Quiz grading will provide the accuracy results for our solution.


```r
source_test <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
dim(source_test)
```

```
## [1]  20 160
```

```r
quiz_rf <- predict(model_rf, source_test)

quiz_rf
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

----
