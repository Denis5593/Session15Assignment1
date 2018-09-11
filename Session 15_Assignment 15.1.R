# Use the below given data set
# a. Predict the no of comments in next H hrs
# b. Use regression technique
# c. Report the training accuracy and test accuracy

library(data.table)
library(MatrixModels)

library(e1071)
library(FNN)
library(glmnet)
library(ranger)
library(xgboost)

# load and combine dataset
train = fread("BlogFeedback-Train.csv")
test = fread("BlogFeedback-Test.csv")

# error measure
mse = function(y_hat, y) {
  mse = mean((y - y_hat)^2)
  
  return(mse)
}

# create design matrices
train_x = model.Matrix(V281 ~ . - 1, data = train, sparse = F)
train_x_sparse = model.Matrix(V281 ~ . - 1, data = train, sparse = T)
train_y = train$V281

test_x = model.Matrix(V281 ~ . - 1, data = test, sparse = F)
test_y = test$V281

train_xgb = xgb.DMatrix(data = as.matrix(train_x), label = train_y)
test_xgb = xgb.DMatrix(data = as.matrix(test_x), label = test_y)