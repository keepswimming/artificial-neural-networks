Webwork 10 
Prolem 1
At its heart, the goal of an artificial neural network (ANN) is to approximate a nonlinear function. Usually we do this using 
combinations of logistic (sigmoidal) functions. In this problem, you'll use a neural network to approximate the function .
a. Start by generating a vector  of values from 0 to 2pi, in increments of .01. Let y=sin(x). 
Combine these columns in a data frame called sim.

x = seq(0, 2*pi, .01)
y = sin(x)
sim = data.frame(x, y)

b. Load the nnet library and set the random seed to 20. Then use the following code to fit a neural network with 0 hidden nodes:

library(nnet)
library(caret)
library(dplyr)
library(ggformula)
set.seed(20)
data_used = sim
ctrl = trainControl(method = "cv", number = 5)
fit0 = train(y ~ x,
             data = data_used,
             method = "nnet",
             tuneGrid = expand.grid(size = 0, decay = 0),
             linout = TRUE,
             skip = TRUE, #tells nnet to include "skip-layer" connections, These are edges that go directly from the input layer to the output layer, without connecting to any hidden nodes. They are necessary when fitting a model with 0 hidden nodes. Occasionally they are also used in models with hidden nodes.
             trace = FALSE,
             preProc = c("center", "scale"),
             trControl = ctrl)
#c.Use the following code to plot  vs.  for the original "data" and for the predicted values based on the model:
  
sim <- sim %>%
  mutate(fit0_pred = predict(fit0))
sim %>%
  gf_point(y ~ x) %>%
  gf_line(fit0_pred ~ x, color = "red")

#problem 2
In the previous problem, we saw that using an ANN with linout = T and no hidden nodes results in a linear model for  as a 
function of the predictors. In fact, assuming that the ANN converges, this model is equivalent to the one we would get using 
lm(y~x)!
a. Now create fit1, an ANN with 1 hidden node, and fit2, an ANN with 2 hidden nodes. (Do not include skip-layer connections.) 
Plot the results.
# do NOT reset the seed after running Problem 1
data_used = sim
ctrl = trainControl(method = "cv", number = 5)
fit1 = train(y ~ x,
             data = data_used,
             method = "nnet",
             tuneGrid = expand.grid(size = 1, decay = 0),
             linout = TRUE,
             trace = FALSE,
             preProc = c("center", "scale"),
             trControl = ctrl)
fit2 = train(y ~ x,
             data = data_used,
             method = "nnet",
             tuneGrid = expand.grid(size = 2, decay = 0),
             linout = TRUE,
             trace = FALSE,
             preProc = c("center", "scale"),
             trControl = ctrl)
sim <- sim %>%
  mutate(fit1_pred = predict(fit1),
         fit2_pred = predict(fit2))
sim %>%
  gf_point(y ~ x) %>%
  gf_line(fit1_pred ~ x, color = "blue", lwd = 2, lty = 2) %>%
  gf_line(fit2_pred ~ x, color = "gold", lwd = 2, lty = 3)

#answer: fit2 gives an excellent fit. fit1 is better than fit0, but not as good as fit2.

#Problem 3
#Now, let's take a closer look at the convergence of fit1 and fit2.

fit1$finalModel$convergence #to check whether the model with 1 hidden node converged. Did this model converge?YES

fit2$finalModel$convergence #Converged? NO

#When an ANN doesn't converge, the first thing to try is:Increasing the value of the maxit argument.

#d. Samir used leave-one-out cross-validation to fit an ANN model twice on the same data. 
#Both models converged. When he used a random seed of 1, he found an RMSE of 0.15. When he used a random seed of 20, 
#he found an RMSE of 0.043. The most likely explanation is: 

#Problem 4
a. Recale that we scale variables by subtracting the mean and dividing by the standard deviation.
b.
x = c(-.5299, -.4971)
weights = c(-3.33, 0.23, -1.61, 1.09, -30.70)

zH1 = weights[1] + sum(weights[2:3] * x)#Compute the value of  at the hidden node for Mr. Owen Harris Braund.
zH1 #Normally, you would get this from fit_titanic$finalModel$wts). 

sigmaH1 = 1/(1+exp(-zH1))#Compute the output of the activation function at the hidden node,
sigmaH1

zOut = weights[4] + sigmaH1 * weights[5] # Compute z at the output node for Mr. Owen Harris Braund.
zOut

1/(1+exp(-zOut))# Compute f(z), the predicted probability of survival, for Mr. Owen Harris Braund.

#Problem 5
In this problem, we will build an artificial neural network to predict whether a credit-card user will default on their credit card debt. 
Load the ISLR library in R, and view a summary of the Default data set.
library(ISLR)
summary(Default)
head(Default)
str(Default)
a. Create a new variable, student01, which equals 1 if the customer is a student, and equals 0 otherwise. 
Add student01 as the last column on the data frame, and remove student from the data frame.
nnet() can work with categorical predictor variables, so this step is not always necessary. However, it will be helpful for the standardization we'll do in the next problem.
Keeping your columns in the same order as the problem expects will help your results be consistent with your classmates' (because random initial weights are assigned based on the order of the columns).
#library(ISLR)
Default <- Default %>%
  mutate(student01 = ifelse(student == "Yes", 1, 0)) %>%
  select(-student)
#b. Which of the following best describes the response variable, default? BINARY
#c. Set the random seed equal to 4 and use caret to perform 5-fold CV for an artificial neural network that models 
#default as a function of the other variables in the data frame. Use 1 hidden node and no weight decay. For purposes 
#of this problem, do not standardize the data. Then use
#summary(predict(fit_default, type = "prob")) to view a summary of the predicted probabilities. What do you notice?
set.seed(4)
data_used = Default

ctrl = trainControl(method = "cv", number = 5)
fit_default = train(default ~.,
                    data = data_used,
                    method = "nnet",
                    #tuneGrid = expand.grid(size = 0, decay = 0),
                    #preProc = c("center", "scale"),
                    #maxit = 5000,
                    trace = FALSE, #suppress printing the cost func q 10 iterations
                    trControl = ctrl)

#fit_species
summary(predict(fit_default, type = "prob")) #Almost all the data points have the same probabilities. 
#?Why is that? ?not standardization?

#Problem 6
c. Set the seed to 4 and perform 5-fold CV for the ANN again. This time, use the preProc argument to center and 
scale the data. Use maxit = 200 to ensure convergence.

set.seed(4)
data_used = Default

ctrl = trainControl(method = "cv", number = 5)
fit_default = train(default ~.,
                    data = data_used,
                    method = "nnet",
                    tuneGrid = expand.grid(size = 0, decay = 0),
                    preProc = c("center", "scale"),
                    maxit = 200,
                    trace = FALSE, #suppress printing the cost func q 10 iterations
                    trControl = ctrl)

fit_default

summary(predict(fit_default, type = "prob")) 

What is the CV accuracy of the resulting model? 0.9733

#Problem 7
a. Use the following code to make a plot of the neural network you created in the previous problem:
library(NeuralNetTools)
par(mar = c(.1,.1,.1,.1))
plotnet(fit_default)
What color is the edge from H1 to O1? What does this tell you?
  
summary(fit_default)# view the weights of each edge    

Positive weights correspond to black edges in the graph.
Negative weights correspond to gray edges in the graph.
The thickness of each edge in the graph corresponds to the magnitude (absolute value) of each weight.

fit_default$finalModel$wts # more decimal places of accuracy in the weights 
#Notice that the weight on the edge from H1 to O1 is negative. This means that if H1 increases, then O1 will 
#decrease, and vice versa. What values of the predictors are associated with decreasing H1?

d. 
For fun, you can add the weights to the graph using the text() function:
text(-.15, .78, round(fit_default$finalModel$wts[1], 2))
text(-.44, .73, round(fit_default$finalModel$wts[2], 2))
text(-.36, .53, round(fit_default$finalModel$wts[3], 2))
text(-.36, .32, round(fit_default$finalModel$wts[4], 2))
text(.68, .75, round(fit_default$finalModel$wts[5], 2))
text(.40, .53, round(fit_default$finalModel$wts[6], 2))

I found these positions by trial and error. You can make this process easier by adding axes to the graph:
par(mar = c(5, 4, 4, 2) + 0.1) # Use the default margins to make the axes visible
plotnet(fit_default)
axis(1, at = seq(-1, 1, by = .1))
axis(2, at = seq(0, 1, by = .1))

#Problem 8
Let's use our model to make categorical predictions about whether each customer will default on their 
credit-card debt.
a. Start by using the predict() function to classify each customer into the most likely category:
In other words, if a customer has a probability of defaulting of .5 or greater, then we predict that they will 
default.

default_class = predict(fit_default) 

#Make the confusion matrix. What is the misclassification rate (that is, the classification error rate) on the 
#training data? (Enter your answer as a proportion between 0 and 1.)
conf_mat = table(predicted = default_class, actual = Default$default)
1-sum(diag(conf_mat))/sum(conf_mat)

#b. Suppose we only want to classify customers as "Yes" (defaulters) or "No" (non-defaulters) if their 
probability of belonging to that class exceeds .8. In other words, if the person's predicted probability of default is greater than .8, we will classify them as "Yes"; if it is less than .2, we will classify them as "No"; and if it is between .2 and .8, we will not make a prediction (NA).
Compute the predicted probabilities.
Add a column to the dataframe of predicted probabilities, called default_class.8.Create a vector of NAs that has the same length as the vector of observed responses, default. Call it default_class.8.
Set default_class.8 equal to "Yes" for all entries for which the predicted probability of "Yes" is > .8.
Set default_class.8 equal to "No" for all entries for which the predicted probability of "No" is > .8.
Set default_class.8 equal to NA_character_ for all other entries.

probabilities = predict(fit_default, type = "prob")
probabilities <- probabilities %>%
  mutate(default_class.8 = case_when(No > .8 ~ "No",
                                     Yes > .8 ~ "Yes",
                                     TRUE ~ NA_character_))

#c. What is the misclassification error rate among observations for which we make a prediction at the .8 
#probability threshold? 
conf_mat = table(predicted = probabilities$default_class.8, actual = Default$default)
1-sum(diag(conf_mat))/sum(conf_mat)

#answer: gives an answer of 0.0142 but it's wrong, answer is 0.01392524


#d. For how many observations do we fail to make a prediction at the .8 probability threshold?
probabilities %>%
group_by(default_class.8) %>%
summarise(n()) #gives an answer of 425 but its wrong, answer is 449

#Problem 9
#Make an Olden plot to assess the importance of the predictor variables in predicting default. Which variable 
#is second-most important?
olden(fit_default)
  
#Problem 10
Up to now, we have only considered artificial neural networks for Default with a single hidden node. In this 
problem, you will use cross-validation to select an optimal number of hidden nodes.
a. Set the random seed to 4. Use caret to perform 5-fold cross-validation to test numbers of hidden nodes 
from 1 to 8. Use the preProc argument to center and scale the data. Use maxit = 200 to ensure convergence.

set.seed(4)
data_used = Default

ctrl = trainControl(method = "cv", number = 5)
fit_default = train(default ~.,
                    data = data_used,
                    method = "nnet",
                    tuneGrid = expand.grid(size = 1:8, decay = 0.5),
                    preProc = c("center", "scale"),
                    maxit = 200,
                    trace = FALSE, #suppress printing the cost func q 10 iterations
                    trControl = ctrl)

fit_default

summary(predict(fit_default, type = "prob")) 
