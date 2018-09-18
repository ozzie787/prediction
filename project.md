---
title: "Building A Predictive Model For Human Activity Recognition"
author: "Stephen J. Osborne"
output:
     html_document:
          keep_md: yes
          toc: yes
---





## Summary

To build a a prediction algorithm for human activity recognition, a data set by Ugulino *et al.* [1] was utilized to train a C5.0 decision tree using repeated k-fold cross validation. With both accuracy and simplicity of the model in mind, a model using 20 predictors was selected as it provided a balance of these two factors. The selected model showed excellent accuracy when predicting on a previously unseen test set with scores for sensitivity, specificity, recall and precision of at least >0.98 and an out of sample error of approximately 0.7%.

## Libraries Used


```r
library(data.table) # Version 1.11.4
library(car)        # Version 3.0-2
library(caret)      # Verison 3.0-1
library(ggplot2)    # Version 3.0.0
```

## The Data

In this project I use the a data set by Ugulino *et al.* [1] to build a prediction algorithm for human activity recognition. This data was obtained by asking six young, healthy participants to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in one of five different fashions:

+ **Class A** Exactly according to the specification of the Unilateral Dumbbell Biceps Curl
+ **Class B** Throwing their elbows to the front
+ **Class C** Lifting the dumbbell only halfway
+ **Class D** Lowering the dumbbell only halfway
+ **Class E** Throwing their hips to the front

The available measurements were captured from a series of sensors placed on the forearm, upper arm, a belt around the participants lower torso and on the dumbbell itself. The sensors measured acceleration, orientation and magnetic field strength.

### Loading and Partitioning

The data was loaded from its .csv file using `fread()` from the `data.table` package. The values in the variable `classe` were converted to factors using `factor()` for later processing. The names of the columns of interest from a visual inspection of the data frame were extracted into a list using `grep()` to find column names beginning with one from: roll_; pitch_; yaw_; total; gyros; accel_; or mag. The data was split 70%/30% into a training and test set named `train` and `test` respectively using `createDataPartition()` from the caret package with `p=0.7` and the outcome set to the variable `dtrain$classe`. This process can be seen outlined below in R code.


```r
ftrain  <- fread("pml-training.csv")
ftrain$classe <- factor(ftrain$classe)
cols    <- grep("^roll_|^pitch_|^yaw_|^total|^gyros|^accel_|^mag", names(ftrain))
dtrain  <- cbind(classe=ftrain$classe, ftrain[, ..cols])
inTrain <- createDataPartition(y=dtrain$classe, p=0.7, list=FALSE )
train   <- dtrain[inTrain,]
test    <- dtrain[-inTrain,]
```

A visual inspection of the data using `featurePlot()` does not show many clear correlations or differences in the distributions of the data that could be exploited to form a model. However, some exploratory models found the magnetic field data to be important to detect the dumbbell (which is likely made of steel), followed by the data from the gyros and the accelerometers.

In order to obtain an insight into which variables are either important or can be striped away, a general boosted model (gbm) was fitted with `classe` as the outcome and the rest of the data set as the predictors. The gbm model was fitted using `train()` from the caret package with the verbosity of output set to be silent (i.e. `verbose=FALSE`). Using `trainControl()`, `train()` was set to implement 10-fold cross-validation while training the model. The resulting model was placed in a variable named `modfitgbm`.


```r
train_control <- trainControl(method="cv", number = 10)
modfitgbm <- train(classe ~ .,
                 data = train,
                 trControl=train_control,
                 method = 'gbm',
                 verbose = FALSE
                 )
```

The resulting model was tested using the `test` data set using `predict()` from the stats package. The frequency of each predicted activity versus actual activity label was calculated. These frequencies were then expressed as a percentage of the actual frequency of the activity in the `test` data set. This table was plotted as a tile plot using the function `plotter()`. The result of this can be seen in Figure 1.


```r
plotgbm <- plotter(modfitgbm, test, test$classe)
a1 <- accu(modfitgbm, train, train$classe)
a2 <- accu(modfitgbm, test, test$classe)
message(c("Out of sample error on trainning and test set ", a1, "%", " and ", a2, "%"))
```

```
## Out of sample error on trainning and test set 2.72257407002985% and 3.857264231096%
```

![**Figure 1:** Tile plot showing percentage of the activities predicted versus their actual activity label in the `test` data set using the model, `modfitgbm` ](project_files/figure-html/plot_gbm-1.png)

In Figure 1 it can be seen that in general the model predicts the correct activity greater than 95% of the time. The out of sample error was calculated to be 2.7% and 4.0% for the training and test sets respectively. As the model is predicting reasonably well (>90% correct classification), this model is a good starting point to determine the importance of each variable to predicting which activity is being performed. Following the methodology of Ugulino *et al.* [1] where they used a C4.5 decision tree model, I used as a starting point a C5.0 decision tree model. C5.0 was selected over C4.5 due to improvements in speed, memory usage and boosting support.

In order to minimize the number of predictors in the final model versus accuracy the list of relative influences from `modfitgmb` was used as a starting point and placed into a data frame named `inflist`.


```r
inflist <- summary(modfitgbm)
```

A series of models were fitted using the C5.0 decision tree model increasing the number of predictors by 5 each time up to a total maximum of 40. The formula of the predictor was constructed using `paste()`. The right hand side of the formula was formed using `paste()` with `collaspe = " + "` to collapse the list into a single string separated by the specified string.

The C5.0 model was implemented using 10-fold cross validation with 5 repeats and defined in a variable named `train_control`. The C5.0 model was fitted using `train()` from the caret package with the verbosity of output set to be silent (i.e. `verbose=FALSE`). The out of sample error of the outputted model, `modfittemp` is obtained using the imputed function, `accu()` separately with the training and test set. The output of the function was aggregated into a data frame named `errdat`. The outputted model, initially outputted to a variable named, `modfittemp` is reassigned to a variable named modfit*x*, where *i* is the number of predictors included in the model.


```r
errdat <- as.data.frame(matrix(0, ncol = 3, nrow = 0))

for (i in seq(5,40,by = 5)) {
     nmes  <- as.character(inflist$var[1:i])
     modn1 <- "classe ~ "
     modn2 <- paste(nmes, collapse = " + ")
     modn  <- paste(modn1,modn2, sep="")
     assign(paste("rmod",i,sep = ""), modn)

     train_control <- trainControl(method="repeatedcv", number=10, repeats=5)
     
     modfittmp <- train(as.formula(modn),
                      data = train,
                      trControl=train_control,
                      method = 'C5.0',
                      verbose = FALSE
                      )
     
     errtmp1 <- accu(modfittmp, train, train$classe)
     errtmp2 <- accu(modfittmp, test, test$classe)
     
     errdat <- rbind(errdat, c(i, errtmp1, errtmp2))
     assign(paste("modfit",i,sep = ""), modfittmp)
}

colnames(errdat) <- c("Variables_used", "Train", "Test")
```

In order to plot `errdat` the data frame was reshaped to long form using the function, `melt()` from the data.table package with the id column set to `errdat$Variables_used`. The resulting plot can be seen below in Figure 2. 


```r
de1 <- melt(errdat, id="Variables_used", value.name="err")

insetplot <- ggplotGrob(
     ggplot(data=de1, aes(x=Variables_used, y=err, color=variable)) +
     geom_line() +
     geom_point() +
     labs(x="", y="") +
     scale_x_continuous(limits = c(4,41), breaks = seq(0,40,by = 5 )) +
     scale_y_continuous(limits = c(0,0.4), breaks = seq(0,0.35,by = 0.005 )) +
     coord_cartesian(ylim=c(0,0.025)) +
     theme_classic() + theme(legend.position = "none")
)

ggplot(data=de1, aes(x=Variables_used, y=err, color=variable)) +
     geom_line() +
     geom_point() +
     labs(x="Number of Predictors", y="Out of Sample Error / %") +
     scale_x_continuous(limits = c(4,41), breaks = seq(0,40,by = 5 )) +
     scale_y_continuous(limits = c(0,4), breaks = seq(0,4,by = 0.5 )) +
     scale_color_discrete(name="Data Set", breaks=c("Train","Test"),labels=c("Training","Test")) +
     geom_hline(aes(yintercept = 0.5), linetype = "dashed") +
     theme_classic() + theme(legend.position = "bottom") +
     annotation_custom(
     grob = insetplot,
     xmin = 15,
     xmax = Inf,
     ymin = 1,
     ymax = Inf
)
```

![**Figure 2:** Plot showing out of sample error versus the number of predictors in the C5.0 decisson tree when predicitions are carried out using the training and test set. Dashed line at 0.5 denotes a out of sample error of 0.5% to illustrate how the value is converging. Inset plot shows zoomed in section of the plot to show the small out of sample error observed when the model predicts against the training set ](project_files/figure-html/err_plot-1.png)

In Figure 2 it can be seen that even with very few predictors the C5.0 model is better at predicting the activity than the gbm model fitted earlier. With an increasing number of predictors the out of sample error on the testing set decreases until it converges at approximately 0.5% at 25 predictors. The out of sample error on the model with the training set very rapidly converges to 0% which is expected as the data set the model is being tested against, is the data it has been trained with. In the interest of minimizing the number of predictors used while maintaining accuracy I concluded that a final model that uses 20 predictors gives the best balance.

The resulting model was tested using the `test` data set using `predict()` from the stats package. The frequency of each predicted activity versus actual activity label was calculated. These frequencies were then expressed as a percentage of the actual frequency of the activity in the `test` data set. Using the function `plotter()` the percentage accuracy of the selected model, `modfit20` was plotted as a tile plot as seen in Figure 3.


```r
modfit <- modfit20
plotmodtest <- plotter(modfit, test, test$classe)
plotmodtest
```

![**Figure 3:** Tile plot showing percentage of the activities predicted versus their actual activity label in the `test` data set using the model, `modfit20`](project_files/figure-html/c50plot-1.png)

In Figure 3 it can be seen that `modfit20` in general predicts the correct activity with excellent accuracy as illustrated with the stark red upwards, left to right diagonal. However, the model seems to struggle to distinguish B and C from each other, A and D in comparison to A and E, but these misclassifications are rare.

The confusion matrix for `modfit20` (seen below) was calculated using `confusionMatrix()` from the caret package. The model shows excellent selectivity and specificity meaning as required the model can predict the correct activity when tested against the test set, `test` which it is never trained against. The selectivity and specificity of the model is backed up by large F1 scores (>0.98) meaning the model has excellent precision and recall. The out of sample error of the `modfit20` on the test set was calculated using the function, `accu()` and found to be approximately 0.7% which means the model predicts the correct model >99% of the time which is more than acceptable for the intended application.


```r
pred1 <- predict(modfit, test)
cm1 <- confusionMatrix(test$classe, pred1)
cm1$byClass
```

```
##          Sensitivity Specificity Pos Pred Value Neg Pred Value Precision
## Class: A       0.995       1.000          0.999          0.998     0.999
## Class: B       0.992       0.997          0.989          0.998     0.989
## Class: C       0.995       0.997          0.984          0.999     0.984
## Class: D       0.989       1.000          0.998          0.998     0.998
## Class: E       0.999       1.000          0.999          1.000     0.999
##          Recall    F1 Prevalence Detection Rate Detection Prevalence
## Class: A  0.995 0.997      0.285          0.284                0.284
## Class: B  0.992 0.990      0.193          0.191                0.194
## Class: C  0.995 0.990      0.172          0.172                0.174
## Class: D  0.989 0.993      0.165          0.163                0.164
## Class: E  0.999 0.999      0.184          0.184                0.184
##          Balanced Accuracy
## Class: A             0.997
## Class: B             0.995
## Class: C             0.996
## Class: D             0.994
## Class: E             0.999
```

```r
a1 <- accu(modfit, test, test$classe)
message(c("Out of sample error on test set ", a1, "%"))
```

```
## Out of sample error on test set 0.577740016992356%
```

## Conclusions

A model for human activity recognition was successfully constructed using a C5.0 decision tree with 20 predictors. on When tested to predict upon an unseen testing set, he selected model showed excellent sensitivity (>0.985) and specificity (>0.995) and had a out of sample error of approximately 0.7%. The F1 scores showed that the selected model also demonstrated excellent precision and recall on the test data set.

## List of Created Functions

### `accu()`

| Variable  | IN/OUT | Description |
|-----------|--------|-------------|
| `model`   | IN     | Fitted model produced by `train()` |
| `data`    | IN     | Data to predict the outcomes from |
| `outcome` | IN     | Outcome column(s) of data frame |
| `a1`      |   OUT  | Out of sample error expressed as a percentage |

Using `predict()` from the stats package, the input is taken and put through the model and the predictions placed into a variable named `p1`. The out of sample error, `a1` was calculated by one minus the sum of the predictions that match with `p1` in the provided `outcome` data.


```r
accu <- function(model,data,outcome) {
     p1 <- predict(model, data)
     a1 <- (1 - (sum(p1 == outcome)/length(p1)))*100
     a1
}
```

### `plotter()`

| Variable  | IN/OUT | Description |
|-----------|--------|-------------|
| `model`   | IN     | Fitted model produced by `train()` |
| `data`    | IN     | Data to predict the outcomes from |
| `outcome` | IN     | Outcome column(s) of data frame |
| `plot`    |   OUT  | Plot of predicted outcome vs. actual outcome. Values expressed as a percentage and coloured on a blue-red color scale. |

Using `predict()` from the stats package, the input is taken and put through the model and the predictions placed into a variable named `p1`. Two data frames, `d1` and `d2` were constructed using `table()` from the data of the actual activity labels and the predicted activity labels to get the frequencies of the actual and predicted activities. `d1` and `d2` were merged by the frequency of the activity in `data` to form the data frame `d3`. The percentage frequency was calculated and the result plotted in `plot`.


```r
plotter <- function(model,data,outcome) {
     p1 <- predict(model, data)
     d1 <- as.data.frame(table(outcome))
     colnames(d1) <- c("Actual","ActualFreq")
     d2 <- as.data.frame(table(outcome,p1))
     colnames(d2) <- c("Actual","Predicted","Freq")
     d3 <- merge(d1,d2, by="Actual")
     d3$Percent <- d3$Freq/d3$ActualFreq*100
     
     plot <- ggplot() +
          geom_tile(aes(x=Actual, y=Predicted,fill=Percent),data=d3, color="black",size=0.1) +
          labs(x="Actual",y="Predicted") +
          geom_text(aes(x=Actual,y=Predicted, label=sprintf("%.1f", Percent)),data=d3, size=3, colour="black") +
          scale_fill_gradient(low="blue",high="red", limits = (c(0,100)), name="%Accuracy") +
          theme_classic()
     plot
}
```

## References

[1] Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 
