if("pacman" %in% rownames(installed.packages())==FALSE){install.packages("pacman")}

pacman::p_load("tensorflow","keras","xgboost","dplyr","caret",
               "ROCR","lift","glmnet","MASS","e1071"
               ,"mice","partykit","rpart","randomForest","dplyr"   
               ,"lubridate","ROSE","smotefamily","DMwR","caretEnsemble"
               ,"MLmetrics","beepr","klaR")

data<-read.csv("C:/Users/anuj/Documents/Anuj/MMA/Maching Learning & AI/Kaggle Competition/diabetes_train.csv")
act_test<-read.csv("C:/Users/anuj/Documents/Anuj/MMA/Maching Learning & AI/Kaggle Competition/diabetes_test.csv")
#Count of missing values per column
lapply(data, function(x) sum(is.na(x)))
lapply(act_test, function(x) sum(is.na(x)))

str(data)
str(act_test)

act_test<-act_test[,-1]

act_test$diabetes<-0

act_test$diabetes<-as.factor(act_test$diabetes)


act_test$feat<-ifelse(act_test$BMI>0,round(act_test$age/act_test$BMI,0),0)
act_test$feat<-ifelse(act_test$feat==0,mean(act_test$feat),act_test$feat)
act_test$feat1<-round(act_test$serum_insulin/act_test$plasma_glucose,0)
act_test$feat1<-ifelse(act_test$feat1==0,mean(act_test$feat1),act_test$feat1)
act_test$feat2<-ifelse(act_test$num_times_pregnant>0,round(act_test$triceps_skin/act_test$num_times_pregnant,0),0)
act_test$feat2<-ifelse(act_test$feat2==0,mean(act_test$feat2),act_test$feat2)


table(data$diabetes)


data=data[,-1]

data$diabetes<-as.factor(data$diabetes)

data$feat<-ifelse(data$BMI>0,round(data$age/data$BMI,0),0)
data$feat<-ifelse(data$feat==0,mean(data$feat),data$feat)
data$feat1<-ifelse(data$plasma_glucose>0,round(data$serum_insulin/data$plasma_glucose,0),0)
data$feat1<-ifelse(data$feat1==0,mean(data$feat1),data$feat1)
data$feat2<-ifelse(data$num_times_pregnant>0,round(data$triceps_skin/data$num_times_pregnant,0),0)
data$feat2<-ifelse(data$feat2==0,mean(data$feat2),data$feat2)

str(data)

levels(data$diabetes)

#Data Split

set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime
inTrain <- createDataPartition(y = data$diabetes,
                               p = 0.8, list = FALSE)
training <- data[ inTrain,]
testing <- data[ -inTrain,]

table(training$diabetes)

#Smote Balancing
data_balance<-SMOTE(diabetes~.,data=training,perc.over=100)

table(data_balance$diabetes)

str(data_balance)

data_balance$feat1


#Logistic
logistic_fit<-glm(diabetes~.,data=data_balance,family=binomial(link = "logit"))

summary(logistic_fit)

logisitc_AIC_fit<-stepAIC(logistic_fit,direction = "both",trace=1)

summary(logisitc_AIC_fit)

par(mfrow=c(1,4))
plot(logisitc_AIC_fit) 
par(mfrow=c(1,1))

log_predict<-predict(logisitc_AIC_fit,newdata = testing,type="response")
log_classification<-rep("1",114)
log_classification[log_predict<0.8]="0"
log_classification<-as.factor(log_classification)


###Confusion matrix  
confusionMatrix(log_classification,testing$diabetes,positive = "1") #Display confusion matrix

####ROC Curve
logistic_ROC_prediction <- prediction(log_predict, testing$diabetes)
logistic_ROC <- performance(logistic_ROC_prediction,"tpr","fpr") 
plot(logistic_ROC) 

####AUC (area under curve)
auc.tmp <- performance(logistic_ROC_prediction,"auc") 
logistic_auc_testing <- as.numeric(auc.tmp@y.values) 
logistic_auc_testing

#### Lift chart
plotLift(log_predict, testing$diabetes, cumulative = TRUE, n.buckets = 10) 

write.csv(log_classification,'C:/Users/anuj/Documents/Anuj/MMA/Maching Learning & AI/Kaggle Competition/op_log3.csv')

F1_Score(log_classification,testing$diabetes)



#Hyperparameter Tuning
control<-trainControl(method="repeatedcv",number=10,repeats=1,search="random")
start_time_rf<-Sys.time()
for(i in c(25,50,100)){
  rf_random<-train(converted_in_7days~.,
                   data=data_balance,ntree=i,method="rf",metric="Accuracy",tuneLength=3,trControl=control)
  print(i)
  print(rf_random)
  plot(rf_random)
}
end_time_rf<-Sys.time()

#-----Random Forest(Enter the ntree and mtry based on the results of Cross Validation)
model_forest <- randomForest(diabetes~., data=data_balance, 
                             type="classification",
                             importance=TRUE,
                             ntree = 25,                                    
                             mtry = 4,                                          
                             nodesize = 10,         
                             maxnodes = 10,         
                             cutoff = c(0.5, 0.5)   
) 

beep(8)
plot(model_forest)  
varImpPlot(model_forest) 

###Finding predicitons: probabilities and classification
forest_probabilities<-predict(model_forest,newdata=testing,type="prob") 
forest_classification<-rep("1",114)
forest_classification[forest_probabilities[,2]<0.8]="0" 
forest_classification<-as.factor(forest_classification)

confusionMatrix(forest_classification,testing$diabetes, positive="1") 



####ROC Curve
forest_ROC_prediction <- prediction(forest_probabilities[,2], testing$diabetes) 
forest_ROC <- performance(forest_ROC_prediction,"tpr","fpr") 
plot(forest_ROC) 

####AUC (area under curve)
AUC.tmp <- performance(forest_ROC_prediction,"auc") 
forest_AUC <- as.numeric(AUC.tmp@y.values) 
forest_AUC 

#### Lift chart
plotLift(forest_probabilities[,2],  testing$converted_in_7days, cumulative = TRUE, n.buckets = 10)

Lift_forest <- performance(forest_ROC_prediction,"lift","rpp")
plot(Lift_forest)

#xgBoost
set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime
inTrain1 <- createDataPartition(y = data_balance$diabetes,
                                p = 0.8, list = FALSE)
training1 <- data_balance[ inTrain1,]
testing1 <- data_balance[ -inTrain1,]


data_matrix<-data.matrix(dplyr::select(data_balance,-diabetes))
data_matrix1<-data.matrix(dplyr::select(data,-diabetes))


x_train <- data_matrix[ inTrain1,]
x_test <- data_matrix1[ -inTrain,]

y_train <-training1$diabetes
y_test <-testing$diabetes

#xgBoost Hyper-parameter tuning
control<-trainControl(method="repeatedcv",
                      number=5,repeats=1,search="random")

start_time_xg<-Sys.time()

xg_Boost<-train(y=y_train,
                x=x_train,method="xgbTree",metric="Accuracy",
                tuneLength=3,trControl=control)

end_time_xg<-Sys.time()
beep(8)

#Print summary of all Cross validations
print(xg_Boost)

#Print hyperparameters best tuned model
xg_Boost$bestTune

#Calculate Probability and build confusion matrix
XGboost_prediction<-predict(xg_Boost,newdata=x_test,type="prob") 
confusionMatrix(as.factor(ifelse(XGboost_prediction[,2]>0.8,1,0)),y_test,positive="1") 
xg<-as.factor(ifelse(XGboost_prediction[,2]>0.8,1,0))

xg_pred<-as.factor(ifelse(XGboost_prediction[,2]>0.8,1,0))

write.csv(xg_pred,'C:/Users/anuj/Documents/Anuj/MMA/Maching Learning & AI/Kaggle Competition/op2.csv')

####ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction[,2], y_test) 
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr") 
plot(XGboost_ROC_testing) #Plot ROC curve

####AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") 
XGboost_auc_testing <- as.numeric(auc.tmp@y.values)
XGboost_auc_testing 

F1_Score(xg,y_test)

#### Lift chart
plotLift(XGboost_prediction, y_test, cumulative = TRUE, n.buckets = 10)

Lift_XGboost <- performance(XGboost_ROC_prediction,"lift","rpp")
plot(Lift_XGboost)

