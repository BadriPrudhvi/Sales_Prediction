library(plyr)
library(dplyr)
library(lme4)
library(lubridate)
library(caret)
library(VIM)
library(dummies)
library(xgboost)
library(RANN)
library(rpart)
library(DMwR)

##### Reading Data #####
train_file <- read.csv("~/Documents/Analytics_Vidhya/Sales_Prediction/Sales_Prediction/Train.csv")
test_file <- read.csv("~/Documents/Analytics_Vidhya/Sales_Prediction/Sales_Prediction/Test.csv")

##### Preprocessing Data #####

test_file$Item_Outlet_Sales <- 1
Full_Data <- rbind(train_file,test_file)

Full_Data$Item_Fat_Content <- revalue(Full_Data$Item_Fat_Content,
                                    c("LF" = "Low Fat","low fat" = "Low Fat", "reg" = "Regular"))

Full_Data$Item_Fat_Content <- ifelse(Full_Data$Item_Fat_Content == "Low Fat",1,0)

levels(Full_Data$Outlet_Size)[1] <- "Other"

Full_Data$Item_Visibility <- ifelse(Full_Data$Item_Visibility == 0 ,
                                    median(Full_Data$Item_Visibility,na.rm = T),
                                    Full_Data$Item_Visibility)

q <- substr(Full_Data$Item_Identifier,1,2)
Full_Data$Item_Type_Derived <- ifelse( q == "FD","FOOD",
                                       ifelse( q == "DR", "Drinks","Non-Consumable"))
Full_Data$Item_Type_Derived <- as.factor(Full_Data$Item_Type_Derived)

##### Impute Missing Values ######

anova_mod <- rpart(Item_Weight ~ . -Item_Outlet_Sales, 
                   data=Full_Data, 
                   method="anova", 
                   na.action=na.omit)
Item_Weight_pred <- predict(anova_mod, Full_Data[is.na(Full_Data$Item_Weight), ])
Full_Data$Item_Weight[is.na(Full_Data$Item_Weight)] <- Item_Weight_pred

##### Creating dummy variables #####

Full_Data <- dummy.data.frame(Full_Data, names = c('Outlet_Size','Outlet_Location_Type',
                                                   'Outlet_Type','Item_Type_Derived'),  sep='_')

##### Feature Engineering #####
Full_Data <- Full_Data %>%
  group_by(Outlet_Identifier) %>%
  dplyr::mutate(Outlet_ID_Count = n()) 

Full_Data <- Full_Data %>%
  group_by(Item_Identifier) %>%
  dplyr::mutate(Item_ID_Count = n()) 

Full_Data <- Full_Data %>%
  mutate(Time_Difference = 2013 - Outlet_Establishment_Year)

####### Splitting Data in Train and Test Sets ######

Full_Data[,c("Item_Identifier","Item_Type","Outlet_Identifier","Outlet_Establishment_Year")] <- NULL
train_data <- Full_Data[1:nrow(train_file),]
test_data <- Full_Data[-(1:nrow(train_file)),]
test_data$Item_Outlet_Sales <- NULL
#### Removing Outliers ####

# train_data <- subset(train_data, Item_Outlet_Sales < quantile(train_data$Item_Outlet_Sales,0.75,na.rm = T) + 1.5*IQR(train_data$Item_Outlet_Sales,na.rm = T))

#### Building Models ####
control <- trainControl(method = "cv",
                        number = 10,
                        verboseIter = TRUE)

# Fit GBM
modelGbm <- train(Item_Outlet_Sales~., 
                  data=train_data, 
                  method="gbm", 
                  trControl=control, 
                  # preProcess = c("center","scale"),
                  verbose= TRUE)
print(modelGbm)
plot(modelGbm)
gbmPred <- predict(modelGbm, test_data)
gbm_output <- data.frame(Item_Identifier = test_file$Item_Identifier,Outlet_Identifier = test_file$Outlet_Identifier,Item_Outlet_Sales = gbmPred)
write.csv(gbm_output,row.names = FALSE,"~/Documents/Analytics_Vidhya/Sales_Prediction/Sales_Prediction/gbm_output.csv")

# Fit random forest
modelRF <- train(Item_Outlet_Sales~., 
                 data=train_data, 
                 tuneLength = 3,
                 method = "ranger",
                 tuneGrid = data.frame(mtry=c(1,2,3,7,9,15)),
                 trControl = control)

# Print model to console
print(modelRF)
plot(modelRF)
RFPred <- predict(modelRF, test_data)
RF_output <- data.frame(Item_Identifier = test_file$Item_Identifier,
                        Outlet_Identifier = test_file$Outlet_Identifier,
                        Item_Outlet_Sales = RFPred)
write.csv(RF_output,row.names = FALSE,"~/Documents/Analytics_Vidhya/Sales_Prediction/Sales_Prediction/RF_output.csv")

# Fit parallel random forest
modelPRF <- train(Item_Outlet_Sales~., 
                 data=train_data, 
                 tuneLength = 3,
                 method = "parRF",
                 tuneGrid = data.frame(mtry=c(1,2,3,7,9,15)),
                 trControl = control)

# Print model to console
print(modelPRF)
plot(modelPRF)
PRFPred <- predict(modelPRF, test_data)
PRF_output <- data.frame(Item_Identifier = test_file$Item_Identifier,
                        Outlet_Identifier = test_file$Outlet_Identifier,
                        Item_Outlet_Sales = PRFPred)
write.csv(PRF_output,row.names = FALSE,"~/Documents/Analytics_Vidhya/Sales_Prediction/Sales_Prediction/PRF_output.csv")


## Fit XGBOOST

train_xgb <- train_data

xgb_params_1 = list(
  objective = "reg:linear",                                               
  eta = 0.01,                                                                  # learning rate
  max.depth = 6,                                                               # max tree depth
  eval_metric = "rmse"                                                          # evaluation/loss metric
)

# cross-validate xgboost to get the accurate measure of error
xgb_cv_1 = xgb.cv(params = xgb_params_1,
                  data = as.matrix(train_xgb %>%
                                     select(-Item_Outlet_Sales)),
                  label = train_xgb$Item_Outlet_Sales,
                  nrounds = 1000, 
                  nfold = 10,                                                   # number of folds in K-fold
                  prediction = TRUE,                                           # return the prediction using the final model 
                  verbose = TRUE
)

nround = which(xgb_cv_1$dt$test.rmse.mean == min(xgb_cv_1$dt$test.rmse.mean))[1]
xgb_cv_1$dt[nround,]

XGBModel <- xgboost(param=xgb_params_1, 
                    data = as.matrix(train_xgb %>%
                                       select(-Item_Outlet_Sales)),
                    label = train_xgb$Item_Outlet_Sales,
                    nrounds=nround)

xgb_predict <- predict(XGBModel, as.matrix(test_data))
xgb_output <- data.frame(Item_Identifier = test_file$Item_Identifier,
                         Outlet_Identifier = test_file$Outlet_Identifier,
                         Item_Outlet_Sales = xgb_predict)
write.csv(xgb_output,row.names = FALSE,"~/Documents/Analytics_Vidhya/Sales_Prediction/Sales_Prediction/xgb_output.csv")
var.names = colnames(train_data)

Imp <- xgb.importance(feature_names = var.names, model = XGBModel)
Imp[1:15]


#********************************************************
#********************************************************
#********************************************************

# Create model_list
model_list <- list(gbm = modelGbm, rf = modelRF, prf = modelPRF)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)
bwplot(resamples,metric="RMSE")
varImp(modelGbm)


