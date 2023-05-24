library(tidyverse)
library(lubridate)
library(fastDummies)
library(xgboost)
library(caret)


# data prep ---------------------------------------------------------------

df <- read.csv("germancredit.txt", sep =" ", header=FALSE)

df <- df %>% 
    dummy_cols(remove_first_dummy = TRUE,
               remove_selected_columns = TRUE) %>% 
    mutate(V21 = if_else(V21==1,1,0))


# xgboost ----------------------------------------------------------------
#make this example reproducible
set.seed(0)

#split into training (80%) and testing set (20%)
parts = createDataPartition(df$V21, p = .8, list = F)
train = df[parts, ]
test = df[-parts, ]

#define predictor and response variables in training set
train_x = data.matrix(train[, -8])
train_y = train[,8]

#define predictor and response variables in testing set
test_x = data.matrix(test[, -8])
test_y = test[, 8]


#define watchlist
watchlist = list(train=xgb_train, test=xgb_test)

#run xboost on training data
xgb <- xgboost(data = as.matrix(train_x), 
               label = as.matrix(train_y), 
               max.depth = 20, 
               eta = 0.2, 
               nthread = 2, 
               nrounds = 100, 
               verbose=2, 
               objective = "binary:logistic")
#prediction on test data
pred <- predict(xgb, as.matrix(test_x))

# get unique threshold values from the predicted values
thr=sort(unique(c(pred,0,1)), decreasing=TRUE)

# set up variables for ROC and AUC
n = length(pred)
y = test_y
pos = length(y[y==1])  # number of positive values
neg = length(y[y==0])  # number of negative values
auc=0
last_tpr = 0
last_tnr = 1
# data frame to store results
res.df = data.frame(Thr=double(), TNR=double(), TPR=double(), Acc=double()) #, AUC=double(), ltnr=double(), ltpr=double())

# capture TNR, TPR, Accuracy, AUC contribution at each threshold from predicted values
for (i in thr){
    pred_round <- as.integer(pred > i) 
    acc = sum(y==pred_round)/n
    tp = sum(y[y==1]==pred_round[y==1])
    tn = sum(y[y==0]==pred_round[y==0])
    tpr = tp / pos
    tnr = tn / neg
    # calc AUC contribution
    if (i<1){
        auc = auc + (last_tpr*(last_tnr - tnr))
    }
    df = data.frame(Thr=i, TNR=tnr, TPR=tpr, Acc=acc) #, AUC=auc, ltnr=last_tnr, ltpr=last_tpr)
    res.df = rbind(res.df, df)
    last_tpr = tpr
    last_tnr = tnr
}
auc
# plot ROC
plot(res.df$TNR, res.df$TPR, type='l', xlim=c(1.002,0), ylim=c(0,1.002), 
     yaxs="i", xaxs="i", col='blue', ylab='Sensitivity (TPR)', xlab='Specificity (TNR)', main='ROC curve')
abline(1,-1, col='gray')
legend('center', legend=paste('AUC = ',round(auc,4)), bty='n')

res.df

