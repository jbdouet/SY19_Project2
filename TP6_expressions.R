###### Classifieurs sur les expressions du visage ######  
library(caret)

#### Obtention des données 
data_expressions <- read.csv("data/expressions_train.txt",sep = " ")

X_expressions <- data_expressions[,1:4200]
y_expressions <-data_expressions$y

table(y_expressions)

I<-matrix(as.matrix(X_expressions[14,]),60,70)
I1 <- apply(I, 1, rev)
image(t(I1),col=gray(0:255 / 255))
#image(matrix(unlist(I1),ncol=70, byrow =TRUE ),col=gray(0:255 / 255))

#### Separation train-test

n=nrow(data_expressions)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
data_expressions.test<-data_expressions[-train,]
data_expressions.train<-data_expressions[train,]


############################### Models ############################### 


#################### SVM linear #################### 
#data_expressions$y <- as.factor(as.integer(data_expressions))

n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- data_expressions[-test_i, ]
  test_xy <- data_expressions[test_i, ]
  print(k)
  model_svmLinear <- caret::train(train_xy[,1:4200],train_xy$y,method='svmLinear',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_svmLinear<-predict.train(object=model_svmLinear,test_xy[,1:4200])
  cf_svmLinear<-caret::confusionMatrix(data= predictions_svmLinear,reference=test_xy$y) 
  CV[k]<- cf_svmLinear$overall["Accuracy"]
}
CVerror= sum(CV)/length(CV)
CV
CVerror
cf
test_xy$y
predictions_svmLinear
I<-matrix(as.matrix(test_xy[4,1:4200]),60,70)
I1 <- apply(I, 1, rev)
image(t(I1),col=gray(0:255 / 255))


#################### SVM radial #################### 
#data_expressions$y <- as.factor(as.integer(data_expressions))

n_folds <- 1
folds_i <- sample(rep(1:n_folds, length.out = 36)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,n_folds)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- data_expressions[-test_i, ]
  test_xy <- data_expressions[test_i, ]
  print(k)
  model_svmRadial <- caret::train(train_xy[,1:4200],train_xy$y,method='mlp',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_svmRadial<-predict.train(object=model_svmRadial,test_xy[,1:4200])
  cf_svmRadial<-caret::confusionMatrix(data= predictions_svmRadial,reference=test_xy$y) 
  CV[k]<- cf_svmLinear$overall["Accuracy"]
}
CVerror= sum(CV)/length(CV)
CV
CVerror
dim(train_xy)
dim(test_xy)
dim(data_expressions)
test_xy$y
cf
TEST <-confusionMatrix(test_xy$y,predictions_svmRadial)
TEST$
predictions_svmRadial
I<-matrix(as.matrix(test_xy[4,1:4200]),60,70)
I1 <- apply(I, 1, rev)
image(t(I1),col=gray(0:255 / 255))







################## XGBOOST ################## 
