#image(matrix(unlist(I1),ncol=70, byrow = TRUE), col = gray(0:255 / 255))

################################################################################################
#########################################CHARACTER##############################################
################################################################################################
################################################################################################
library(caret)
library("e1071")
library(randomForest)


character <- read.csv("characters_train.txt", sep =" ")


n=nrow(character)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
character.test<-character[-train,]
character.train<-character[train,]


#Random forest avec CV OKLM
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- character.train[-test_i, ]
  test_xy <- character.train[test_i, ]
  print(k)
  rf <- randomForest(Y ~ ., data = train_xy)
  pred_rf<-predict(rf, newdata = test_xy, type = "response")
  prop.table(table(test_xy$Y,pred_rf))
  cm= as.matrix(table(test_xy$Y,pred_rf))
  CV[k]<- sum(diag(cm)) / sum(cm)
}
CVerror= sum(CV)/length(CV)
CV
CVerror# 0.9343

#SVM +CV
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- character.train[-test_i, ]
  test_xy <- character.train[test_i, ]
  print(k)
  svm_train <- svm(Y ~ ., data = train_xy)
  pred_svm<-predict(svm_train, newdata = test_xy, type = "response")
  prop.table(table(test_xy$Y,pred_svm))
  cm= as.matrix(table(test_xy$Y,pred_svm))
  CV[k]<- sum(diag(cm)) / sum(cm)
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.906


#Naive Bayes + double CV
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- character.train[-test_i, ]
  test_xy <- character.train[test_i, ]
  print(k)
  model_naive_bayes <- train(train_xy[,-1],train_xy$Y,method='nb',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_naive_bayes<-predict.train(object=model_naive_bayes,test_xy[,-1])
  cf<-confusionMatrix(predictions_naive_bayes,test_xy$Y) 
  CV[k]<- cf$overall["Accuracy"]
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.69

#RDA
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- character.train[-test_i, ]
  test_xy <- character.train[test_i, ]
  print(k)
  rda <- train(train_xy[,-1],train_xy$Y,method='rda',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  pred_rda<-predict.train(object=rda,test_xy[,-1])
  cf<-confusionMatrix(pred_rda,test_xy$Y) 
  CV[k]<- cf$overall["Accuracy"]
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.87


################################################################################################
###########################################PAROLES##############################################
################################################################################################
################################################################################################

parole <- read.csv("parole_train.txt", sep =" ")


n=nrow(parole)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
parole.test<-parole[-train,]
parole.train<-parole[train,]

#Random forest avec CV OKLM
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- parole.train[-test_i, ]
  test_xy <- parole.train[test_i, ]
  print(k)
  rf <- randomForest(y ~ ., data = train_xy)
  pred_rf<-predict(rf, newdata = test_xy, type = "response")
  prop.table(table(test_xy$y,pred_rf))
  cm= as.matrix(table(test_xy$y,pred_rf))
  CV[k]<- sum(diag(cm)) / sum(cm)
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.918


#SVM +CV
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- parole.train[-test_i, ]
  test_xy <- parole.train[test_i, ]
  print(k)
  svm_train <- svm(y ~ ., data = train_xy)
  pred_svm<-predict(svm_train, newdata = test_xy, type = "response")
  prop.table(table(test_xy$y,pred_svm))
  cm= as.matrix(table(test_xy$y,pred_svm))
  CV[k]<- sum(diag(cm)) / sum(cm)
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.92


#Naive Bayes + double CV
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- parole.train[-test_i, ]
  test_xy <- parole.train[test_i, ]
  print(k)
  model_naive_bayes <- train(train_xy[,-257],train_xy$y,method='nb',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_naive_bayes<-predict.train(object=model_naive_bayes,test_xy[,-257])
  cf<-confusionMatrix(predictions_naive_bayes,test_xy$y) 
  CV[k]<- cf$overall["Accuracy"]
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.88


#RDA
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- parole.train[-test_i, ]
  test_xy <- parole.train[test_i, ]
  print(k)
  rda <- train(train_xy[,-257],train_xy$y,method='rda',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  pred_rda<-predict.train(object=rda,test_xy[,-257])
  cf<-confusionMatrix(pred_rda,test_xy$y) 
  CV[k]<- cf$overall["Accuracy"]
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.921
