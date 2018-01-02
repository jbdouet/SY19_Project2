#image(matrix(unlist(I1),ncol=70, byrow = TRUE), col = gray(0:255 / 255))

################################################################################################
#########################################CHARACTER##############################################
################################################################################################
################################################################################################
library(caret)
library(car)
library("e1071")
library(randomForest)
library(kernlab)
library(stats)
library(MASS)


character_data <- read.csv("data/characters_train.txt", sep =" ")
character <- read.csv("data/characters_train.txt", sep =" ")

n=nrow(character)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
character.test<-character[-train,]
character.train<-character[train,]


#Random forest avec CV OKLM
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
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
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- character[-test_i, ]
  test_xy <- character[test_i, ]
  print(k)
  svm_train <- svm(Y ~ ., data = train_xy)
  pred_svm<-predict(svm_train, newdata = test_xy, type = "response")
  prop.table(table(test_xy$Y,pred_svm))
  cm= as.matrix(table(test_xy$Y,pred_svm))
  CV[k]<- sum(diag(cm)) / sum(cm)
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.913


#Naive Bayes + double CV
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- character[-test_i, ]
  test_xy <- character[test_i, ]
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

#LDA
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre � la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le m�me nombre d'�l�ments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent �tre les m�mes car c'est le m�me dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une id�e finale de la performance sur le test
  train_xy <- character[-test_i, ]
  test_xy <- character[test_i, ]
  print(k)
  model_lda <- train(train_xy[,-1],train_xy$Y,method='lda',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_lda<-predict.train(object=model_lda,test_xy[,-1])
  cf<-confusionMatrix(predictions_lda,test_xy$Y) 
  CV[k]<- cf$overall["Accuracy"]
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.70


#QDA
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre � la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le m�me nombre d'�l�ments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent �tre les m�mes car c'est le m�me dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une id�e finale de la performance sur le test
  train_xy <- character[-test_i, ]
  test_xy <- character[test_i, ]
  print(k)
  model_lda <- train(train_xy[,-1],train_xy$Y,method='qda',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_lda<-predict.train(object=model_lda,test_xy[,-1])
  cf<-confusionMatrix(predictions_lda,test_xy$Y) 
  CV[k]<- cf$overall["Accuracy"]
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.88


#RDA
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
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

parole <- read.csv("data/parole_train.txt", sep =" ")


n=nrow(parole)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
parole.test<-parole[-train,]
parole.train<-parole[train,]

#Random forest avec CV OKLM
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
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
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- parole[-test_i, ]
  test_xy <- parole[test_i, ]
  print(k)
  
  
  svm_train <- svm(y ~ ., data = train_xy)
  pred_svm<-predict(svm_train, newdata = test_xy, type = "response")
  
  #pc <- prcomp(train_xy[,-257], center = TRUE)
  #mydata <- data.frame(Species = train_xy[, 257], pc$x)
  #t<-svm(Species~PC1:PC2, data = mydata)
  #test.p <- predict(pc, newdata = test_xy[, -257])
  #pred_svm <- predict(t, newdata = data.frame(test.p), type = "response")
  
  
  prop.table(table(test_xy$y,pred_svm))
  cm= as.matrix(table(test_xy$y,pred_svm))
  CV[k]<- sum(diag(cm)) / sum(cm)
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.924


#Naive Bayes + double CV
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
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
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
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

#QDA
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre � la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le m�me nombre d'�l�ments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent �tre les m�mes car c'est le m�me dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une id�e finale de la performance sur le test
  train_xy <- parole[-test_i, ]
  test_xy <- parole[test_i, ]
  print(k)
  model_lda <- train(train_xy[,-257],train_xy$y,method='qda',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
predictions_lda<-predict.train(object=model_lda,test_xy[,-257])
cf<-confusionMatrix(predictions_lda,test_xy$y) 
CV[k]<- cf$overall["Accuracy"]
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.66

#LDA
n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre � la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le m�me nombre d'�l�ments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent �tre les m�mes car c'est le m�me dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une id�e finale de la performance sur le test
  train_xy <- parole[-test_i, ]
  test_xy <- parole[test_i, ]
  print(k)
  model_lda <- train(train_xy[,-257],train_xy$y,method='lda',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_lda<-predict.train(object=model_lda,test_xy[,-257])
  cf<-confusionMatrix(predictions_lda,test_xy$y) 
  CV[k]<- cf$overall["Accuracy"]
}
CVerror= sum(CV)/length(CV)
CV
CVerror#0.917

n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- parole[-test_i, ]
  test_xy <- parole[test_i, ]
  print(k)
  pc <- prcomp(train_xy[,-257], center = TRUE)
  mydata <- data.frame(Species = train_xy[, 257], pc$x)
  t<-lda(Species~PC1:PC2, data = mydata)
  test.p <- predict(pc, newdata = test_xy[, -257])
  pred_svm <- predict(t, newdata = data.frame(test.p), type = "response")
  
  
  prop.table(table(test_xy$y,pred_svm$class))
  cm= as.matrix(table(test_xy$y,pred_svm$class))
  CV[k]<- sum(diag(cm)) / sum(cm)
}
CVerror= sum(CV)/length(CV)
CV
CVerror
#LDA PCA 0.56


n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- parole[-test_i, ]
  test_xy <- parole[test_i, ]
  print(k)
  pc <- prcomp(train_xy[,-257], center = TRUE)
  mydata <- data.frame(Species = train_xy[, 257], pc$x)
  t<-qda(Species~PC1:PC2, data = mydata)
  test.p <- predict(pc, newdata = test_xy[, -257])
  pred_svm <- predict(t, newdata = data.frame(test.p), type = "response")
  
  
  prop.table(table(test_xy$y,pred_svm$class))
  cm= as.matrix(table(test_xy$y,pred_svm$class))
  CV[k]<- sum(diag(cm)) / sum(cm)
}
CVerror= sum(CV)/length(CV)
CV
CVerror
#QDA PCA 0.63


#PCA
#Mean normalization en fait pas besoin les donn�es sont deja centr�es
size<- length(parole[,1])
vec<- rep(0,size)
s<-rep(0,size)
for (k in 2:length(parole)){
  vec<- vec + parole[,k-1]
}
moy<- vec/(length(parole)-1)
parole_normalized<-parole
for (k in 2:length(parole)){
  parole_normalized[,k-1]<- parole[,k-1]-moy
}

n=nrow(parole_normalized)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
parole_normalized.test<-parole_normalized[-train,]
parole_normalized.train<-parole_normalized[train,]

plot(1:256,pc$sdev^2,type="b",xlab="Nb. de facteurs",ylab="Val. Propres")
biplot(pc,cex=0.65)

pc<-prcomp(parole.train[,-257], center = TRUE)
sum(100 * (pc$sdev^2)[1:2] / sum(pc$sdev^2))


pc <- prcomp(parole.train[,-257], center = TRUE)
mydata <- data.frame(Species = parole.train[, 257], pc$x)
t<-svm(Species~PC1:PC2, data = mydata)

test.p <- predict(pc, newdata = parole.test[, -257])
pred <- predict(t, newdata = data.frame(test.p), type = "response")
cm= as.matrix(table(parole.test$y,pred))
CV<- sum(diag(cm)) / sum(cm)
CV#0.64 pas convaincant; les 2 premiers facteurs expliquent pas suffisamment la variance => perte de donn�es; si on passe � 20 vecteurs les resultats sont degueux
plot(x = mydata$PC1, y = mydata$PC2, col= mydata$Species)#on voit que certains groupes sont pas differencier avec seulement 2 dimensions

pairs(princomp(parole[,-257])$scores[,1:5], col=parole$y,
      main="Scatterplot apr�s ACP sur les 5 C.P. (classes visibles par couleur)")

car::leveneTest(Y ~., data=character)
