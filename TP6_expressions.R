###### Classifieurs sur les expressions du visage ######  
library(caret)
set.seed(101)
load("env.Rdata",.GlobalEnv)
load("envJB.Rdata",.GlobalEnv)
#### Obtention des données 
data_expressions <- read.csv("data/expressions_train.txt",sep = " ")

X_expressions <- data_expressions[,1:4200]
y_expressions <-data_expressions$y

table(y_expressions)

I<-matrix(as.matrix(X_expressions[14,]),60,70)
I1 <- apply(I, 1, rev)
image(t(I1),col=gray(0:255 / 255))
#image(matrix(unlist(I1),ncol=70, byrow =TRUE ),col=gray(0:255 / 255))


dim(X_expressions)
dim(X_expressions[complete.cases(X_expressions), ])
dim(na.omit(X_expressions))
tst <-na.omit(unname(X_expressions))
#### Separation train-test

n=nrow(data_expressions)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)

data_expressions.test<-data_expressions[-train,]
data_expressions.train<-data_expressions[train,]


#### Preprocessing ####

# delete zero columns 
X_preprocessed <- X_expressions[, !apply(X_expressions == 0, 2, all)]
data_preprocessed <- data_expressions[, !apply(data_expressions == 0, 2, all)]

#sélections zones expressives du visage 
# Selection zone yeux 
I14<-matrix(as.matrix(X_expressions[14,]),60,70)
I14_eyes<-matrix(as.matrix(I14[301:1260]),nrow = 60,ncol = 16)
Ieyes <- apply(I14_eyes, 1, rev)
image(t(Ieyes),col=gray(0:255 / 255))

# Selection zone bouche 
I14<-matrix(as.matrix(X_expressions[14,]),60,70)
I14_mouth<-matrix(as.matrix(I14[2460:3359]),nrow = 60,ncol = 15)
Imouth <- apply(I14_mouth, 1, rev)
image(t(Imouth),col=gray(0:255 / 255))

par(mfrow = c(1, 2))
image(t(Ieyes),col=gray(0:255 / 255))
image(t(Imouth),col=gray(0:255 / 255))

#Xselected <- cbind(X_expressions[301:1260],X_expressions[2601:3260])
Xselected <- cbind(X_expressions[301:1260],X_expressions[2460:3359])
#parmi les données sélectionnées, on enlève aussi les pixels noirs
X_selproc<- Xselected[, !apply(Xselected == 0, 2, all)]
data_selproc=data.frame(X_selproc,y=y_expressions)

#pca

require(graphics)
prin_comp <- prcomp(X_preprocessed,center=T )

std_dev <- prin_comp$sdev
pr_var <- std_dev^2
pr_var[1:10]
prop_varex <- pr_var/sum(pr_var)

plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

new_data <-  data.frame( prin_comp$x[,1:25],y=y_expressions)


prin_comp_pca <- prcomp(X_selproc, center=T )
dim(X_selproc)
prcomp(X_selproc)
std_dev2 <- prin_comp_pca$sdev
pr_var2 <- std_dev2^2
pr_var2[1:10]
prop_varex2 <- pr_var2/sum(pr_var2)

plot(prop_varex2, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

plot(cumsum(prop_varex2), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

new_data2 <-  data.frame( prin_comp_pca$x[,1:25],y=y_expressions)

pred <- predict(prin_comp_pca, X_selproc[1:50,])
dim(pred)
test <- data.frame( pred[,1:25],y=y_expressions[1:50])
dim(test)
############################### Models ############################### 

#################### Modele a garder #################### 

data_to_use <- new_data2
classifieur_expressions <-caret::train(data_to_use[,1:ncol-1],data_to_use$y,method='rda',trControl=caret::trainControl(
  method = "cv",
  number =10,
  verboseIter = TRUE))


dim(test_xy1)
test_xy <- predict(prin_comp_pca,test_xy1)
predictions_rda<-predict.train(object=classifieur_expressions,test_xy)
predictions_rda
cf_rda<-caret::confusionMatrix(data= predictions_rda,reference=test_xy1$y) 
cf_rda$overall["Accuracy"]
cf_rda
length(predictions_rda)
length(new_data$y[1:25])
dim(new_data[1:25,])

#################### RDA + nouveau pca #################### 

n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  data_to_use <- data_selproc
  ncol <- ncol(data_to_use)
  test_i <- which(folds_i == k)
  train_xy1 <- data_to_use[-test_i, ]
  ytrain <- train_xy1$y
  test_xy1 <- data_to_use[test_i, ]
  ytest <- test_xy1$y
  print(k)
  prin_comp <- prcomp(train_xy1[,1:ncol-1],center=T)
  train_xy<-data.frame( prin_comp$x[,1:25],y=ytrain)
  pred <- predict(prin_comp, test_xy1[,1:ncol-1])
  test_xy <-data.frame( pred[,1:25],y=ytest)  
  npca =25
  ncol= npca+1
  model_rda <- caret::train(train_xy[,1:ncol-1],train_xy$y,method='rda',trControl=caret::trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_rda<-predict.train(object=model_rda,test_xy[,1:ncol-1])
  cf_rda<-caret::confusionMatrix(data= predictions_rda,reference=test_xy$y) 
  CV[k]<- cf_rda$overall["Accuracy"]
}

CVerror= sum(CV)/length(CV)
CV
CVerror # 0.74
#################### RDA + PCA #################### 

n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  data_to_use <- new_data
  ncol <- ncol(data_to_use)
  test_i <- which(folds_i == k)
  train_xy <- data_to_use[-test_i, ]
  test_xy <- data_to_use[test_i, ]
  print(k)
  model_rda <- caret::train(train_xy[,1:ncol-1],train_xy$y,method='rda',trControl=trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_rda<-predict.train(object=model_rda,test_xy[,1:ncol-1])
  cf_rda<-caret::confusionMatrix(data= predictions_rda,reference=test_xy$y) 
  CV[k]<- cf_rda$overall["Accuracy"]
}

CVerror= sum(CV)/length(CV)
CV
CVerror # 0.74

data_to_use <- new_data2
classifieur_expressions <-  caret::train(data_to_use[,1:ncol-1],data_to_use$y,method='rda',trControl=trainControl(
  method = "cv",
  number =10,
  verboseIter = TRUE))

predictions_rda<-predict.train(object=classifieur_expressions,test[,1:25])
cf_rda<-caret::confusionMatrix(data= predictions_rda,reference=test$y) 
cf_rda$overall["Accuracy"]
cf_rda
length(predictions_rda)
length(new_data$y[1:25])
dim(new_data[1:25,])
#################### SVM linear #################### 
#data_expressions$y <- as.factor(as.integer(data_expressions))

n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  data_to_use <- data_preprocessed
  ncol <- ncol(data_to_use)
  test_i <- which(folds_i == k)
  train_xy <- data_to_use[-test_i, ]
  test_xy <- data_to_use[test_i, ]
  print(k)
  model_svmLinear <- caret::train(train_xy[,1:ncol-1],train_xy$y,method='svmLinear',trControl=trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_svmLinear<-predict.train(object=model_svmLinear,test_xy[,1:ncol-1])
  cf_svmLinear<-caret::confusionMatrix(data= predictions_svmLinear,reference=test_xy$y) 
  CV[k]<- cf_svmLinear$overall["Accuracy"]
}

CVerror= sum(CV)/length(CV)
CV
CVerror # 0.74
predictions_svmLinear
test_xy$y
I<-matrix(as.matrix(test_xy[4,1:4200]),60,70)
I1 <- apply(I, 1, rev)
image(t(I1),col=gray(0:255 / 255))



#################### SVM radial #################### 
#data_expressions$y <- as.factor(as.integer(data_expressions))

n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,n_folds)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- data_expressions[-test_i, ]
  test_xy <- data_expressions[test_i, ]
  print(k)
  model_svmRadial <- caret::train(train_xy[,1:4200],train_xy$y,method='svmPoly',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_svmRadial<-predict.train(object=model_svmRadial,test_xy[,1:4200])
  cf_svmRadial<-caret::confusionMatrix(data= predictions_svmRadial,reference=test_xy$y) 
  CV[k]<- cf_svmRadial$overall["Accuracy"]
}
train_xy
cf_svmRadial
CVerror= sum(CV)/length(CV)
CV
CVerror
dim(train_xy)
dim(test_xy)
dim(data_expressions)
test_xy$y
predictions_svmRadial
cf
TEST <-confusionMatrix(test_xy$y,predictions_svmRadial)
predictions_svmRadial
I<-matrix(as.matrix(test_xy[4,1:4200]),60,70)
I1 <- apply(I, 1, rev)
image(t(I1),col=gray(0:255 / 255))




#################### Random Forest #################### 

n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,n_folds)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  data_to_use <- data_expressions
  ncol <- ncol(data_to_use)
  train_xy <- data_to_use[-test_i, ]
  test_xy <- data_to_use[test_i, ]
  print(k)
  model_rf <- caret::train(train_xy[,1:ncol-1],train_xy$y,method='rf',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_rf<-predict.train(object=model_rf,test_xy[,1:ncol-1])
  cf_rf<-caret::confusionMatrix(data= predictions_rf,reference=test_xy$y) 
  CV[k]<- cf_rf$overall["Accuracy"]
}

cf_svmRadial
CVerror= sum(CV)/length(CV)
CV
CVerror

#################### Boosted classification trees #################### 

n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,n_folds)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- data_expressions[-test_i, ]
  test_xy <- data_expressions[test_i, ]
  print(k)
  model_ada <- caret::train(train_xy[,1:4200],train_xy$y,method='ada',trControl= trainControl(
    method = "cv",
    number =10,
    verboseIter = TRUE))
  predictions_ada<-predict.train(object=model_ada,test_xy[,1:4200])
  cf_ada<-caret::confusionMatrix(data= predictions_ada,reference=test_xy$y) 
  CV[k]<- cf_ada$overall["Accuracy"]
}

cf_ada
CVerror= sum(CV)/length(CV)
CV
CVerror
################## XGBOOST ################## 

library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

# Except the y vector, all the feature are numerical so we do not need any transformation there
#sparse_matrix <- sparse.model.matrix(response ~ .-1, data = campaign)
df_X<- X_expressions
sparse_matrix <- sparse.model.matrix(response ~ .-1, data = campaign)
output_vector = df[,response] == "Responder"
################## Keras - CNN ##################
library(keras)
use_backend(backend = "tensorflow")
use_condaenv("r-tensorflow",required = TRUE )

# Input image dimensions
img_rows <- 60
img_cols <- 70

X_train <- data_expressions.train[,1:4200]
X_test <- data_expressions.test[,1:4200]
y_train <-  data_expressions.train$y
y_test <-  data_expressions.test$y

# reshapes for dense layers
X_train1<- array_reshape(unname(X_train), dim=c(dim(X_train)[1], dim(X_train)[2])) 
X_test1<- array_reshape(unname(X_test), dim=c(dim(X_test)[1], dim(X_test)[2])) 
# rescale
X_train1 <- X_train / 255
X_test1 <- X_test / 255

#reshapes for cnn layers 
X_train2 <- array_reshape(unname(X_train), c(nrow(X_train), img_rows, img_cols, 1))
X_test2 <- array_reshape(unname(X_test), c(nrow(X_test), img_rows, img_cols, 1))

input_shape <- c(img_rows, img_cols, 1)

# transforms labels in vectors
y_train_cat <- to_categorical(y_train)
y_test_cat <- to_categorical(y_test)

y_train_cat2 <- y_train_cat[,2:7]
y_test_cat2 <- y_test_cat[,2:7]

#nb of classes
num_classes=6

model <- keras_model_sequential() 
model %>% 
  #layer_dense(units = 10, activation = 'relu', input_shape = c(4200)) %>% 
  layer_conv_2d(filters = 12, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 24, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 6, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

summary(model)

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = opt,
  metrics = c('accuracy')
)

history <- model %>% fit(
  X_train2, y_train_cat2, 
  epochs = 10, batch_size = 1, 
  validation_split = 0.2
)


model %>% evaluate(X_test2, y_test_cat2)

model %>% predict_classes(X_test2)
y_test
################## H2O - CNN ################## 

library(h2o)

#start a local h2o cluster
local.h2o <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads=-1)

# pass dataframe from inside of the R environment to the H2O instance

trData<-as.h2o(data_expressions.train)
tsData<-as.h2o(data_expressions.test)

head(trData)

res.dl <- h2o.deeplearning(x = 1:4200, y = 4201, trData, activation = "Tanh", hidden=rep(160,5),epochs = 100)

#use model to predict testing dataset
pred.dl<-h2o.predict(object=res.dl, newdata=tsData[,1:4200])
pred.dl.df<-as.data.frame(pred.dl)

summary(pred.dl)
test_labels<-data_expressions.test[,4201]

#calculate number of correct prediction
sum(diag(table(test_labels,pred.dl.df[,1])))
table(test_labels,pred.dl.df[,1])
length(test_labels)
28/36
h2o.shutdown(prompt = FALSE)





