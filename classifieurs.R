classifieur_expressions <- function(dataset) {
  # Chargement de l’environnement
  load("env.Rdata",.GlobalEnv)
  X <- dataset[,1:4200]
  y <-dataset$y
  # On sélectionne les parties expressives du visage 
  Xselec <- cbind(X[301:1260],X[2460:3359])
  #parmi les données sélectionnées, on enlève aussi les pixels noirs
  X_selpro<- Xselec[, !apply(Xselec == 0, 2, all)]
  data_selpro=data.frame(X_selpro,y=y)
  pred_pca <- predict(.GlobalEnv$prin_comp_pca, X_selpro)
  new_data2 <-  data.frame( pred_pca[,1:25],y=y)
  predictions<- predict(.GlobalEnv$classifieur_character, dataset)
  return(predictions)
}

classifieur_characters <- function(dataset) {
  # Chargement de l’environnement
  require(randomForest)
  load("env.Rdata",.GlobalEnv)
  predictions<- predict(.GlobalEnv$classifieur_character, dataset)
  return(predictions)
}
classifieur_parole <- function(dataset) {
  # Chargement de l’environnement
  require('e1071')
  load("env.Rdata",.GlobalEnv)
  predictions<- predict(.GlobalEnv$classifieur_parole, dataset)
  return(predictions)
}