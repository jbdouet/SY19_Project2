classifieur_expressions <- function(dataset) {
  # Chargement de l’environnement
  load("env.Rdata")
  # Mon algorithme qui renvoie les prédictions sur le jeu de données
  # ‘dataset‘ fourni en argument.
  # ...
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