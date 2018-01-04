source("classifieurs.R")
data_expressions <- read.csv("data/expressions_train.txt",sep = " ")
data_characters<- read.csv("data/characters_train.txt",sep = " ")
data_parole <- read.csv("data/parole_train.txt")
classifieur_expressions(data_expressions)
classifieur_characters(data_characters)
classifieur_parole(data_parole)



