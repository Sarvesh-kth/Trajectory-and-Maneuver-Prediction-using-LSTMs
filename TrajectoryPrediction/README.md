The aim of the project is to perform Trajectory and Maneuver class Prediction for a vehicle using past coordinate information of the target and neighbouring vehicles. 
HighD dataset was used. Performed preprocessing for necessary data extraction.
LSTMs encoder decoder structure was used to extract temporal information.
Two networks are built, one for trajectory prediction and one for maneuver class. The maneuver classes are divided into 2 types with respect to the next few predicted seconds, latitudinal and logitudinal. Latitudinal refers to whether a lane change occurs and logitudinal refers to whether the vehicle must slow down. 
The maneuver predicted from one model is integrated to the data for trajectory prediction
