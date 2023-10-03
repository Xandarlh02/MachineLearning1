using Microsoft.ML;
using ML1_Resturante.ML;


var context = new MLContext();
var trainer = new Trainer(context);

Predictor predictor = new Predictor();

trainer.Train("C:\\Users\\Alexandar Lackovic\\Documents\\GitHub\\MachineLearning1\\ML1-Resturante\\Data\\sampledata.csv");
predictor.Predict("Very poor food");
