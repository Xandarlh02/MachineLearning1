using Microsoft.ML;
using ML1_Resturante.ML;
using ML1_Resturante.Tools;

var context = new MLContext();
IDataLoader dataLoader = new TextFileDataLoader(context);
ITrainer trainer = new Trainer(context, dataLoader);
Predictor predictor = new Predictor();

//trainer.LoadAndTrain("C:\\Users\\Alexandar Lackovic\\Documents\\GitHub\\MachineLearning1\\ML1-Resturante\\Data\\sampledata.csv");
string PositiveOrNegative = predictor.Predict("Very poor food");
Console.WriteLine("Word is " + PositiveOrNegative);
