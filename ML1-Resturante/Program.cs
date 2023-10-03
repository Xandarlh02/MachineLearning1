using ML1_Resturante.ML;

Trainer trainer = new Trainer();
Predictor predictor = new Predictor();

trainer.LoadFromTextFile("C:\\Users\\Alexandar Lackovic\\source\\repos\\ML1-Resturante\\Data\\sampledata.csv");
predictor.Predict("Very good");
