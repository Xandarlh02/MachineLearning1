using Microsoft.ML;
using ML1_Resturante.ML.Objects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML1_Resturante.ML
{
    public class Predictor
    {
        public void Predict(string inputData)
        {
            var context = new MLContext();

            if (!File.Exists(Environment.CurrentDirectory + "//TrainedModel.mdl"))
            {
                Console.WriteLine($"Failed to find model at {Environment.CurrentDirectory + "//TrainedModel.mdl"}");
                return;
            }
            ITransformer mlModel;

            using (var stream = new FileStream(Environment.CurrentDirectory + "//TrainedModel.mdl", FileMode.Open, FileAccess.Read,
            FileShare.Read))
            {
                mlModel = context.Model.Load(stream, out _);
            }
            if (mlModel == null)
            {
                Console.WriteLine("Failed to load model");
                return;
            }
            var predictionEngine = context.Model.CreatePredictionEngine<ResturantFeedback,ResturantPrediction>(mlModel);
            var prediction = predictionEngine.Predict(new ResturantFeedback { Text = inputData });

            Console.WriteLine($"Based on \"{inputData}\", the feedback is predicted to be:{ Environment.NewLine}{ (prediction.Prediction ? "Negative" : "Positive")} at a { prediction.Probability:P0} confidence");
        }
    }
}
