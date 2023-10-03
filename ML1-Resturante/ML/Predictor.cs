using Microsoft.ML;
using ML1_Resturante.ML.Objects;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML1_Resturante.ML
{
    public class Predictor
    {
        public string Predict(string inputData)
        {
            var context = new MLContext();

            if (!File.Exists(Environment.CurrentDirectory + "//TrainedModel.mdl"))
            {
                return "File does not exist";
            }
            ITransformer mlModel;

            using (var stream = new FileStream(Environment.CurrentDirectory + "//TrainedModel.mdl", FileMode.Open, FileAccess.Read,
            FileShare.Read))
            {
                mlModel = context.Model.Load(stream, out _);
            }
            if (mlModel == null)
            {
                return "Model is null";
            }
            var predictionEngine = context.Model.CreatePredictionEngine<ResturantFeedback,ResturantPrediction>(mlModel);
            var prediction = predictionEngine.Predict(new ResturantFeedback { Text = inputData });

            return prediction.Prediction ? "Negative" : "Positive";

            //Console.WriteLine($"Based on \"{inputData}\", the feedback is predicted to be:{ Environment.NewLine}{ (prediction.Prediction ? "Negative" : "Positive")} at a { prediction.Probability:P0} confidence");
        }

        public string PredictEmp(string inputData) 
        {
            var context = new MLContext();

            if (!File.Exists(Environment.CurrentDirectory + "//TrainedModelEmp.mdl"))
            {
                return "File does not exist";
            }
            ITransformer mlModel;

            using (var stream = new FileStream(Environment.CurrentDirectory + "//TrainedModelEmp.mdl", FileMode.Open, FileAccess.Read,
            FileShare.Read))
            {
                mlModel = context.Model.Load(stream, out _);
            }
            if (mlModel == null)
            {
                return "Model is null";
            }
            var predictionEngine = context.Model.CreatePredictionEngine<EmploymentHistory, EmploymentHistoryPrediction>(mlModel);
            var json = File.ReadAllText(inputData);
            var prediction =
            predictionEngine.Predict(JsonConvert.DeserializeObject<EmploymentHistory>(json));

            Console.WriteLine($"Based on input json:{System.Environment.NewLine}" +
            $"{json}{System.Environment.NewLine}" +
            $"The employee is predicted to work {prediction.DurationInMonths:#.##} months");

            return "";
        }
    }
}
