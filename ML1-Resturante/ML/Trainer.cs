using Microsoft.ML;
using ML1_Resturante.ML.Objects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML1_Resturante.ML
{
    public class Trainer
    {
        public void LoadFromTextFile(string filePath)
        {
            var context = new MLContext();

            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Failed to find training data file ({filePath}");
            }
            else
            {
                var data = context.Data.LoadFromTextFile<ResturantFeedback>(filePath);
                Train(context, data);

            }
        }

        private static void Train(MLContext context, IDataView data)
        {
            var dataSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);
            var dataProcessPipeline = context.Transforms.Text.FeaturizeText(outputColumnName: "Features",
                inputColumnName: nameof(ResturantFeedback.Text));
            var sdcaRegressionTrainer = context.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(ResturantFeedback.Label),
            featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);
            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);
            context.Model.Save(trainedModel, dataSplit.TrainSet.Schema, Environment.CurrentDirectory + "//TrainedModel.mdl" );
            var testSetTransform = trainedModel.Transform(dataSplit.TestSet);
            var modelMetrics = context.BinaryClassification.Evaluate(data: testSetTransform,
                labelColumnName: nameof(ResturantFeedback.Label),
                scoreColumnName: nameof(ResturantPrediction.Score));
        }
    }
}
