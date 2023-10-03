using Microsoft.ML;
using ML1_Resturante.ML.Objects;
using ML1_Resturante.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML1_Resturante.ML
{
    public class Trainer : ITrainer
    {
        private readonly MLContext _context;
        private readonly IDataLoader _dataLoader;

        public Trainer(MLContext context, IDataLoader dataLoader)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _dataLoader = dataLoader ?? throw new ArgumentNullException(nameof(dataLoader));
        }

        public void LoadAndTrain(string filePath)
        {
            var data = _dataLoader.LoadData(filePath);
            if (data != null)
            {
                Train(data);
            }
        }

        private void Train(IDataView data)
        {
            var dataSplit = _context.Data.TrainTestSplit(data, testFraction: 0.2);
            var dataProcessPipeline = _context.Transforms.Text.FeaturizeText(outputColumnName: "Features",
                inputColumnName: nameof(ResturantFeedback.Text));
            var sdcaRegressionTrainer = _context.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(ResturantFeedback.Label),
            featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);
            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);
            _context.Model.Save(trainedModel, dataSplit.TrainSet.Schema, Environment.CurrentDirectory + "//TrainedModel.mdl");
            var testSetTransform = trainedModel.Transform(dataSplit.TestSet);
            var modelMetrics = _context.BinaryClassification.Evaluate(data: testSetTransform,
                labelColumnName: nameof(ResturantFeedback.Label),
                scoreColumnName: nameof(ResturantPrediction.Score));
        }
    }

}
