using Microsoft.ML;
using ML1_Resturante.ML.Objects;
using ML1_Resturante.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Reflection;
using ML1_Resturante.Common;

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
            var data = _dataLoader.LoadData<ResturantFeedback>(filePath);
            if (data != null)
            {
                Train(data);
            }
        }

        public void LoadAndTrainEmp(string filePath)
        {
            var data = _dataLoader.LoadData<EmploymentHistory>(filePath);
            if(data != null)
            {
                TrainEmp(data);
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

        private void TrainEmp(IDataView data)
        {
            var dataSplit = _context.Data.TrainTestSplit(data, testFraction: 0.2);
            var dataProcessPipeline = _context.Transforms.CopyColumns("Label", nameof(EmploymentHistory.DurationInMonths))
                .Append(_context.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.IsMarried)))
                .Append(_context.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.BsDegree)))
                .Append(_context.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.MsDegree)))
                .Append(_context.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.YearsExperience)))
                .Append(_context.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.AgeAtHire)))
                .Append(_context.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.HasKids)))
                .Append(_context.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.WithinMonthOfVesting)))
                .Append(_context.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.DeskDecorations)))
                .Append(_context.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.LongCommute)))
                .Append(_context.Transforms.Concatenate("Features",
                    nameof(EmploymentHistory.IsMarried),
                    nameof(EmploymentHistory.BsDegree),
                    nameof(EmploymentHistory.MsDegree),
                    nameof(EmploymentHistory.YearsExperience),
                    nameof(EmploymentHistory.AgeAtHire),
                    nameof(EmploymentHistory.HasKids),
                    nameof(EmploymentHistory.WithinMonthOfVesting),
                    nameof(EmploymentHistory.DeskDecorations),
                    nameof(EmploymentHistory.LongCommute)
                ));

            ITransformer trainedModel = dataProcessPipeline.Fit(dataSplit.TrainSet);
            _context.Model.Save(trainedModel, dataSplit.TrainSet.Schema, Environment.CurrentDirectory + "//TrainedModelEmp.mdl");
            var testSetTransform = trainedModel.Transform(dataSplit.TestSet);

            var trainer = _context.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var modelMetrics = _context.Regression.Evaluate(testSetTransform);
        }
    }

}
