using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML1_Resturante.ML
{
    public interface ITrainer
    {
        void LoadAndTrain(string filePath);
    }
}
