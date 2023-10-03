using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML1_Resturante.ML.Objects
{
    public class ResturantFeedback
    {
        [LoadColumn(0)]
        public bool Label { get; set; }
        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
