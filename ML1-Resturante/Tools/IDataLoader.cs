﻿using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML1_Resturante.Tools
{
    public interface IDataLoader
    {
        IDataView LoadData<T>(string filePath) where T : class;
    }
}
