using Microsoft.ML;
using ML1_Resturante.ML.Objects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML1_Resturante.Tools
{
    public class TextFileDataLoader : IDataLoader
    {
        private readonly MLContext _context;

        public TextFileDataLoader(MLContext context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
        }

        public IDataView LoadData<T>(string filePath) where T : class
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Failed to find training data file ({filePath}");
                return null;
            }

            return _context.Data.LoadFromTextFile<T>(filePath);
        }

    }
}
