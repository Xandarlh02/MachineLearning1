using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML1_Resturante.ML.Objects
{
    public class EmploymentHistory
    {
        public double DurationInMonths { get; set; }
        public int IsMarried { get; set; }  // Consider using a bool if only 0 or 1 is possible
        public int BsDegree { get; set; }   // Consider using a bool if only 0 or 1 is possible
        public int MsDegree { get; set; }   // Consider using a bool if only 0 or 1 is possible
        public int YearsExperience { get; set; }
        public int AgeAtHire { get; set; }
        public int HasKids { get; set; }    // Consider using a bool if only 0 or 1 is possible
        public int WithinMonthOfVesting { get; set; } // Consider using a bool if only 0 or 1 is possible
        public int DeskDecorations { get; set; } // Consider using a bool if only 0 or 1 is possible
        public int LongCommute { get; set; }  // Consider using a bool if only 0 or 1 is possible
    }
}
