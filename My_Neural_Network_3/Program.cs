using System;

using MathNet.Numerics.LinearAlgebra;

namespace My_Neural_Network_3
{
    class Program
    {
        static void Main(string[] args) {
            Neural n = new Neural();

            Vector<double>[] input = {
                Vector<double>.Build.DenseOfArray(new double[] {0,0,0}),
                Vector<double>.Build.DenseOfArray(new double[] {0,1,0}),
                Vector<double>.Build.DenseOfArray(new double[] {1,0,0}),
                Vector<double>.Build.DenseOfArray(new double[] {1,1,0}),
            };

            Vector<double>[] answer = {
                Vector<double>.Build.DenseOfArray(new double[] {0,0}),
                Vector<double>.Build.DenseOfArray(new double[] {0,1}),
                Vector<double>.Build.DenseOfArray(new double[] {0,1}),
                Vector<double>.Build.DenseOfArray(new double[] {1,0}),
            };

            n.Train(input, answer, 0.1, 10000);

            var d = n.Calc(input[3]);

            Console.WriteLine(d);

            Console.WriteLine("Finished");
            Console.ReadLine();
        }
    }
}
