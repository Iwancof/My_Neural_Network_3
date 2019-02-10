using System;
using System.Collections.Generic;
using System.Text;

using MathNet.Numerics.LinearAlgebra;
//using MathNet.Numerics.LinearAlgebra.Double;

namespace My_Neural_Network_3
{
    partial class Neural
    {
        public Neural() {
            //3-2-2のネットワークを構築する。

            Random rand = new Random(100); //わかりやすくするため

            double[,] _Input_Weigth = new double[3, 3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    //_Input_Weigth[i, j] = (double)rand.Next(0, 10000) / 10000d;
                    _Input_Weigth[i, j] = 1;
            Input_Weigth = Matrix<double>.Build.DenseOfArray(_Input_Weigth);

            double[,] _Hidden_Weigth = new double[3, 2];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++)
                    _Hidden_Weigth[i, j] = (double)rand.Next(0, 10000) / 10000d;
            Hidden_Weigth = Matrix<double>.Build.DenseOfArray(_Hidden_Weigth);

            double[,] _Output_Weigth = new double[2, 2];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    _Output_Weigth[i, j] = (double)rand.Next(0, 10000) / 10000d;
            Output_Weigth = Matrix<double>.Build.DenseOfArray(_Output_Weigth);

            double[] _Input_Bias = new double[3];
            for (int i = 0; i < 2; i++)
                //_Input_Bias[i] = (double)rand.Next(0, 1000) / 10000d;
                _Input_Bias[i] = 0;
            Input_Bias = Vector<double>.Build.DenseOfArray(_Input_Bias);

            double[] _Hidden_Bias = new double[2];
            for (int i = 0; i < 2; i++)
                _Hidden_Bias[i] = (double)rand.Next(0, 1000) / 10000d;
            Hidden_Bias = Vector<double>.Build.DenseOfArray(_Hidden_Bias);

            double[] _Output_Bias = new double[2];
            for (int i = 0; i < 2; i++)
                _Output_Bias[i] = (double)rand.Next(0, 1000) / 10000d;
            Output_Bias = Vector<double>.Build.DenseOfArray(_Output_Bias);
        }

        public Vector<double> Sigmoid(Vector<double> x) {
            return x.Map(a => 1 / (1 + Math.Exp(-a)));
        }
    }

}
