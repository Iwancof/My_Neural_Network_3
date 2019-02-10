using System;
using System.Collections.Generic;
using System.Text;

using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;

namespace My_Neural_Network_3
{
    partial class Neural
    {
        Matrix<double> Input_Weigth;
        Matrix<double> Hidden_Weigth;
        Matrix<double> Output_Weigth;
        Vector<double> Input_Bias;
        Vector<double> Hidden_Bias;
        Vector<double> Output_Bias;
        //いずれも受け取る時

        Vector<double> InputLayerDate;
        Vector<double> HiddenLayerData;
        Vector<double> OutputLayerData;

        public Vector<double> Calc(Vector<double> PureInput) {
            //Vector<double> PureInput = Vector<double>.Build.DenseOfArray(__Inputdata);

            InputLayerDate = Sigmoid(PureInput * Input_Weigth + Input_Bias);
            HiddenLayerData = Sigmoid(InputLayerDate * Hidden_Weigth + Hidden_Bias);
            OutputLayerData = HiddenLayerData * Output_Weigth + Output_Bias; //Sigmoidはかけない

            return OutputLayerData;
        }

        public void Train(Vector<double>[] Inputs,Vector<double>[] Answers,double Eps,int Epoc) {
            if (Inputs.Length != Answers.Length) throw new Exception("");
            for(int TrainCount = 0;TrainCount < Epoc; TrainCount++) {
                for (int TrainDataCount = 0; TrainDataCount < Inputs.Length; TrainDataCount++) {
                    var input = Inputs[TrainDataCount];
                    var answer = Answers[TrainDataCount];

                    Update_Weigth(input, answer, Eps);
                }
            }
        }

        public void Update_Weigth(Vector<double> inp,Vector<double> ans,double Eps) {
            Calc(inp);

            var Output_Delta = (OutputLayerData - ans) * OutputLayerData * (1.0 - OutputLayerData);
            Output_Weigth -= Eps * Output_Delta * HiddenLayerData;

            var Hidden_Delta = Output_Weigth * Output_Delta * HiddenLayerData * (1.0 - HiddenLayerData);
            double[,] tmp = new double[3, 2];
            for(int i = 0;i < 3; i++) {
                for(int j = 0;j < 2; j++) {
                    tmp[i, j] = Hidden_Delta[j] * InputLayerDate[i];
                }
            }
            Matrix<double> tmp_ = Matrix<double>.Build.DenseOfArray(tmp);
            Hidden_Weigth -= Eps * tmp_;
        }
    }
}
