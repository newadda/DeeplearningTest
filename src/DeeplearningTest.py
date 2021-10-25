# 패키지
# numpy tensorflow sklearn

import csv
import numpy as np
import math

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler



class Parameter:
    algorithm=None # 알고리즘 : 'LSTM',GRU
    loss=None # 손실함수명 : 'mae','mse','msle','mape','kld'
    optimizer=None # 최적화명 : 'adam','adamax','nadam',
    activation_fuc=None # 활성화함수명 : 'relu','sigmoid','tanh',
    hidden_layers=[5, 2]# 열은 히든 레이어를 나타냄, 각 열값은 해당 히든 레이어의 neurons 개수
    epoch_arg=50
    input_count=1
    output_count=1





class DeepLearning:
    def __init__(self,parameter):
        self.parameter = parameter

    def createModel(self):
        algorithm=self.parameter.algorithm;
        loss=self.parameter.loss;
        optimizer=self.parameter.optimizer;
        activation_fuc=self.parameter.activation_fuc;
        hidden_layers=self.parameter.hidden_layers;
        epoch_arg=self.parameter.epoch_arg;
        input_count=self.parameter.input_count;
        output_count=self.parameter.output_count;

        model = keras.Sequential()

        if algorithm.lower() == 'LSTM'.lower() :
            model.add(layers.LSTM(10,input_shape=(input_count,1)))
        elif algorithm.lower() == 'GRU'.lower():
            model.add(layers.GRU(10,input_shape=(input_count,1)))


        for neurons in hidden_layers:
            model.add(layers.Dense(neurons))



        model.add(layers.Dense(output_count))
        model.compile(loss=loss,optimizer=optimizer)
        self.model = model



def main():
    train_per = 0.7

    ## csv 데이터 읽기
    filePath = "../sample/deeplearning_sample.csv"
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        datas=list(reader)

    ## numpy array 로 변경
    datas = np.array(datas)

    ## array type float 로 변경
    datas = datas.astype(float)

    ## 입력 데이터 다양화( x1^2, x2^2 ,  x1 * x2, sin(x1), sin(x2)
    datas_ex = []
    for i in datas:
        x1 = i[0]
        x2 = i[1]
        datas_ex.append([math.pow(x1,2),math.pow(x2,2),x1*x2, math.sin(x1),math.sin(x2)])

    print(datas_ex)


    ## x 와 y 분리
    x_datas = datas[0:,0:2]
    x_datas_ex = np.concatenate((x_datas, datas_ex), axis=1)

    y_datas=datas[0:,2]
    y_datas = np.reshape(y_datas,(-1,1))





    #### 스케일링
    ## x, y 스케일링할 최소, 최대 지정
    x_scale_fix = [[0,0,0,0,0,-1,-1],[999,999,math.pow(999,2),math.pow(999,2),math.pow(999,2),1,1]]
    y_scale_fix = [[0],[999*999]]

    x_scaler = MinMaxScaler()
    x_scaler.fit(x_scale_fix)
    x_scale_datas= x_scaler.transform(x_datas_ex)

    y_scaler = MinMaxScaler()
    y_scaler.fit(y_scale_fix)
    y_scale_datas= y_scaler.transform(y_datas)


    ## 스케일링 복구
    #inverse_datas = scaler.inverse_transform(datas)



    ## train, test 데이터 나누기
    totalRowCount = len(x_scale_datas);
    trainRowCount=round(totalRowCount*train_per,0);
    testRowCount=totalRowCount-trainRowCount;

    trainRowCount=int(trainRowCount)
    testRowCount=int(testRowCount)

    x_trainDatas = x_scale_datas[0:trainRowCount]
    x_testDatas = x_scale_datas[trainRowCount:]

    y_trainDatas = y_scale_datas[0:trainRowCount]
    y_testDatas = y_scale_datas[trainRowCount:]





    parameter = Parameter()
    parameter.algorithm='GRU' # 알고리즘 : 'LSTM',GRU
    parameter.loss='mae' # 손실함수명 : 'mae','mse','msle','mape','kld'
    parameter.optimizer='adam' # 최적화명 : 'adam','adamax','nadam',
    parameter.activation_fuc='relu' # 활성화함수명 : 'relu','sigmoid','tanh',
    parameter.hidden_layers=[]# 열은 히든 레이어를 나타냄, 각 열값은 해당 히든 레이어의 neurons 개수
    parameter.epoch_arg=15000
    parameter.batch_size=80
    parameter.input_count=7
    parameter.output_count=1

    trainInputDatas = np.reshape(x_trainDatas,(-1,x_trainDatas.shape[1],1))
    trainoutputDatas = np.reshape(y_trainDatas,(-1,y_trainDatas.shape[1]))

    testInputDatas=x_testDatas.reshape((-1,x_testDatas.shape[1],1))
    testoutputDatas=y_testDatas.reshape((-1,y_testDatas.shape[1]))


    deepLearning = DeepLearning(parameter)
    deepLearning.createModel()

    deepLearning.model.fit(trainInputDatas,trainoutputDatas,epochs= parameter.epoch_arg
                         ,batch_size=parameter.batch_size,verbose=1)

    #### 오차 확인
    test_result=deepLearning.model.evaluate(testInputDatas,testoutputDatas,batch_size=50)

    print(f'test result = {test_result}')


    #input_predict_datas = np.array(testDatas)
    y_hat = deepLearning.model.predict(testInputDatas,batch_size=10)

    print(f'{testInputDatas.shape}')
    inverse_y_hat = y_scaler.inverse_transform(y_hat)
    inverse_testoutputDatas = y_scaler.inverse_transform(testoutputDatas)
    print(inverse_y_hat[0:10])
    print(inverse_testoutputDatas[0:10])




    print(test_result)





main()