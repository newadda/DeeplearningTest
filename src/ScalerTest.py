from sklearn.preprocessing import MinMaxScaler

fixDatas = [[0,0],[500,1000]]

testDatas =[[250,500],[400,900],[100,100]]


x_scaler = MinMaxScaler()

## 최소, 최대 데이터를 통해 핏팅하기.
x_scaler.fit(fixDatas)

## 핏팅데이터를 근거로 스케일링 하기
transDatas = x_scaler.transform(testDatas)
print(transDatas)


## fit_transform()은  fit()과 transform ()를 순차적으로 진행하는 함수이다. fix()를 덮어쓰므로 사용에 주의하자.
#x_scaler.fit_transform(testDatas)

