import csv
import numpy as np

def loadFileCSV(f):
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    datas=list(reader)
    return headers , datas


## csv file 열기

filePath = "../sample/csv1.csv"
with open(filePath, 'r') as f:
    headers , datas =loadFileCSV(f);

print(headers)
print(datas)
'''
['column1', 'column2', 'column3', 'column4', 'column5']
[['1', '2', '3', '4', '5'], ['11', '22', '33', '44', '55'], ['111', '222', '333', '444', '555']]
'''


###  배열을 numpy 배열로 바꾸기

npDatas = np.array(datas);
print(npDatas)
'''
[['1' '2' '3' '4' '5']
 ['11' '22' '33' '44' '55']
 ['111' '222' '333' '444' '555']]
'''

### numpy 배열의 데이터 형태를 float 으로 바꾸기
npDatas = npDatas.astype(float)
print(npDatas)
'''
[[  1.   2.   3.   4.   5.]
 [ 11.  22.  33.  44.  55.]
 [111. 222. 333. 444. 555.]]
'''

### 배열 분할
print(f'분할 {npDatas[0:,0:1]}')




### numpy 배열 반복문 예

for i in npDatas:
    for index,item in enumerate(i):
        print(f'{index} : {item}')
'''
0 : 1.0
1 : 2.0
2 : 3.0
3 : 4.0
4 : 5.0
0 : 11.0
..........
'''