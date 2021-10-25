import numpy as np

test = [1,2,3,4,5,6,7,8]


############ reshape test
## numpy array 로 바꾸기
reshape_array=np.array(test);
print(reshape_array)
print(np.shape(reshape_array))

##
reshape_test1 = np.reshape(reshape_array,(-1))
print(reshape_test1)


reshape_test1 = np.reshape(reshape_array,(-1,2))
print(reshape_test1)



reshape_test1 = np.reshape(reshape_array,(1,2,-1))
print(reshape_test1)



## numpy array 합치기
a = np.array([[1, 2], [3, 4]])
b = np.array([[7, 8], [9, 10]])

print('a.T')
print(a.T) ## 전치된 배열

print('np.concatenate((a, b), axis=0)')
print(np.concatenate((a, b), axis=0)) ## 배열 뒤에 붙이기
print('np.concatenate((a, b), axis=1)')
print(np.concatenate((a, b), axis=1)) ## 배열들의 같은 행끼리 합치기
print('np.concatenate((a, b), axis=None)')
print(np.concatenate((a, b), axis=None)) ## 배열을 하나의 1차원으로 합치기
