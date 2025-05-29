def print2DArray_DND(x_):
    # print(len(x_[0])) #Columns
    # print(len(x_))#Rows
    for i in range(len(x_)):
        for j in range(len(x_[0])):
            print(x_[i][j], end = "   ")
        print()


def circular_covolution(a,h_n):
    x_n = np.zeros(shape=(int(len(a)),int(len(a))))  # Initialize an empty list for the matrix

    
    for i in range(len(a)):
        x_n[i] = np.roll(a, i)

    x_n = x_n.T

    # Now Matrix Multiplication, y(n) = a*h 
    return np.dot(x_n, h_n)


def final_step_of_overlap_add(y_, M):
    y = []
    for i in range(len(y_)):
        for j in range(len(y_[0])):
            if i != 0: #Not 1st row!
                if j <= (M -1 -1 ):
                    y[j-(M-1)] += float(y_[i][j])
                    continue           
            y.append(float(y_[i][j]))
    return y

import numpy as np

x = [1,2,3,3,2,1,-1,-2,-3,5,6,-1 ,2,0,2,1]
h = [3, 2,1,1] #impulse response
N = 7 #BLOCK_Length when not specified -> N = 2**len(h) # BLOCK_SIZE
M = int(len(h))
L = N - M + 1
h = h + [0 for _ in range(L - 1)]
x = x + [0 for _ in range(L - (len(x)%L))]  # Padding the input signal to make its length a multiple of L

x_ = np.zeros(shape=(int(len(x)/L),L+M-1))  # Initialize an empty list for the matrix
# Create a 2D list (matrix) with L+M-1 rows and len(x)/L columns
# Initialize the matrix with zeros


# Converts x (1-D) matix to 2-D Matrix x_ , according to the L
j = -1
for i in range(len(x)):
    if i%L == 0:
        j += 1
    x_[j][i%L] = x[i]

#Testing
# print("Circular Convolution Result:",circular_covolution([1,2,3,3,0,0,0], [3,2,1,1,0,0,0]))

# Circular Convolution of x_n with h
y_ = np.zeros(x_.shape)  # Initialize an empty list for the matrix
for i in range(len(x_)):
    y_[i] = circular_covolution(x_[i], h)
