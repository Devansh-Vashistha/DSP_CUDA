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


def final_step_of_overlap_save(y_, M):
    y = []
    for i in range(len(y_)):
        for j in range((M - 1),len(y_[0])):
            y.append(float(y_[i][j]))
    return y


import numpy as np

x = [3,-1,0,1,3,2,0,1,2,1]
h = [1,1,1] #impulse response
N = 2**len(h) # BLOCK_SIZE

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
    if (j > 0) and (i%L <= (M -1 -1)):
        x_[j][i%L] = x_[j-1][(i%L)-(M-1)]   
    x_[j][(i%L)+(M-1)] = x[i]

#print2DArray_DND(x_)

y_ = np.zeros(x_.shape)  # Initialize an empty list for the matrix
for i in range(len(x_)):
    y_[i] = circular_covolution(x_[i], h)

#print2DArray_DND(y_)

y = final_step_of_overlap_save(y_, M)
#print("Final Output:", y)