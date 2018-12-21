import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\\Brett\\Downloads\\students-performance-in-exams\\StudentsPerformance.csv")

df.head()

x = df['writing score'].values
y = df['reading score'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42)

trainingdata = np.array(list(zip(x_train,y_train)))

m = 0
b = 0

def step_gradient(mcurr,bcurr,trainingdata,learningrate):
    ddm = 0 # gradient with respect to m
    ddb = 0 # gradient with respect to b
    N = len(trainingdata)
    for i in range(N):
        x = trainingdata[i,0]
        y = trainingdata[i,1]
        yhat = mcurr*x + bcurr
        ddm += 2/N*(np.sum(-x*(y-yhat)))
        ddb += 2/N*(np.sum(-(y-yhat)))
    newm = mcurr - (learningrate * ddm)
    newb = bcurr - (learningrate * ddb)
    return newm,newb

def cost(m,b,trainingdata):
    J = 0
    for i in range(len(trainingdata)):
        x = trainingdata[i,0]
        y = trainingdata[i,1]
        yhat= (m*x + b)
        J += np.sum((y - yhat)**2)
    return J/len(trainingdata) 

for i in range(50):
    m,b = step_gradient(m,b,trainingdata,0.0001)

print('done')

results = np.zeros(x_test.shape)

for i in range(250):
    results[i] = m * x_test[i] + b
    
error = results - y_test
error
