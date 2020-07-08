import numpy as np
import pandas as pd

# Courtesy of https://towardsdatascience.com/introduction-to-naïve-bayes-classifier-fa59e3e24aaf

data=[['A','S',0],
      ['A','M',0],
      ['A','M',1],
      ['A','S',1],
      ['A','S',0],
      ['B','S',0],
      ['B','M',0],
      ['B','M',1],
      ['B','L',1],
      ['B','L',1],
      ['C','L',1],
      ['C','M',1],
      ['C','M',1],
      ['C','L',1],
      ['C','L',0]
      ]
data = pd.DataFrame(data,columns=['x1','x2','y'])

###########################
# Train Naive Bayes Model
###########################
def Naive_Bayes(data):
    # Step 1: Calculate Prior Probability
    y_unique = data.y.unique() # data.y.unique() = 2
    prior_probability = np.zeros(len(data.y.unique()))
    for i in range(0, len(y_unique)):
        prior_probability[i] = sum(data['y'] == y_unique[i]) / len(data['y'])

    # Step 2: Calculate Conditional Probability
    conditional_probability = {} # Creates and empty dictionary
    for i in range(1, data.shape[1]): # data.shape[1] = 3
        x_unique = list(set(data['x' + str(i)])) # set() returns unique values
        x_conditional_probability = np.zeros((len(data.y.unique()), len(set(data['x' + str(i)]))))
        for j in range(0, len(y_unique)): # len(y_unique) = 2
            for k in range(0, len(x_unique)): # len(x_unique) = 3
                x_conditional_probability[j, k] = data.loc[(data['x' + str(i)] == x_unique[k]) & (data['y'] == y_unique[j])].shape[0] / sum(data['y'] == y_unique[j])

        x_conditional_probability = pd.DataFrame(x_conditional_probability, columns=x_unique, index=y_unique)
        conditional_probability['x' + str(i)] = x_conditional_probability # Updates the dictionary values of x1 and x2

    return prior_probability, conditional_probability


####################
# Prediction
####################
def prediction(x_list):
    x1, x2 = x_list
    p0 = prior_probability[0] * conditional_probability['x1'][x1][0] * conditional_probability['x2'][x2][0]
    p1 = prior_probability[1] * conditional_probability['x1'][x1][1] * conditional_probability['x2'][x2][1]

    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


prior_probability, conditional_probability = Naive_Bayes(data)
print(prediction(['B', 'S']))



