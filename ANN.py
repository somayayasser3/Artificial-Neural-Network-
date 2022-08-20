import numpy as np
import random

# opens the file we will read from
my_file = open("E:\Bioninformatics\year3,sem2\Genetic Algorithms\Assignments\Assignment4\FBNN4.txt", "r")
# reads the content of the file
content = my_file.read()
content_list = content.split("\n")  # split by line
eta = random.uniform(0.01, 0.99)  # random learning rate
# eta = 0.05    # best learning weight value
new_list = []
# iterates over the content and split by the spaces
for i in content_list:
    Str = i.split(" ")
    new_list.append(Str)

inputN = int(new_list[0][0])  # number of input nodes
hidden = int(new_list[0][1])  # number of hidden nodes
output = int(new_list[0][2])  # number of output nodes
points = int(new_list[1][0])  # number of training examples
del new_list[0:2]
InputPoints = []  # training data
OutputPoints = []  # desired outputs
for p in new_list:  # loops over the file content after being added to a list of lists
    l1 = [1]  # appends the bias value
    l2 = [1]  # appends the bias value
    for i in range(len(p)): # loops over each list the list of lists
        if i < inputN:  # along the number of input nodes
            l1.append(float(p[i]))  # appends to the input list
        else:   #
            l2.append(float(p[i]))  # appends to output list
    InputPoints.append(l1)   # appends list to input list
    OutputPoints.append(l2)  # appends list to output list
inputWeights = []
hiddenWeights = []
numOfWeights = ((1 + inputN) * hidden) + ((1 + hidden) * output)    # calculating the number of weights
randomNum = 1 / numOfWeights    # to get the range of the weight


# Generates random input weights
def initialInputWeights(inputN, hidden):
    for i in range(inputN + 1):
        w = []
        for j in range(hidden):
            w.append(random.uniform(-randomNum, randomNum))  # random initialization of the first input weights
        inputWeights.append(w)
    return inputWeights


# Generated random weights going out from the hidden layer and going to the output nodes
def initialHiddenWeights(hidden, output):
    for i in range(hidden + 1):
        w = []
        for j in range(output):
            w.append(random.uniform(-randomNum, randomNum))     # random initialization of the initial hidden output weights
        hiddenWeights.append(w)
    return hiddenWeights


# Calculates the sum of products of the hidden nodes
def SOP(i, k):
    sop = 0
    for j in range(len(InputPoints[i])):
        sop += InputPoints[i][j] * inputWeight[j][k]    # Summation of input * input weight
    return sop


# Calculates the sum of product of the output nodes
def SOPout(f, hFinal, h):
    sopOut = 0
    for i in range(len(hFinal[f])):
        sopOut += hFinal[f][i] * hiddenWeight[i][h]     # Summation od hidden output * hidden weight
    return sopOut


# Activation Function(Squash function)
def sigmoid(z):
    res = (1 / (1 + np.exp(-z)))    # sigmoid equation
    return res


# Calculates the feed forward propagation
def feed_forward():
    hFinal = []
    hFinal2 = []
    for i in range(inputN):
        hOut = [1]  # adds the value that is calculated by the bias
        for k in range(hidden):  # loops over hidden nodes value
            sop = SOP(i, k)     # calls SOP to calculate the hidden nodes input
            out = sigmoid(sop)  # calls activation function, to calculate the hidden nodes output
            hOut.append(out)    # appends to the list of hidden output
        hFinal.append(hOut)  # appends each hidden node output to a list
    for f in range(len(hFinal)):
        hOut = [1]  # adds the value that is calculated by the bias
        for h in range(output):  # loops over output values
            on = SOPout(f, hFinal, h)   # calls SOP to calculate the output
            out = sigmoid(on)   # calculates output nodes output using the activation function
            hOut.append(out)    # appends to the list of output
        hFinal2.append(hOut)    # appends the output of each output node to a list
    return hFinal, hFinal2  # returns both the output of both the hidden and output nodes


# Calculates the mean square error
def MSE(predict):
    Sum = 0
    mse = []
    for i in range(len(OutputPoints)):
        for j in range(len(OutputPoints[i])):
            Sum += (predict[i][j] - OutputPoints[i][j]) ** 2    # summation of calculated(predicted) - desired(actual) value all squared
        mse.append((1 / (2 * output)) * Sum)    # 1/2*m m= number of output, to decrease the error rate
    return mse


# Calculates the error of the output node
def errorOutput(h):
    errors = []
    for i in range(len(h)):
        l = []
        for j in range(1, len(h[i])):   # starting form 1 to ignore the bias
            x = (h[i][j] * (1 - h[i][j])) * (OutputPoints[i][j] - h[i][j])  # errorA = outputA * (1-outputA) * (desired value of outputA - outputA)
            l.append(x)     # appends the output of each input to a list
        errors.append(l)    # appends all the outputs to an error list of lists
    return errors   # returns the list of lists containing the error


# updates the weights to be used after the back propagation error calculation
def updateWeight(hiddenOut, errorOut, errorHidden, inputList):
    # first two loops are used to update the hidden nodes weights
    for i in range(len(hiddenWeight)):  # number of lists
        for j in range(len(hiddenWeight[i])):   # number of values in each list
            # WBA' = WBA + (learningRate(eta) * errorB * outputA)
            hiddenWeight[i][j] = hiddenWeight[i][j] + (eta * errorOut[i][j] * hiddenOut[i][j + 1])  # put j+1 to ignore the bias

    # these two loops are to update the input weights that we have randomly initialized earlier in the code
    for i in range(1, len(inputWeight)):    # starting from 1 to ignore the bias
        for j in range(len(inputWeight[i])):
            # WAB' = WAB + (learningRate * (errorB * inputA )
            inputWeight[i][j] = inputWeight[i][j] + (eta * errorHidden[i - 1][j] * inputList[i])


# Calculates error of hidden nodes
def errorHidden(errorOut, hiddenOut):
    hiddenError = []
    for i in range(len(hiddenOut)):
        LIST = []
        for j in range(1, len(hiddenOut[i])):   # starting from 1 to ignore the bias
            # errorHiddenA = hiddenOutA * (1-hiddenOutA) * (WAB * errorB + WAC * errorC)
            # divided the rule over x and n as we have multiple errors for multiple weights
            x = hiddenOut[i][j] * (1 - hiddenOut[i][j])  # hiddenOutA * (1-hiddenOutA)
            n = 0
            for k in range(len(errorOut[i])):
                n += (errorOut[i][k] * hiddenWeight[i][k])  # (WAB * errorB + WAC * errorC)
            x *= n
            LIST.append(x)
        hiddenError.append(LIST)
    return hiddenError # returns hiddenOutput errors list of lists


# initially calls the initialization of weights functions
inputWeight = initialInputWeights(inputN, hidden)
hiddenWeight = initialHiddenWeights(hidden, output)
mse = 0
# iterates 500 times to get the least errors for updated weights
for i in range(500):
    # calls the feedForward function
    hiddenOut, h = feed_forward()
    # calls the function to calculate the output error
    errorOut = errorOutput(h)
    # calls the function to calculate the hidden error
    errHid = errorHidden(errorOut, hiddenOut)
    m = 0
    # saves the MSE to a variable to loop over
    mse = MSE(h)
    x = 0   # flag to check acceptable mse
    for j in range(len(mse) - 1):    # loops over the mse
        if mse[j] > mse[j + 1]:  # checks which value has the bigger error to choose which list of the input point will be used
            m = j
        if mse[j] > 1.0:    # the acceptable weight condition
            x = 1   # to make sure all the values in the mse are less than 1
    if x == 0:
        break
    updateWeight(hiddenOut, errorOut, errHid, InputPoints[m])   # updates the weights and starts over by calling the feedForward function and so on

# prints the mse for back propagation
print(mse)

# writes the weights to a file
file1 = open("out.txt", "w")

# write input weights to the file
file1.write(str(inputWeight))
file1.write("\n")

# write hidden weights to the file
file1.write(str(hiddenWeight))
file1.write("\n")

file1.close()


# calling the feedForward function(program) after updating weights in back propagation
hiddenOut, h = feed_forward()

# prints the mse for feedForward propagation
print(MSE(h))


