import numpy as np
import math
import scipy.io
import scipy.signal
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import linear_model, naive_bayes, model_selection
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut


def logisticRegression(in_train, in_test, out_train, out_test):
    x = in_train
    y = out_train
   # loocv = model_selection.LeaveOneOut()
    model = linear_model.LogisticRegression()
    p = model.fit(x, y)
    results = p.score(in_test, out_test)
   # results = model_selection.cross_val_score(model, x, y, cv=loocv)
   # print("Accuracy: %.3f%% " % (results.mean() * 100.0))
    return results

#A linear-Gaussian model is a Bayes net where all the variables are Gaussian,
#  and each variable's mean is linear in the values of its parents. They are widely
# used because they support efficient inference. Linear dynamical systems are an important special case.
def linearGauClas(in_train, in_test, out_train, out_test):
    x = in_train
    y = out_train
    model = naive_bayes.GaussianNB()
    p = model.fit(x, y)
    results = p.score(in_test, out_test)
    return results

def perceptron(in_train, in_test, out_train, out_test):
    x = in_train
    y = out_train
    model = linear_model.Perceptron()
    p = model.fit(x, y)
    results = p.score(in_test, out_test)
    return results



def knnClas(in_train, in_test, out_train, out_test , n):
    x = in_train
    y = out_train
    model = KNeighborsClassifier(n_neighbors=n)
    p = model.fit(x, y)
    results = p.score(in_test, out_test)
    return results


def MLP(in_train, in_test, out_train, out_test, hlayers):
    x = in_train
    y = out_train
    model = MLPClassifier(hidden_layer_sizes=hlayers)
    p = model.fit(x, y)
    results = p.score(in_test, out_test)
    return results

# function that prints scatters for comparison 0-10
def plotScatters(input, output , number):

    for i in range(number):
        plt.scatter(input[:, i], output, color='k', marker='x')
        plt.ylabel('input')
        plt.xlabel('output')
        plt.title('PCA #'+str(i))
        plt.show()

# function calculates mean val
def getMean(list):
    return float(sum(list))/max(len(list), 1)




def main ():
    # Read data from matlab file
    data = scipy.io.loadmat('sonarTrainData.mat')

    # Split the data as informed in the description
    # Input is a 60 dimensional sonar echo, output is a class [0 or 1],
    # 0 for mine and 1 for rock
    info = data['info']
    inputTrain = data['inputSonarTrain']
    inputTest = data['inputSonarTest']
    outputTrain = data['outputSonarTrain']

    leave1out = LeaveOneOut()

    results_LR_DS = []
    results_LGC_DS = []
    results_P_DS = []
    results_kNN1_DS = []
    results_kNN3_DS = []
    results_kNN5_DS = []
    results_MLP_DS = []

    results_LR_PCA = []
    results_LGC_PCA = []
    results_P_PCA = []
    results_kNN1_PCA = []
    results_kNN3_PCA = []
    results_kNN5_PCA = []
    results_MLP_PCA = []


    for train, test in leave1out.split(inputTrain, outputTrain):

        #  VARIABLE TRANSFORMATION USING LEAVE ONE OUT MODEL
        # --------------------------------------   Ex 1   -----------------------------------------------#

                # DOWNSAMLING  (DS)
        # downsampling consists in summing the power in six bands,
        # thus lowering the number of inputs to ten
        inputTrainDS = scipy.signal.decimate(inputTrain, 6)

        in_train, in_test = inputTrainDS[train], inputTrainDS[test]
        out_train, out_test = outputTrain[train], outputTrain[test]


       ## score_LR_DS = (logisticRegression(in_train, in_test, out_train, out_test))
       # if( score_LR_DS == 1):
        #    results_LR_DS.append(score_LR_DS)

        score_LGC_DS = (linearGauClas(in_train, in_test, out_train, out_test))
        #if (score_LGC_DS == 1):
        results_LGC_DS.append(score_LGC_DS)

     #   score_P_DS = (perceptron(in_train, in_test, out_train, out_test))
     #   if (score_P_DS == 1):
     #       results_P_DS.append(score_P_DS)

     #   score_kNN1_DS = (knnClas(in_train, in_test, out_train, out_test, 1))
    #    if (score_kNN1_DS == 1):
     #       results_kNN1_DS.append(score_kNN1_DS)

        score_kNN3_DS = (knnClas(in_train, in_test, out_train, out_test, 3))
        #if (score_kNN3_DS == 1):
        results_kNN3_DS.append(score_kNN3_DS)

     #   score_kNN5_DS = (knnClas(in_train, in_test, out_train, out_test, 5))
     #   if (score_kNN5_DS == 1):
     #       results_kNN5_DS.append(score_kNN5_DS)

        score_MLP_DS = (MLP(in_train, in_test, out_train, out_test, (20, 20, 20, 20)))
        #if (score_MLP_DS == 1):
        results_MLP_DS.append(score_MLP_DS)



        """
        # Principal component analysis (PCA) with Standarzation
        pca = PCA(.95)
        inputTrainPCA = pca.fit_transform(inputTrain)

        in_train, in_test = inputTrainPCA[train], inputTrainPCA[test]
        out_train, out_test = outputTrain[train], outputTrain[test]


        score_LR_PCA = (logisticRegression(in_train, in_test, out_train, out_test))
        if( score_LR_PCA == 1):
            results_LR_PCA.append(score_LR_PCA)

        score_LGC_PCA = (linearGauClas(in_train, in_test, out_train, out_test))
        if (score_LGC_PCA == 1):
            results_LGC_PCA.append(score_LGC_PCA)

        score_P_PCA = (perceptron(in_train, in_test, out_train, out_test))
        if (score_P_PCA == 1):
            results_P_PCA.append(score_P_PCA)

        score_kNN1_PCA = (knnClas(in_train, in_test, out_train, out_test, 1))
        if (score_kNN1_PCA == 1):
            results_kNN1_PCA.append(score_kNN1_PCA)

        score_kNN3_PCA = (knnClas(in_train, in_test, out_train, out_test, 3))
        if (score_kNN3_PCA == 1):
            results_kNN3_PCA.append(score_kNN3_PCA)

        score_kNN5_PCA = (knnClas(in_train, in_test, out_train, out_test, 5))
        if (score_kNN5_PCA == 1):
            results_kNN5_PCA.append(score_kNN5_PCA)
        """

        score_MLP_PCA = (MLP(in_train, in_test, out_train, out_test, (20, 20, 20, 20)))
        #if (score_MLP_PCA == 1):
        results_MLP_PCA.append(score_MLP_PCA)


    """
    print('-------- DS data -------')
    print 'LR_Classifier Accuracy:  %.3f%%' % ((float(len(results_LR_DS))/104)*100)
    print 'LG_Classifier Accuracy:  %.3f%%' % ((float(len(results_LGC_DS))/104)*100)
    print 'P_Classifier Accuracy:  %.3f%%' % ((float(len(results_P_DS))/104)*100)
    print 'k1_Classifier Accuracy:  %.3f%%' % ((float(len(results_kNN1_DS))/104)*100)
    print 'k3_Classifier Accuracy:  %.3f%%' % ((float(len(results_kNN3_DS))/104)*100)
    print 'k5_Classifier Accuracy:  %.3f%%' % ((float(len(results_kNN5_DS))/104)*100)
    print 'MLP_Classifier Accuracy:  %.3f%%' % ((float(len(results_MLP_DS))/104)*100)


    print('-------- PCA data -------')
    print 'LR_Classifier Accuracy:  %.3f%%' % ((float(len(results_LR_PCA))/104)*100)
    print 'LG_Classifier Accuracy:  %.3f%%' % ((float(len(results_LGC_PCA))/104)*100)
    print 'P_Classifier Accuracy:  %.3f%%' % ((float(len(results_P_PCA))/104)*100)
    print 'k1_Classifier Accuracy:  %.3f%%' % ((float(len(results_kNN1_PCA))/104)*100)
    print 'k3_Classifier Accuracy:  %.3f%%' % ((float(len(results_kNN3_PCA))/104)*100)
    print 'k5_Classifier Accuracy:  %.3f%%' % ((float(len(results_kNN5_PCA))/104)*100)
    print 'MLP_Classifier Accuracy:  %.3f%%' % ((float(len(results_MLP_PCA))/104)*100)
    """


    #np.savetxt('results_LR_DS', results_LR_DS)
    np.savetxt('results_LGC_DS', results_LGC_DS)
    #np.savetxt('results_P_DS', results_P_DS)

    #np.savetxt('results_LR_PCA', results_LR_PCA)
    #np.savetxt('results_LGC_PCA', results_LGC_PCA)
    #np.savetxt('results_P_PCA', results_P_PCA)
    
    #np.savetxt('results_kNN1_DS', results_kNN1_DS)
    np.savetxt('results_kNN3_DS', results_kNN3_DS)
    #np.savetxt('results_kNN5_DS', results_kNN5_DS)
    #np.savetxt('results_kNN1_PCA', results_kNN1_PCA)
    #np.savetxt('results_kNN3_PCA', results_kNN3_PCA)
    #np.savetxt('results_kNN5_PCA', results_kNN5_PCA)

    np.savetxt('results_MLP_DS', results_MLP_DS)
    np.savetxt('results_MLP_PCA', results_MLP_PCA)



main()

