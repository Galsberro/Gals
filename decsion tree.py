import csv
import random
import numpy as np
import math
import pandas as pd
import scipy
from scipy.stats import chisquare


class tree:
    def __init__(self, function, input):
        if (function == 1): #build tree
            self.build_tree(input)
        elif (function==2): # k- fold cross validation
            self.tree_error(input)
        else:
            self.is_busy(input)



    def readData (self):#read all data
        self.data = pd.read_csv("Smoking.csv")
        self.data = self.data.drop("ID", axis=1)

    def build_tree(self,ratio):  # build tree
        self.nodes = [] # for pruning
        self.organizeData(ratio)
        attributes = []
        self.attributes = [] #remove maybe
        for i in range (0,24):
            attributes.append(i) #column index - questions
        tempData = self.learningSet.copy()
        self.tree = self.decision_tree_learning(tempData,attributes,0,tempData) #level = 1 - root
        self.pruneTree(self.tree,0,0)
        tempTestData = self.testSet.copy()
        if ratio == 1:
            print("All the data is for learning")
            return
        error_r = self.getErrorRate(self.tree,tempTestData)
        print(error_r)

    def pruneTree(self,node,father,fatherData):
            if all(x in [0, 1] for x in node.children): #final question only answers after
                if self.pruneCheck(node,father,father.dataSet):
                    return True
                return False #no need to prune
            else:
                for n in range (0,len(node.children)):
                    if not (node.children[n]== 0 or node.children[n] ==1):
                        if self.pruneTree(node.children[n],node,node.dataSet): #need to prune
                            node.children[n] = self.pluralityValue(node.dataSet) #instead of question

    def pruneCheck(self,node,father,fatherData):#prune / no
        tempData = node.dataSet #child data
        smokingRows = fatherData[fatherData['smoking'] == 1]
        T = len(smokingRows.index)
        F = len(fatherData.drop((smokingRows.index)))
        n = len(fatherData.index)
        nk = len(node.dataSet.index)
        T_k = nk*(T/n)
        F_k = nk*(F/n)
        delta = 0
        questionIndex = node.attIndex #question index
        criticalVal = scipy.stats.chi2.ppf(1 - .05, df=n - 1)  # critical val
        for v in range (0,len(node.attValues)):
            if questionIndex == 0 or questionIndex == 23: #string
                data = tempData[tempData.iloc[:, questionIndex] == node.attValues[v]]
                Tk = len(data[data['smoking'] == 1])  # smoking only
                Fk = len(data[data['smoking'] == 0]) #not smoking
                delta += ((((Tk-T_k)**2)/T_k)+(((Fk-F_k)**2)/F_k))
            else: # not categorical
                data = tempData[tempData.iloc[:, questionIndex] <= node.attValues[v]]
                Tk = len(data[data['smoking'] == 1])  # smoking only
                Fk = len(data[data['smoking'] == 0])  # not smoking
                delta += ((((Tk - T_k) ** 2) / T_k) + (((Fk - F_k) ** 2) / F_k))
                tempData = tempData.drop((data.index)) #update
        if delta<criticalVal:
            return True #need to prune
        return False

    def tree_error(self,k): #k = group num
        self.readData()
        self.nodes = []  # for pruning
        attributes = []
        for i in range(0, 24):
            attributes.append(i)  # column index - questions
        tempData = self.data.copy()
        dataGroups = self.createGroups(tempData,k)
        tempDataGroups = dataGroups.copy()
        errors = [] #collect error rates
        temp =0 #help variable
        for i in range (0,len(dataGroups)):#for each data set
            self.tree = self.decision_tree_learning(dataGroups[i],attributes,0,dataGroups[i]) #build tree
            self.pruneTree(self.tree, 0, 0)
            tempDataGroups.pop(i) #others groups
            errorRate = self.checkOthersData(self.tree,tempDataGroups)
            tempDataGroups = dataGroups.copy()
            errors.append(errorRate)
        for e in errors:
            temp+=e
        AVGerror = temp / len(errors)
        print(AVGerror)

    def is_busy(self,row_input):
        self.nodes = [] # for pruning
        row_input.pop(0) #remove ID
        row_input.pop(24)
        self.readData()
        attributes = []
        for i in range(0, 24):
            attributes.append(i)  # column index - questions
        tempData = self.data.copy()
        self.tree = self.decision_tree_learning(tempData, attributes, 0, tempData)  # level = 1 - root
        self.pruneTree(self.tree, 0, 0)
        ans = self.validateTree(row_input,self.tree)
        print(ans)

    def checkOthersData(self,tree,dataGroups):
        testData = dataGroups[0]
        for i in range (1,len(dataGroups)): #for each group need to test
            testData = pd.concat([testData,dataGroups[i]]) #unite data
        errorRate = self.getErrorRate(tree,testData)
        return errorRate

    def createGroups(self,tempData,k): #split data into groups
        dataSize = len(tempData.index)
        groupSize = round(dataSize / k) #integer
        groups = []  # enter the data groups into a list
        for i in range(0, k):  # number of iteration
            if i == (k - 1):  # last group - rest of the data
                group = tempData.copy()
                groups.append(group)
                continue
            group = tempData.sample(n=groupSize)
            groups.append(group)
            tempData = tempData.drop(group.index)
        return groups #ans

    def getErrorRate(self,tree,testSet): #get error rate
        correctAnswers = 0
        wrongAnswers = 0
        while (testSet.empty == False):
            first_row = testSet.iloc[0].tolist()
            testSet = testSet.drop(testSet.index[0])
            ans = self.validateTree(first_row, tree)
            if ans == first_row[24]:  # correct ans
                correctAnswers += 1
            else:
                wrongAnswers += 1
        error_Rate = wrongAnswers / (correctAnswers + wrongAnswers)
        return error_Rate

    def validateTree(self,row,tree):
        currQuestion = tree.attIndex
        testVal = row[currQuestion]
        curValues = tree.attValues
        flag = True #indicate if it is categorical attribute or no
        for v in range(0,len(curValues)):
            if testVal == curValues[v]:
                if tree.children[v] == 0 or tree.children[v] == 1:
                    return tree.children[v] #answer 0 / 1
                flag = False
                ans = self.validateTree(row,tree.children[v])# enter new node
                break
        if flag==True: #not categorical
            for v in range(0, len(curValues)):
                if testVal < curValues[v]:#find his bucket
                    if tree.children[v] == 0 or tree.children[v] == 1:
                        return tree.children[v]  # answer 0 / 1
                    else:
                        ans = self.validateTree(row,tree.children[v])# enter new node
                        break
                if v == (len(curValues)-1): #last bucket element - must go in
                    if tree.children[v] == 0 or tree.children[v] == 1:
                        return tree.children[v]  # answer 0 / 1
                    ans = self.validateTree(row,tree.children[v])# enter new node
        return ans

    def decision_tree_learning(self, data, attributes, level, fatherData):
        if data.empty: #no data - father plurality
            return self.pluralityValue(fatherData)
        elif data['smoking'].nunique() == 1: #same classification
            return data['smoking'].iloc[0] #ans
        elif len(attributes)==0: #attributes is empty
            return self.pluralityValue(data)
        else:
            tempData = data.copy()
            nextQuestionIndex = self.chooseBestAttribute(attributes,data) #find best question index
            tempAtt = attributes.copy()  # temp attribute list for no change
            tempAtt.remove(nextQuestionIndex)
            if self.checkIfCategorical(tempData, nextQuestionIndex):  # categorical
                diffAns = data.iloc[:, nextQuestionIndex].unique()
                node = Node(level, nextQuestionIndex, diffAns,tempData) #create node
                for att in diffAns:
                    exs = tempData[tempData.iloc[:, nextQuestionIndex] == att]  # update data
                    output = self.decision_tree_learning(exs, tempAtt, level + 1, data)  # return Child / ans
                    tempData = data.drop(exs.index)  # update data
                    node.addChild(output)
            else: #not categorical
                boarders = self.bucketSplit(nextQuestionIndex,tempData)
                node = Node(level, nextQuestionIndex, boarders,tempData)  # create node
                for b in range (0,2):
                    exs = tempData[tempData.iloc[:, nextQuestionIndex] <= boarders[b]]  # update data
                    output = self.decision_tree_learning(exs, tempAtt, level + 1, data)  # return Child / ans
                    node.addChild(output)
                    tempData = data.drop(exs.index)  # update data
        self.nodes.append(node) #for pruning
        return node

    def pluralityValue(self,data): #give ans base on the plurality
        smokingRows = data[data['smoking'] == 1]  # smoking only
        Tk = len(smokingRows.index)  # smoking number
        Fk = len(data.drop((smokingRows.index)))  # non smoking number
        if Tk>Fk:
            return 1
        if Tk<Fk:
            return 0
        return 0 #else

    def checkIfCategorical(self,data,nextQuestionIndex): #a function that check if the attributes is categorical
        if data.iloc[:, nextQuestionIndex].nunique() > 3:  # need to split to buckets
            return False #is not categorical
        else:
            return True #categorical

    def chooseBestAttribute (self, attributes, data): #choose next question
        fatherEntropy = self.calinitialEntropy(data)
        diffEntropy = 0 #entropy after the question
        for att in attributes: #run on the questions
            if self.checkIfCategorical(data,att): #categorical
                questionG = self.entropyForCategorical(data, att)
                if (fatherEntropy - questionG) >= diffEntropy:  # best question
                    bestQuestion = att  # update best question index so far
                    diffEntropy = fatherEntropy - questionG  # update best
            else: #not categorical
                questionGain = self.entropyForBuckets(data, att)
                if (fatherEntropy - questionGain) >= diffEntropy:
                    bestQuestion = att  # update best question index so far
                    diffEntropy = fatherEntropy - questionGain
        return bestQuestion

    def entropyForCategorical(self,data,att): #cal entropy for categorical attribute
        diffAns = data.iloc[:, att].unique()
        ans = []
        temp = 0
        for a in diffAns:
            attData = data[data.iloc[:, att] == a]  # all the data in the gap
            smokingRows = attData[attData['smoking'] == 1]  # smoking only
            Tk = len(smokingRows.index)  # smoking number
            Fk = len(attData.drop((smokingRows.index)))  # non smoking number
            Nk = len(attData.index)  # Nk number
            n = len(data.index)
            if Nk==0 or Tk==0 or Fk==0:
                ans.append(0)
                continue
            ans.append((Nk/n) * (-((Tk / Nk) * (math.log2(Tk / Nk)) + (Fk / Nk) * (math.log2(Fk / Nk)))))  # for every att-value
        for e in ans:
            temp +=e
        return temp

    def entropyForBuckets(self,data,att):#cal entropy for continuous attributes
        boardrs = self.bucketSplit(att,data)
        i = 0  #help variable - run on the bucket borders
        n = len(data.index)
        ans = [] #entropy for every value in attributes
        temp=0
        while i < len(boardrs):
            attData = data.loc[(data.iloc[:, att] <= boardrs[i])]  # all the data in the gap #all the data in the gap
            data = data.drop(attData.index)
            smokingRows = attData[attData['smoking'] == 1]  # smoking only
            Tk = len(smokingRows.index)  # smoking number
            Fk = len(attData.drop((smokingRows.index)))  # non smoking number
            Nk = len(attData.index)  # Nk number
            if Nk == 0 or Tk == 0 or Fk == 0:
                ans.append(0)
                i += 1
                continue
            i+=1
            ans.append((Nk / n) * ( -((Tk / Nk) * (math.log2(Tk / Nk)) + (Fk / Nk) * (math.log2(Fk / Nk)))))  # for every att-value
        for e in ans:
            temp += e
        return temp

    def bucketSplit(self,att,data): #choose the bucket
        maxValue = data.iloc[:, att].max()
        minValue = data.iloc[:, att].min()
        gap = (maxValue-minValue)/3
        return [minValue + (2 * gap), maxValue]  # gaps

    def organizeData (self, ratio): #organize the data - to learning data and test data
        self.readData()
        self.learningSet = self.data.sample(frac=ratio)  # chosen data
        self.testSet = self.data.drop(self.learningSet.index)  # rest of the data

    def calinitialEntropy (self,data): #initial entropy
        smokingRows = data[data["smoking"] == 1] #smoking only
        smokingNum = len(smokingRows.index) #smoking number
        nonSmokingNum = len(data.drop((smokingRows.index))) #non smoking number
        if smokingNum==0 or nonSmokingNum == 0: #got ans smoking / not
            return 0
        totalRows = len(data.index)
        return -((smokingNum/totalRows)*math.log2(smokingNum/totalRows)+(nonSmokingNum/totalRows)*math.log2(nonSmokingNum/totalRows))

class Node:
    def __init__(self,level,attIndex,attValues,dataSet):
        self.level = level
        self.attIndex = attIndex
        self.attValues = attValues
        self.children = []
        self.dataSet = dataSet # for pruning

    def addChild(self,child): #add child (subtree)
        self.children.append(child)


def build_tree(ratio):  # build tree
    sm = tree(1,ratio) # 1=build tree
def tree_error(k): # k- fold cross validation
    sm = tree(2,k)
def is_busy(row_input):
    sm=tree(3,row_input)
