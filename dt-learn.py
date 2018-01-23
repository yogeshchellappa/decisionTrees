#!/usr/bin/env python
import sys
import math
from random import randint
from scipy.io import arff

"""
Node Class
Keeps track of all the nodes in the tree
"""
class Node():
	def __init__(self, data, feature, label, majority):
		self.data = data
		self.feature = feature
		self.label = label
		self.majority = majority
		self.children = []

	def addChild(self, child):
		self.children.append(child)

"""
Determines the candidate split for numeric and nominal features	
"""
def determineCandidateSplits(data, attributes):
	candidateSplits = []
	for i in range(len(attributes)):
		feature = attributes[i]
		attributeValues = list(set([record[i] for record in data]))
			
		# For numeric features
		if feature[1] == 'numeric': 
			if len(attributeValues) > 1:
				midpoints = []
				
				attributeValues.sort()
				
				for value in range(len(attributeValues) - 1):
					midpoints.append((float(attributeValues[value]) + float(attributeValues[value + 1]))/2)
					
				candidateSplit = [feature[0], i, midpoints]  
				candidateSplits.append(candidateSplit)
		
		# For nominal features
		elif feature[1] == 'nominal': 
			if len(attributeValues) > 1:
				candidateSplit = [feature[0], i, feature[2]] 	
				candidateSplits.append(candidateSplit)
	
	return candidateSplits

"""
This method finds the best split using information gain with the highest value among the features
"""
def findBestSplit(data, candidates, classificationLabels):	
	infoGain = 0
	featureWithHighestInfoGain = None
	
	targetLabels = [record[-1] for record in data]
	targetEntropy = getEntropy(targetLabels.count(classificationLabels[0]), targetLabels.count(classificationLabels[1]))
	
	# Go through all candidates and calculate entropy for each candidate
	for i in range(len(candidates)):
		attributeAndLabel = list(zip([record[candidates[i][1]] for record in data], [record[-1] for record in data]))

		# Numeric feature
		if isinstance(candidates[i][2][0], float): 
			for threshold in candidates[i][2]: 
				setOne = len([value for value in attributeAndLabel if value[0] <= threshold and value[1] == classificationLabels[0]]) 
				setTwo = len([value for value in attributeAndLabel if value[0] <= threshold and value[1] == classificationLabels[1]])
				entropyOne = getEntropy(setOne, setTwo)
				
				setThree = len([value for value in attributeAndLabel if value[0] > threshold and value[1] == classificationLabels[0]]) 
				setFour = len([value for value in attributeAndLabel if value[0] > threshold and value[1] == classificationLabels[1]])
				entropyTwo = getEntropy(setThree, setFour)
				
				featureEntropy = (entropyOne * (setOne + setTwo) + entropyTwo * (setThree + setFour))/(setOne + setTwo + setThree + setFour)
				
				gainForCurrentThreshold = targetEntropy - featureEntropy
				
				if gainForCurrentThreshold > infoGain:
					infoGain = gainForCurrentThreshold
					featureWithHighestInfoGain = [candidates[i][0], candidates[i][1], threshold]

		# Nominal feature
		else: 
			dataPartition = []
			entropies = []
			featureEntropy = 0
			
			for candidateValue in candidates[i][2]:
				setOne = len([value for value in attributeAndLabel if value[0] == candidateValue and value[1] == classificationLabels[0]]) 
				setTwo = len([value for value in attributeAndLabel if value[0] == candidateValue and value[1] == classificationLabels[1]])
				dataPartition.append([setOne, setTwo])				

			for partition in dataPartition:
				probabilityOfCurrentAttribute = (partition[0] + partition[1]) / float(len(data))
				entropy = getEntropy(partition[0], partition[1])
				featureEntropy += probabilityOfCurrentAttribute * entropy
			
			gain = targetEntropy - featureEntropy
			if gain > infoGain:
				infoGain = gain
				featureWithHighestInfoGain = candidates[i]
			
	if infoGain == 0:
		featureWithHighestInfoGain = None
		
	return featureWithHighestInfoGain
				
"""
Creates the decision tree
"""		
def makeSubTree(data, attributes, classificationLabels, stoppingCriteria, node):
	candidateSplits = determineCandidateSplits(data, attributes)
	targetLabels = [record[-1] for record in data]

	# Stopping criterion
	if (len(data) < stoppingCriteria or len(set(targetLabels)) == 1 or len(candidateSplits) == 0):
		createLeafNode(data, targetLabels, classificationLabels, node)

	else:
		bestSplit = findBestSplit(data, candidateSplits, classificationLabels) 
		
		# No positive information gain
		if bestSplit == None: 
			createLeafNode(data, targetLabels, classificationLabels, node)
		
		# Numeric features					
		elif isinstance(bestSplit[2], float):
			data1 = [record for record in data if float(record[bestSplit[1]]) <= bestSplit[2]]		
			classLabelOne, classLabelTwo, majority = getMajority(data1, classificationLabels, node)				
			childOne = Node([classLabelOne, classLabelTwo], bestSplit[0] + " <= " + str('%.6f' % round(bestSplit[2], 6)), None, majority)
			node.addChild(childOne)
			
			data2 = [record for record in data if float(record[bestSplit[1]]) > bestSplit[2]]
			classLabelOne, classLabelTwo, majority = getMajority(data2, classificationLabels, node)			
			childTwo = Node([classLabelOne, classLabelTwo], bestSplit[0] + " > " + str('%.6f' % round(bestSplit[2], 6)), None, majority)
			node.addChild(childTwo)
			
			makeSubTree(data1, attributes, classificationLabels, stoppingCriteria, childOne)
			makeSubTree(data2, attributes, classificationLabels, stoppingCriteria, childTwo)
		
		# Nominal features
		else:
			for attributeValue in bestSplit[2]:
				nominalData = [item for item in data if item[bestSplit[1]] == attributeValue]
				classLabelOne, classLabelTwo, majority = getMajority(nominalData, classificationLabels, node)		
				childNode = Node([classLabelOne, classLabelTwo], bestSplit[0] + " = " + attributeValue, None, majority)
				node.addChild(childNode)
				makeSubTree(nominalData, attributes, classificationLabels, stoppingCriteria, childNode)

"""
Calculates entropy
"""
def getEntropy(sampleSetOne, sampleSetTwo):
	entropy = 0
	
	# Both sampleSetOne and sampleSetTwo cannot be zero
	if sampleSetOne != 0 or sampleSetTwo != 0:	
		combinedSampleSize = sampleSetOne + sampleSetTwo
		probabilityDistribution = [sampleSetOne/float(combinedSampleSize), sampleSetTwo/float(combinedSampleSize)]
		
		for value in probabilityDistribution:
			if value != 0:
				entropy += -value * math.log(value)
				
	return entropy
				
"""
Gets the majority label
"""
def getMajority(inputData, classificationLabels, node = None):
	majorityLabel = None
	
	classLabelOne = len([record for record in inputData if record[-1] == classificationLabels[0]])
	classLabelTwo = len([record for record in inputData if record[-1] == classificationLabels[1]])
	
	if(classLabelOne == classLabelTwo):
		majorityLabel = node.majority if node else None
	else:
		majorityLabel = classificationLabels[0] if classLabelOne > classLabelTwo else classificationLabels[1]		

	return classLabelOne, classLabelTwo, majorityLabel

"""
Creates a leaf node
"""
def createLeafNode(inputData, targetLabels, classificationLabels, node):
	labelCount = [targetLabels.count(classificationLabels[0]), targetLabels.count(classificationLabels[1])]
	
	# Training instances is 0 or the leaf is equally represented
	if len(inputData) == 0 or labelCount[0] == labelCount[1]:
		node.label = node.majority
	else:
		node.label = classificationLabels[0] if labelCount[0] > labelCount[1] else classificationLabels[1]	
	
"""
Make class label predictions on test data and write to file
"""
def printPredictions(testFile, root, attributes, stoppingCriteria):
	predictedLabels = []
	correctPredictions = 0
	
	# Read the test file
	testData, testAtrributes = arff.loadarff(testFile)

	for record in testData:
		predictedLabel = predictOutcome(attributes ,root, record)
		if predictedLabel == record[-1]:
			correctPredictions += 1
		predictedLabels.append(predictedLabel)
			
	printDecisionTree(root, 0)
	
	print("<Predictions for the Test Set Instances>")
	for i in range(len(testData)):
		print(str(i + 1) + ": Actual: " + testData[i][-1]+ " Predicted: "+ predictedLabels[i])
	print("Number of correctly classified: " + str(correctPredictions) + " Total number of test instances: "+ str(len(testData)))

"""
Writes the decision tree to file
"""
def printDecisionTree(node, depth, isChildNode = False):
	if isChildNode == True:		
		indent = ""
		
		# Add the tabbing and pipe
		for i in range(depth):
			indent += "|" + "\t"
			
		statement = ""	
		if node.label != None:
			statement = indent + str(node.feature) + " [" + str(node.data[0]) + " " + str(node.data[1]) + "]: " + str(node.label)
		else:
			statement = indent + str(node.feature) + " [" + str(node.data[0]) + " " + str(node.data[1]) + "]"
		
		print statement

		depth += 1
		
	for child in node.children:
		printDecisionTree(child, depth, True)
	
"""
Predict the label for the test sample
"""		
def predictOutcome(attributes, node, record):
	while node.label == None:
		for child in node.children:
			vector = child.feature.split(" ")
			featureNames = [attribute[0] for attribute in attributes]
			index = featureNames.index(vector[0])
			
			# Nominal feature
			if vector[1] == "=": 
				if vector[2] == record[index]:
					node = child
				
			# Numeric feature
			else: 
				if (vector[1] == "<=" and float(record[index]) <= float(vector[2])) or (vector[1] == ">" and float(record[index]) > float(vector[2])):
					node = child
	return node.label

"""
Preprocess the data
"""	
def preprocessData(data, meta):
	# Create a copy of the data
	labeledData = data[:]
	
	# Get the attribute names, types and values
	mapping = []
	for count, attribute in enumerate(meta):
		mapping.append({attribute:count})
    
	attributes = []
	for i in mapping[:-1]:
		attributeName = i.keys()[0]
		attributeType =  meta[attributeName][0]
		attributeValues = list(meta[attributeName][1]) if meta[attributeName][1] is not None else None
		attributes.append([attributeName, attributeType, attributeValues])

	# Get the classificationLabels
	classificationLabels = list(meta['class'][1])
	
	return labeledData, attributes, classificationLabels
	
"""
Computes the accuracies for Part 2
"""	
def part2(labeledData, attributes, classificationLabels, testFile, stoppingCriteria = 4):
	trainingDatasize  = len(labeledData)
	trainingDataSubSampleSizes = [0.05 * trainingDatasize, 0.1 * trainingDatasize, 0.2 * trainingDatasize, 0.5 * trainingDatasize]
	trainingDataSubSampleSizes = map(lambda x: math.floor(x), trainingDataSubSampleSizes)
	
	for i in trainingDataSubSampleSizes:
		accuracies = []	
		for iteration in range(10): 
			numberGenerated = []
			trainingSubSample = []
			
			while(len(trainingSubSample) <= i):
				randomIndex = randint(0, len(labeledData)-1)
				if randomIndex not in numberGenerated:
					trainingSubSample.append(labeledData[randomIndex])
					numberGenerated.append(randomIndex)

			# We now have our reduced trainingData ready, create the root of the tree
			classLabelOne, classLabelTwo, majority = getMajority(trainingSubSample, classificationLabels)	
			root = Node([classLabelOne, classLabelOne], None, None, majority)
			makeSubTree(trainingSubSample, attributes, classificationLabels, stoppingCriteria, root)

			correctPredictions = 0

			# Read the test file
			testData, testAtrributes = arff.loadarff(testFile)
			for record in testData:
				predictedLabel = predictOutcome(attributes ,root, record)
				if predictedLabel == record[-1]:
					correctPredictions += 1
			
			accuracies.append(float(correctPredictions)/len(testData))
		
		print str(i) + " samples. " +  "Average: " + str(sum(accuracies)/10.0) + " Min: " + str(min(accuracies)) + " Max: " + str(max(accuracies))
		
"""
Driver
"""
def main():
	# Read values from command line
	trainingFile = sys.argv[1]
	testFile = sys.argv[2]
	stoppingCriteria = int(sys.argv[3])
	
	# Load the training data 
	data, meta = arff.loadarff(trainingFile)
	
	# Preprocessing of data
	labeledData, attributes, classificationLabels = preprocessData(data, meta)
	
	# Create the root of the tree
	classLabelOne, classLabelTwo, majority = getMajority(labeledData, classificationLabels)	
	root = Node([classLabelOne, classLabelOne], None, None, majority)
	
	makeSubTree(labeledData, attributes, classificationLabels, stoppingCriteria, root)
	
	printPredictions(testFile, root, attributes, stoppingCriteria)
	
	###
	# The following line runs part2 of the assignment and hence has been commented out. 
	# part2(labeledData, attributes, classificationLabels, testFile)
	###
	pass
	
main()



		
		

		
		
	