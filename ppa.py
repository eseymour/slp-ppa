#!/usr/bin/env python3

# Aleksandr Diamond and Edward Seymour, UTEP, September 2018
# Speech and Language Processing,
# Assignment C: Prepositional Phrase Attachment

# Based on a skeleton of code provided by Dr. Nigel Ward, UTEP, August 2018

# for the input files, the fields are:
#   ID, verb, noun1, preposition, noun2, attachment,
# for example:
#   0 join board as director V
#   1 is chairman of N.V. N

import sys, re  # regular expressions

def parsefile(filename):  #-----------------------------------------------------

  inputfp = open(filename, 'r')
  array = []
  for line in inputfp.readlines():
    array.append(line.split())
  return array

def evalPrint(predictions, labels, title): #------------------------------------
  print(f'\n--- Performance of {title} ---')
  matches = [x == y[5] for x, y in zip(predictions, labels)]
  print('  Accuracy is %.2f' % (1.0 * sum(matches) / len(predictions)))
  predVtrueV = [x == 'V' and y[5] == 'V' for x, y in zip(predictions, labels)]
  predVtrueN = [x == 'V' and y[5] == 'N' for x, y in zip(predictions, labels)]
  predNtrueV = [x == 'N' and y[5] == 'V' for x, y in zip(predictions, labels)]
  predNtrueN = [x == 'N' and y[5] == 'N' for x, y in zip(predictions, labels)]
  print('                true V   true N')
  print('  predicted V    %4d     %4d ' % (sum(predVtrueV), sum(predVtrueN)))
  print('  predicted N    %4d     %4d ' % (sum(predNtrueV), sum(predNtrueN)))


def countAttachments(data, attachmentType): #-----------------------------------
  return len([sample for sample in data if sample[5] == attachmentType])

def buildMajorityClassModel(data): #--------------------------------------------
  if countAttachments(data, 'V') > countAttachments(data, 'N'):
    return 'V'
  else:
    return 'N'

def runMajorityClassModel(model, data): #---------------------------------------
  predictions = [model] * len(data)
  return predictions

def computeVNratioFeature(sample): #--------------------------------------------
  verb = sample[1]
  noun1 = sample[2]
  return len(verb) / len(noun1)

def buildRatioModel(data): #----------------------------------------------------
  # a crazy model, based on the idea of more attachments to longer words
  sumVattachmentRatios = 0.0
  sumNattachmentRatios = 0.0
  for sample in data:
      VNratio = computeVNratioFeature(sample)
      if sample[5] == 'V':
          sumVattachmentRatios = sumVattachmentRatios + VNratio
      else:
          sumNattachmentRatios = sumNattachmentRatios + VNratio
  avgVratio = sumVattachmentRatios / countAttachments(data, 'V')
  avgNratio = sumNattachmentRatios / countAttachments(data, 'N')
  threshold = (avgVratio + avgNratio) / 2
  print('  Building the Ratio Model ... ')
  print('    for V attachments, ratio of V length to N1 length is %.2f' % avgVratio)
  print('    for N attachments, ratio of V length to N1 length is %.2f' % avgNratio)
  print('    so setting threshold to be %.2f ' % threshold)
  return threshold

def typeForSample(sample, threshold):  #----------------------------------------
   if computeVNratioFeature(sample) > threshold:
     return'V'
   else:
     return 'N'

def runRatioModel(threshold, data):  #------------------------------------------
  return [typeForSample(sample, threshold) for sample in data]


# main -------------------------------------------------------------------------

print('------ ppaStub ------\n')
trainData = parsefile('training')
model1 = buildMajorityClassModel(trainData)
model2 = buildRatioModel(trainData)

if len(sys.argv) >= 2 and (sys.argv[1] == 'yesThisReallyIsTheFinalRun'):
  testData = parsefile('test')
  finalPredictions = runRatioModel(model2, testData)
  predictions1 = runMajorityClassModel(model1, testData)
  evalPrint(predictions1, testData, 'Majority Model on testData')
  evalPrint(finalPredictions, testData, 'ratio model on test data')

else:
  devData = parsefile('devset')
  predictions1 = runMajorityClassModel(model1, devData)
  evalPrint(predictions1, devData, 'Majority Model on devData')
  predictions2 = runRatioModel(model2, devData)
  evalPrint(predictions2, devData, 'Ratio Model on devData')



