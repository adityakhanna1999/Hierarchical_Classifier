import numpy as np
import csv
import random
import sys
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

# GENERAL PARAMETERS
MIN_CAT_REPRESENTATION = 500
CHI_SQUARE_SIZE = 10000

###############
# CHECK INPUT #
###############

# Enter min cat representation and chi square size manually in that order, if different from standard values
if len(sys.argv) == 3:
    MIN_CAT_REPRESENTATION = sys.argv[1]
    CHI_SQUARE_SIZE = sys.argv[2]

tokenizer = RegexpTokenizer(r'\w+')

##################
# HELPER METHODS #
##################

#Format given text to remove non alphabetic characters and make all words lowercase
def formatText(text):
    tokens = tokenizer.tokenize(text)
    returnString = ""
    for word in tokens:
        if not word.isdigit() and len(word) > 1:
            if len(returnString) > 0:
                returnString += ' '
            returnString += word.lower()
    return returnString

#Filter out all tokens that are not chosen in chi-square calculation, only retain tokens present in the 'tokens' variable
def filterTokens(summaries, tokens):
    new_summaries = {}
    for hadm in summaries:
        new_summary = ""
        text = summaries[hadm].split(' ')
        for word in text:
            if word in tokens:
                if len(new_summary) != 0:
                    new_summary += " "
                new_summary += word
        new_summaries[hadm] = new_summary
    return new_summaries

#Only retain elements with sufficient representation
def filterDictionairy(dict, treshold):
    selection = set()
    for key in dict.keys():
        if dict[key] >= treshold:
            selection.add(key)
    return selection


#Clean both dictionaries to only contain common keys
def cleanDictionaries(dict1, dict2):
    removeDict1 = set()
    for key in dict1.keys():
        if key not in dict2:
            removeDict1.add(key)
    for key in removeDict1:
        del dict1[key]
    removeDict2 = set()
    for key in dict2.keys():
        if key not in dict1:
            removeDict2.add(key)
    for key in removeDict2:
        del dict2[key]

def clean_preprocess(dict1):
	for keys in dict1.keys():
		s=dict1[keys]
		if(s.find('admission date ')!=-1):
			s=s.split('admission date ')[1]
		if(s.find('discharge date ')!=-1):
			s=s.split('discharge date ')[1]
		if(s.find('date of birth ')!=-1):
			s=s.split('date of birth ')[1]
		if(s.find('sex ')!=-1):
			s=s.split('sex ')[1]
		print(s);
		p=""
		words=s.split(' ')
		for word in words:
			if word != "newline" :
				p+=word
				p+=" "
		s=p
		if(s.find('end of report')!=-1):
			s=s.split('end of report')[0]
		dict1[keys]=s;

#Check whether keyword is not a part of and ICD code
def isValidKeyword():
    return True

#Load all ICD CODES into memory
filePathICD = "../MIMIC/D_ICD_DIAGNOSES.csv"
fileICD = open(filePathICD, 'r')
csv_reader_ICD = csv.reader(fileICD, delimiter=',')
all_ICD = set()
for row in csv_reader_ICD:
    all_ICD.add(row[1].strip())

#Select tokens based on Chi-Square values
def getChiSquareTokens(tokens, categories, summaries, diagnoses, size):
    table = []
    indexToken = {}
    index = 0
    for token in tokens:
        indexToken[token] = index
        index += 1
    indexCategory = {}
    index = 0
    for category in categories:
        indexCategory[category] = index
        index += 1
    for x in range(len(tokens)):
        table.append([0.0]*len(categories))
    countCategories = [0.0]*len(categories)
    countTokens = [0.0]*len(tokens)
    totalCount = 0.0
    for hadm in summaries:
        text = summaries[hadm].split(' ')
        cats = diagnoses[hadm]
        for t in text:
            for c in cats:
                if c not in categories:
                    continue
                table[indexToken[t]][indexCategory[c]] += 1.0
                countCategories[indexCategory[c]] += 1.0
                countTokens[indexToken[t]] += 1.0
                totalCount += 1.0
    scores = {}
    for token in tokens:
        if token in all_ICD:
            print('DELETED ICD token: ' + str(token))
            continue
        score = 0.0
        for cat in categories:
            calc = table[indexToken[token]][indexCategory[cat]]
            expect = (countTokens[indexToken[token]]*countCategories[indexCategory[cat]])/totalCount
            if calc == 0 or expect == 0:
                continue
            score += (calc-expect)*(calc-expect)/expect
        #Insert category name/ID removal
        scores[token] = score
    sortedTokens = [token for token in sorted(scores, key=scores.get, reverse=True)]
    return sortedTokens[:size]

def filterDiagnosesMinimumRepresentation(allDiagnoses, categories):
    diagnoses = {}
    for key in allDiagnoses.keys():
        selectedDiagnoses = set()
        for diag in allDiagnoses[key]:
            if diag in categories:
                selectedDiagnoses.add(diag)
        if len(selectedDiagnoses) > 0:
            diagnoses[key] = selectedDiagnoses
    return diagnoses

#Get category representation of the given ICD-code for the first level in the tree
def getL1(code):
    breakpoints = [140, 240, 280, 290, 320, 390, 460, 520, 580, 630, 680, 710, 740, 760, 780, 800, 1000]
    minBound = 1
    category = 'EV'
    if str(code)[0] != 'E' and str(code)[0] != 'V':
        index = 0
        cat = int(str(code)[:3])  # Only take first three digits in int format
        while (cat >= breakpoints[index]):
            minBound = breakpoints[index]
            index += 1
        category = str(minBound) + '-' + str(breakpoints[index]-1)
    return category

#Get category representation of the given ICD-code for the second level in the tree, E and V codes all directly under same node in L2 for now
def getL2(code):
    breakpoints = [0,30, 799, 807, 819, 825, 829, 838, 845, 848, 849, 858, 869, 876, 879, 888, 899, 909, 915, 928, 929, 949, 959, 969, 978, 979, 989, 999]
    minBound = 0
    if str(code)[0] == 'E':
        index = 0
        cat = int(str(code[1:4]))
        while (cat > breakpoints[index]):
            minBound = breakpoints[index]+1
            index += 1
        category = 'E'+str(minBound)+'-'+'E'+str(breakpoints[index])
        return category
    if str(code)[0] == 'V':
        return 'V'
    return str(code)[:3]

#Get category representation of the given ICD-code for the third level in the tree
def getL3(code):
    breakpoints = [6, 9, 19, 29, 39, 49, 59, 69, 82, 84, 85, 86, 87, 88, 89, 90, 91]
    minBound = 1
    if str(code)[0] == 'V':
        index = 0
        cat = int(str(code[1:3]))
        while (cat > breakpoints[index]):
            minBound = breakpoints[index]+1
            index += 1
        category = 'V'+str(minBound)+'-' +'V'+str(breakpoints[index])
        return category
    return str(code)[:4]

#Get category representation of the given ICD-code for the fourth level in the tree
def getL4(code):
    return code



############################################
# DATA LOADING AND NECESSARY PREPROCESSING #
############################################

#Load all NOTEEVENTS into memory
filePathData = "../MIMIC/NOTEEVENTS.csv"
fileData = open(filePathData, 'r')
csv_reader_data = csv.reader(fileData, delimiter=',')

#Load all DIAGNOSES_ICD into memory
filePathDiagnoses = "../MIMIC/DIAGNOSES_ICD.csv"
fileDiagnoses = open(filePathDiagnoses, 'r')
csv_reader_diagnoses = csv.reader(fileDiagnoses, delimiter=',')

#Load all NOTEEVENTS into memory
filePathDataTest = "NOTEEVENTS_MIMIC2.csv"
fileDataTest = open(filePathDataTest, 'r')
csv_reader_data_test = csv.reader(fileDataTest, delimiter=',')

#Load all DIAGNOSES_ICD into memory
filePathDiagnosesTest = "DIAGNOSES_ICD_MIMIC2.csv"
fileDiagnosesTest = open(filePathDiagnosesTest, 'r')
csv_reader_diagnoses_test = csv.reader(fileDiagnosesTest, delimiter=',')

#Make map from HADM to set with ICD-9 codes
print('Start reading Diagnoses...')
diagnosesAll = {}
categoryCounts = {}
trainHADMs = set()
trainInstances = open('TrainHADMs.txt', 'r') #TrainHADMS contain just numbers 
for line in trainInstances:
    trainHADMs.add(int(line.strip())) #whitespaces removed --- Contains all the HADMs numbers now
for row in csv_reader_diagnoses:  # MIMIC/Diagnoses_ICD 
    subject = row[1]
    hadm = row[2]
    if hadm == 'HADM_ID':
        continue  #first line hatadi
    icd = row[4]
    if subject == None or hadm == None or icd == None or len(subject) == 0 or len(hadm) == 0 or len(icd) == 0 or int(hadm) not in trainHADMs:
        continue
    if hadm not in diagnosesAll:
        diagnosesAll[hadm] = [] #added to dictionary
    if icd not in categoryCounts:
        categoryCounts[icd] = 0
    categoryCounts[icd] += 1
    diagnosesAll[hadm].append(icd) # contains all the ICD codes corresponding to each hadm value now
testHADMs = set()
testInstances = open('TestHADMs.txt', 'r')
print('DONE reading Diagnoses without test instances: ' + str(len(diagnosesAll)))
for line in testInstances:
    testHADMs.add(int(line.strip()))
for row in csv_reader_diagnoses_test:
    #subject = row[1]
    hadm = row[0]
    if hadm == 'HADM_ID':
        continue
    icd = row[1]
    if hadm == None or icd == None or len(hadm) == 0 or len(icd) == 0 or int(hadm) not in testHADMs:
        continue
    if hadm not in diagnosesAll:
        diagnosesAll[hadm] = []
    if icd not in categoryCounts:
        categoryCounts[icd] = 0
    categoryCounts[icd] += 1
    diagnosesAll[hadm].append(icd)
print('DONE reading Diagnoses: ' + str(len(diagnosesAll)))


print('Start filtering categories on minimum representation...')
categories = filterDictionairy(categoryCounts, MIN_CAT_REPRESENTATION)  #yeh samaj
diagnoses = filterDiagnosesMinimumRepresentation(diagnosesAll, categories) #Basically is sabke baad wohi hai jinka ICD code MIN_CAT times hai, dictionary mei and wo ICD codes ka count
print('DONE filtering categories on minimum representation: ' + str(len(categories)))

#Make map from HADMs that have known diagnoses to textual representations of their discharge summaries
print('Start reading discharge summaries...')  
summaries = {}
for row in csv_reader_data:   #NOTE_EVENTS_DATA file 
    subject = row[1]
    hadm = row[2]
    if hadm not in diagnoses:
        continue
    category = row[6]
    isError = row[9]
    if isError:
        continue
    if category != 'Discharge summary':
        continue
    text = formatText(row[10])
    summaries[hadm] = text
for row in csv_reader_data_test:
    #subject = row[1]
    hadm = row[0]
    if hadm == 'HADM_ID':
        continue
    if hadm not in diagnoses:
        continue
    category = row[1]
    isError = bool(row[2])
    #if isError:
    ##    continue
    if category != 'DISCHARGE_SUMMARY':
        continue
    text = formatText(row[3])
    summaries[hadm] = text
print('DONE reading discharge summaries: ' + str(len(summaries)))
#clean the summaries for extra words
clean_preprocess(summaries)


#Make map from token to input index (possibly a subset generated by Chi Square analysis)
allTokens = set() #All tokens in original data
tokenToIndex = {} #Selected tokens with their corresponding index
for hadm in summaries:
    words = summaries[hadm].split(' ')
    for word in words:
        allTokens.add(word)
print('Started Chi Square values calculation...')
#UNCOMMENT first line below to use CHI SQUARE and comment the second line below
tokens = getChiSquareTokens(allTokens, categories, summaries, diagnoses, CHI_SQUARE_SIZE)
#tokens = allTokens
print('DONE with Chi Square values calculation')

print('Start filtering tokens...')
summaries = filterTokens(summaries, tokens)
print('DONE filtering tokens')

print('Start cleaning dictionaries...')
cleanDictionaries(diagnoses, summaries)
print('DONE cleaning dictionaries: ' + str(len(diagnoses)) + ' ' + str(len(summaries)))


print('Start writing data to files...')
clean_data_file = open('clean_data_BOW_2' + str(MIN_CAT_REPRESENTATION) + '.txt', 'w') #+ str(CHI_SQUARE_SIZE)+ '.txt', 'w')
for hadm in summaries:
    clean_data_file.write(str(hadm) + '|' + str(summaries[hadm]) + '|' + str(diagnoses[hadm]) + '\n')
clean_data_file.close()
chi_square_file = open('chi_square_map_2_' + str(MIN_CAT_REPRESENTATION) + '-' + str(CHI_SQUARE_SIZE) + '.txt', 'w')
for token in tokens:
   chi_square_file.write(token + '\n')
chi_square_file.close()
print('DONE writing data to files')
