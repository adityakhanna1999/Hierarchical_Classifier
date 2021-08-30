import numpy as np
import gc
import random
import sys
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape, Embedding, Conv1D, MaxPooling1D, LSTM, TimeDistributed
from tensorflow.compat.v1.keras.layers import  CuDNNLSTM
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
# import keras.backend as K
from scipy import spatial
from tensorflow.compat.v1.keras.layers import Embedding



# GENERAL PARAMETERS
MIN_CAT_REPRESENTATION = 500
CHI_SQUARE_SIZE = 10000
SEQ_NB = -2

# MODEL PARAMETERS
MAX_NODE_WIDTH = 5

TREE_SELECTION = 1


# INPUT PARAMETERS
INPUT_SEQ_LEN = 10000 #change
VOC_SIZE = -1


def filter_top_diagnoses(amount, summaries, diagnoses):
    new_summaries = {}
    new_diagnoses = {}
    diagnose_counts = {}
    for hadm in diagnoses:
        for diagnose in diagnoses[hadm]:
            if diagnose not in diagnose_counts:
                diagnose_counts[diagnose] = 0
            diagnose_counts[diagnose] += 1
    index = 0
    chosen_diagnoses = []
    while index < amount:
        best = None
        best_amount = -1
        for diagnose in diagnose_counts:
            if not diagnose in chosen_diagnoses:
                if diagnose_counts[diagnose] > best_amount:
                    best_amount = diagnose_counts[diagnose]
                    best = diagnose
        chosen_diagnoses.append(best)
        index += 1
    for hadm in diagnoses:
        old_diagnose = diagnoses[hadm]
        new_diagnose = []
        for diagnose in old_diagnose:
            if diagnose in chosen_diagnoses:
                new_diagnose.append(diagnose)
        if len(new_diagnose) > 0:
            new_diagnoses[hadm] = new_diagnose
            new_summaries[hadm] = summaries[hadm]
    return (new_summaries, new_diagnoses)


print('Started reading data...')
data_file = open('Dis_50_Dataset.txt', 'r') #BOW
# data_file = open('clean_data_500-10000' + '.txt', 'r')  # Chi Square
summaries = {}
diagnoses = {}
for line in data_file.readlines():
    splitted = line.split('|')
    hadm = splitted[0].strip()
    summary = splitted[1].strip()
    diagnose = eval(splitted[2].strip())
    summaries[hadm] = summary
    diagnoses[hadm] = diagnose

print('Summaries', len(summaries))
print('diagnoses', len(diagnoses))
# summaries, diagnoses = filter_top_diagnoses(50, summaries, diagnoses)
print('Summaries', len(summaries))
print('diagnoses', len(diagnoses))
print(diagnoses['110220'])
print(diagnoses['167853'])

chi_square_file = open('chi_square_map_4_500-10000.txt', 'r')
chi_square_map = {}
chi_count = 0
for line in chi_square_file.readlines():
    chi_square_map[line.strip()] = chi_count
    chi_count += 1
print('DONE loading data')


def splitTrainTest(summaries, diagnoses):
    trainInput = {}
    trainOutput = {}
    testInput = {}
    testOutput = {}
    validationInput={}
    validationOutput={}
    indices = list(range(len(summaries)))
    testHADMs = set()
    trainHADMs=set()
    validationHADMs=set()
    random.shuffle(indices)
    testInstances = open('test_50_hadm_ids.csv', 'r')
    for line in testInstances:
        testHADMs.add(int(line.strip()))
    trainInstances = open('train_50_hadm_ids.csv', 'r')
    for line in trainInstances:
        trainHADMs.add(int(line.strip()))
    validationInstances = open('dev_50_hadm_ids.csv', 'r')
    for line in validationInstances:
        validationHADMs.add(int(line.strip()))
    print(len(testHADMs))
    print(len(trainHADMs))
    print(len(validationHADMs))
    counter=0
    # index = 0
    for hadm in summaries:
        if int(hadm) in trainHADMs:
            trainInput[hadm] = summaries[hadm]
            trainOutput[hadm] = diagnoses[hadm]
        if int(hadm) in testHADMs:
            testInput[hadm] = summaries[hadm]
            testOutput[hadm] = diagnoses[hadm]
        if int(hadm) in validationHADMs:
            validationInput[hadm]=summaries[hadm]
            validationOutput[hadm]=diagnoses[hadm]

        # index += 
    return (trainInput, trainOutput, testInput, testOutput, validationInput, validationOutput)


print('Start making train/test split...')
(summaries, diagnoses, test_summaries, test_diagnoses, validation_summaries, validation_diagnoses) = splitTrainTest(summaries, diagnoses)
print('DONE making train/test split: ' + str(len(summaries)) + ' ' + str(len(test_summaries))+ ' ' + str(len(validation_summaries)))

#Get category representation of the given ICD-code for the first level in the tree
def getL1(code):
    breakpoints = [140, 240, 280, 290, 320, 390, 460, 520, 580, 630, 680, 710, 740, 760, 780, 800, 1000]
    breakpoints_P = ["01", "06", "08", 18, 21, 30, 35, 40, 42, 55, 60, 65, 72, 76, 85, 87, 100]
    minBound = 1
    minBound_P = 0
    category = 'EV'
    # print('CODE', code)
    if str(code)[0] != 'E' and str(code)[0] != 'V' and str(code)[0] != 'P':
        index = 0
        cat = int(str(code)[:3])  # Only take first three digits in int format
        while (cat >= int(breakpoints[index])):
            minBound = int(breakpoints[index])
            index += 1
        category = str(minBound) + '-' + str(int(breakpoints[index]-1))
    elif str(code)[0] == 'P':
        index = 0
        cat = int(str(code[1:3]))  # Only take first 2 int digits in int format
        while (cat >= int(breakpoints_P[index])):
            minBound_P = int(breakpoints_P[index])
            index += 1
        category = 'P'+str(minBound_P) + '-' + 'P'+str(int(breakpoints_P[index]) - 1)
    return category

#Get category representation of the given ICD-code for the second level in the tree, E and V codes all directly under same node in L2 for now
def getL2(code):
    breakpoints = ["000", "030", 799, 807, 819, 825, 829, 838, 845, 848, 849, 858, 869, 876, 879, 888, 899, 909, 915, 928, 929, 949, 959, 969, 978, 979, 989, 999]
    minBound = 0
    if str(code)[0] == 'E':
        index = 0
        cat = int(str(code[1:4]))
        while (cat > int(breakpoints[index])):
            minBound = int(breakpoints[index])+1
            index += 1
        category = 'E'+str(minBound)+'-'+'E'+str(int(breakpoints[index]))
        return category
    if str(code)[0] == 'V':
        return 'V'
    return str(code)[:3]

    #Get category representation of the given ICD-code for the third level in the tree
def getL3(code):
    breakpoints = ["06", "09", 19, 29, 39, 49, 59, 69, 82, 84, 85, 86, 87, 88, 89, 90, 91]
    minBound = 1
    if str(code)[0] == 'V':
        index = 0
        cat = int(str(code[1:3]))
        while (cat > int(breakpoints[index])):
            minBound = int(breakpoints[index])+1
            index += 1
        category = 'V'+str(minBound)+'-' +'V'+str(int(breakpoints[index]))
        return category
    if getL2(code) == (code):
        return code
    return str(code)[:5] #including a dot now

    #Get category representation of the given ICD-code for the fourth level in the tree
def getL4(code):
    return code

def precision(y_true, y_pred):
    """Precision metric.
     Only computes a batch-wise average of precision.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
     Only computes a batch-wise average of recall.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
#@tf.function
def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
     The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
     With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
     # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == tf.constant(0,dtype=tf.float32):
        return 0.0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score
#@tf.function
def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
     Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

def function(GLOBAL_LOSS_WEIGHT = 1,BATCH_SIZE = 4,EPOCHS = 25,Embd_dim_Dense = 32,Embd_dim_CNN = 16,Embd_dim_LSTM = 128,dropout_rate_Dense = 0.25,
                 dropout_rate_CNN = 0.5,dropout_rate_LSTM = 0.75,First_dense_dim_Dense = 128,Second_dense_dim_Dense = 128,First_dense_dim_CNN = 64,
                 First_dense_dim_LSTM = 64,Filters = 128,Kernel_Size = 5,LstmHiddenUnit = 64,INPUT_SEQ_LEN = 10000, L = [1, 1, 1,1]):
    # L = [1, 1, 1,1]
    # K.clear_session()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(session)


    class Node1:
        def __init__(self, level, n):
            self.node_level = level
            self.name = n #Name of lexicographically first child
            self.children = []
            self.lower = None
        def get_name(self):
            return self.name
        def get_level(self):
            return self.level
        def set_children(self, ch):
            self.children = ch
        def get_children(self):
            return self.children

        def build_model_1(self, common_input):
            if L[self.node_level] == 0:
                if self.node_level == 3:
                    model = Embedding(input_dim=VOC_SIZE, output_dim=Embd_dim_Dense, input_length=INPUT_SEQ_LEN)(common_input)
                    model = Flatten()(model)
                    model = Dense(units=First_dense_dim_Dense, activation='relu')(model)
                else:
                    model = Flatten()(common_input)
                    model = Dense(units=First_dense_dim_Dense, activation='relu')(model)
                model = Dropout(rate=dropout_rate_Dense)(model)
                next_layer_input = Dense(units=Second_dense_dim_Dense, activation='relu')(model)
                model = Dense(units=len(self.children), activation='sigmoid')(model)

            elif L[self.node_level] == 1:
                if self.node_level == 0:
                    model = Embedding(input_dim=VOC_SIZE, output_dim=Embd_dim_CNN, input_length=INPUT_SEQ_LEN)(common_input)
                    model = Conv1D(Filters, kernel_size=Kernel_Size, strides=2, padding='valid', kernel_initializer='normal',
                                   activation='relu')(model)
                else:
                    model = Conv1D(Filters, kernel_size=Kernel_Size, strides=2, padding='valid', kernel_initializer='normal',
                                   activation='relu')(common_input)
                model = MaxPooling1D(pool_size=4, strides=2, padding='valid')(model)
                next_layer_input = model
                model = Flatten()(model)
                model = Dropout(rate=dropout_rate_CNN)(model)
                model = Dense(units=First_dense_dim_CNN, activation='relu')(model)
                #next_layer_input = Dense(units=32, activation='relu')(model)
                model = Dense(units=len(self.children), activation='sigmoid')(model)

            else:
                if self.node_level == 3:
                    model = Embedding(input_dim=VOC_SIZE, output_dim=Embd_dim_LSTM, input_length=INPUT_SEQ_LEN)(common_input)
                    model = CuDNNLSTM(LstmHiddenUnit, return_sequences=True)(model)
                else:
                    model = CuDNNLSTM(64, return_sequences=True)(common_input)
                model = MaxPooling1D(pool_size=4, strides=2, padding='valid')(model)
                next_layer_input = model
                model = Flatten()(model)
                model = Dropout(rate=dropout_rate_LSTM)(model)
                model = Dense(units=First_dense_dim_LSTM, activation='relu')(model)
                #next_layer_input = Dense(units=32, activation='relu')(model)
                model = Dense(units=len(self.children), activation='sigmoid')(model)

            if self.node_level > 0:
                model = concatenate([model, self.lower.build_model_1(next_layer_input)])
            return model

        #TODO: change to include specific layer as a parameter --> done
        def get_parent_index_1_recurse(self, start_index, parents, parent_layer):
            parent_indices = [-1]*len(self.children)
            for x in range(len(self.children)):
                for y in range(len(parents)):
                    if parent_layer == 3:
                        if getL1(self.children[x]) == parents[y]:
                            parent_indices[x] = start_index+y
                    elif parent_layer == 2:
                        if getL2(self.children[x]) == parents[y]:
                            parent_indices[x] = start_index + y
                    elif parent_layer == 1:
                        if getL3(self.children[x]) == parents[y]:
                            parent_indices[x] = start_index + y
            if self.node_level == 0:
                return parent_indices
            parent_indices.extend(self.lower.get_parent_index_1_recurse(start_index+len(parents), self.children, self.node_level))
            return parent_indices

        #NOTE CHANGE THIS FUNCTION --> fixed
        def get_parent_index_1(self):
            parent_indices = list(range(len(self.children)))
            skip = len(self.children)
            if self.node_level > 0:
                parent_indices.extend(self.lower.get_parent_index_1_recurse(0,self.children, self.node_level))
            return parent_indices

        #TODO: Change this function for upper layers --> quick fixed
        def get_output_1(self, diagnoses):
            if self.node_level == 0:
                outp = []
                for x in range(len(self.children)):
                    child = self.children[x]
                    if child in diagnoses:
                        outp.append(1)
                    else:
                        outp.append(0)
            else:
                parent_diagnoses = set()
                for diag in diagnoses:
                    parent_diagnoses.add(getL1(diag))
                    parent_diagnoses.add(getL2(diag))
                    parent_diagnoses.add(getL3(diag))
                outp = []
                local_outp = []
                for x in range(len(self.children)):
                    child = self.children[x]
                    if child in parent_diagnoses:
                        local_outp.append(1)
                    else:
                        local_outp.append(0)
                outp = self.lower.get_output_1(diagnoses)
                outp = local_outp + outp
            return outp

        #TODO: Ignores parent at the moment, possibly change this -->fixed
        def get_l_output_recurse(self, model_output, index, layer_level):
            l_output = []
            if self.node_level == layer_level:
                for x in range(len(self.children)):
                    l_output.append(model_output[index+x])
                return l_output
            elif self.node_level > layer_level:
                return self.lower.get_l_output_recurse(model_output, index + len(self.children), layer_level)
        def get_l_output(self, model_output, layer_level):
            return self.get_l_output_recurse(model_output, 0, layer_level)


        def get_tree_size(self):
            if self.node_level == 0:
                return len(self.children)
            size = len(self.children)
            for x in range(len(self.children)):
                child = self.children[x]
                size += child.get_tree_size()
            return size

    # Hierarchical loss on the child (only child node gets affected)
    #TODO: change this function to better work with parent dependencies --> fixed
    def hierarchical_loss_parent(tree, weight):
        def loss(y_true, y_pred):
            #Calculating y_parent_pred
            y_parent_indices = tree.get_parent_index_1()
            y_pred_transpose = K.permute_dimensions(y_pred, (1,0))
            y_parent_pred_transpose = []
            for x in range(len(y_parent_indices)):
                y_parent_pred_transpose.append(y_pred_transpose[y_parent_indices[x]])
            y_parent_pred_transpose = K.stack(y_parent_pred_transpose)
            y_parent_pred = K.permute_dimensions(y_parent_pred_transpose, (1,0))
            #Calculate and combine different losses
            hier_loss = K.mean(K.clip(y_pred - y_parent_pred, 0, 1), axis=-1)
            bin_crossentropy_loss = binary_crossentropy(y_true, y_pred)
            total_loss = weight*bin_crossentropy_loss + (1.0-weight)*hier_loss
            return total_loss
        return loss

    def get_IO_Old(summaries, diagnoses, tree, size):
        count = 0
        inputs = []
        outputs = []
        for hadm in summaries:
            if count == size:
                break
            if count == len(summaries):
                break
            count += 1
            summary = summaries[hadm]
            local_input = [0]*CHI_SQUARE_SIZE
            for token in summary.split():
                local_input[chi_square_map[token]] = 1
            inputs.append(local_input)
            diagnose = diagnoses[hadm]
            outputs.append(tree.get_output_1(diagnose))
        return (np.array(inputs), np.array(outputs))

    def get_index_map(summaries_train, summaries_test,summaries_validation, size):
        count = 0
        index = {}
        index_count = 0
        token_set = set()
        for hadm in summaries_train:
            if count == size:
                break
            if count == len(summaries_train):
                break
            count += 1
            summary = summaries_train[hadm]
            for token in summary.split():
                if token not in token_set:
                    token_set.add(token)
                    index[token] = index_count
                    index_count += 1
        count = 0
        for hadm in summaries_test:
            if count == size:
                break
            if count == len(summaries_test):
                break
            count += 1
            summary = summaries_test[hadm]
            for token in summary.split():
                if token not in token_set:
                    token_set.add(token)
                    index[token] = index_count
                    index_count += 1
        count=0
        for hadm in summaries_validation:
            if count == size:
                break
            if count == len(summaries_validation):
                break
            count += 1
            summary = summaries_validation[hadm]
            for token in summary.split():
                if token not in token_set:
                    token_set.add(token)
                    index[token] = index_count
                    index_count += 1
        global VOC_SIZE
        VOC_SIZE = index_count + 1
        return index

    def get_IO(summaries, diagnoses, tree, size, index_map):
        inputs = []
        outputs = []
        index = index_map
        count = 0
        for hadm in summaries:
            if count == size:
                break
            if count == len(summaries):
                break
            count += 1
            summary = summaries[hadm]
            local_input = [0]*INPUT_SEQ_LEN
            token_count = 0
            for token in summary.split():
                if token_count == INPUT_SEQ_LEN:
                    break
                local_input[token_count] = index[token]+1
                token_count += 1
            #local_input = local_input[::-1]
            #print(local_input)
            inputs.append(local_input)
            diagnose = diagnoses[hadm]
            outputs.append(tree.get_output_1(diagnose))
        return (np.array(inputs), np.array(outputs))

    # make tree following the exact hierarchical structure of te ICD tree
    def construct_tree():
        unique_diagnoses = set()
        unique_L1 = set()
        unique_L2 = set()
        unique_L3 = set()
        for hadm in diagnoses:
            for diagnose in diagnoses[hadm]:
                unique_diagnoses.add(diagnose)
                unique_L1.add(getL1(diagnose))
                unique_L2.add(getL2(diagnose))
                unique_L3.add(getL3(diagnose))
        print(unique_diagnoses)
        print('L1: ', len(unique_L1))
        print(sorted(unique_L1))
        print('L2: ', len(unique_L2))
        print('L3: ', len(unique_L3))
        print('L4: ', len(unique_diagnoses))
        tree = Node1(0, 'root') #highest node has level 4 in this case
        # L1 = Node1(2, "L1")
        # tree.lower = L1
        # for L1_code in unique_L1:
        #     tree.children.append(L1_code)
        # L2 = Node1(1, "L2")
        # L1.lower = L2
        # for L2_code in unique_L2:
        #     L1.children.append(L2_code)
        # L3 = Node1(0, "L3")
        # L2.lower = L3
        # for L3_code in unique_L3:
        #     L2.children.append(L3_code)
        for L4_code in unique_diagnoses:
            tree.children.append(L4_code)
        return tree

    print('Start constructing the tree and retrieve IO reps of data...')
    tree = construct_tree()
    index_map = get_index_map(summaries, test_summaries,validation_summaries, 500000)
    (inputs, outputs) = get_IO(summaries, diagnoses, tree, 500000, index_map)
    (test_inputs, test_outputs) = get_IO(test_summaries, test_diagnoses, tree, 500000, index_map)
    validation_inputs,validation_outputs=get_IO(validation_summaries,validation_diagnoses, tree, 500000, index_map)

    print('DONE constructing the tree and retrieving IO reps of data')

    import time

    # print('Start building model...')
    start_training_time = time.time()
    common_input = Input(shape=(INPUT_SEQ_LEN,), dtype='int32')

    model = tree.build_model_1(common_input)
    # print("Done part 1")
    model = Model(common_input, model)
    # print(model.summary())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall, fmeasure])
    history = model.fit(inputs, outputs, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_data=(validation_inputs,validation_outputs), shuffle=True)
    training_time = time.time() - start_training_time
    
    fval = sum(history.history['val_fmeasure'])/EPOCHS
    def evaluate_model_ul(test_inputs, test_outputs, tree, layer_level):
        (amount_of_samples, output_size) = test_outputs.shape
        predictions = model.predict(test_inputs)
        ll_predictions = []
        for x in range(amount_of_samples):
            ll_predictions.append(tree.get_l_output(predictions[x],layer_level))
        predictions = ll_predictions
        ll_test_outputs = []
        for x in range(amount_of_samples):
            ll_test_outputs.append(tree.get_l_output(test_outputs[x],layer_level))
        test_outputs = ll_test_outputs
        output_size = len(test_outputs[0])
        tp = 0.00000001
        tn = 0.00000001
        fp = 0.00000001
        fn = 0.00000001
        for x in range(amount_of_samples):
            for y in range(output_size):
                pred = predictions[x][y]
                outp = test_outputs[x][y]
                if pred > 0.25:
                    if outp > 0.25:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if outp > 0.25:
                        fn += 1
                    else:
                        tn += 1
        acc = round(100*(tp+tn)/(tp+tn+fp+fn), 2)
        prec = round(100*tp/(tp+fp), 2)
        rec = round(100*tp/(tp+fn), 2)
        f1 = round(2*prec*rec/(prec+rec), 2)
        return (acc, prec, rec, f1)



    # print('Start evaluating model...')
    (acc0, prec0, rec0, f10) = evaluate_model_ul(test_inputs, test_outputs, tree,0)
    (acc1, prec1, rec1, f11) = evaluate_model_ul(test_inputs, test_outputs, tree,1)
    (acc2, prec2, rec2, f12) = evaluate_model_ul(test_inputs, test_outputs, tree,2)
    (acc3, prec3, rec3, f13) = evaluate_model_ul(test_inputs, test_outputs, tree,3)
    print(acc0, prec0, rec0, f10)
    print(acc1, prec1, rec1, f11)
    print(acc2, prec2, rec2, f12)
    print(acc3, prec3, rec3, f13)

    print('ACC', acc0, 'PREC', prec0, 'REC', rec0, 'F1', f10)
    
    del model
    gc.collect()
    session.close()
    K.clear_session()
    return acc0, prec0, rec0, f10,acc1, prec1, rec1, f11,acc2, prec2, rec2, f12, acc3, prec3, rec3, f13,fval

acc0, prec0, rec0, f10, acc1, prec1, rec1, f11, acc2, prec2, rec2, f12, acc3, prec3, rec3, f13, fval = function(L = [1,1,1,1])

