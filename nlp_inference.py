import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from numpy import zeros
import numpy as np
from numpy import asarray
from keras.layers import Embedding, concatenate

FILE_PATH = r'F:\ASU\sem3\nlp\train.jsonl' 
DEV_FILE_PATH = r'F:\ASU\sem3\nlp\dev.jsonl'
TEST_FILE_PATH = r'F:\ASU\sem3\nlp\test.jsonl' 
EMBEDDING_FILE_PATH = r'F:\ASU\sem3\nlp\glove.6B\glove.6B.50d.txt'
EPOCHS = 1000 
STEPS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0001

class Textual_Entailment():
    def __init__(self):
        self.tokenizer = None
        self.embedding_file = EMBEDDING_FILE_PATH
    
######################Reading Input File ################################
    
    def read_file(self, file_path):
        with open(file_path) as f:
            lines = f.read().strip().split("\n")
        lines = [json.loads(line) for line in lines]
        return lines

#############Converting Output Labels in One Hot Vector##################

    def score_setup(self, line):
        convert_dict = {
          'entailment': 0,
          'neutral': 1,
          'contradiction': 2
        }
        score = np.zeros((3,))
        for x in range(1,6):
            tag = line["gold_label"]
            if tag in convert_dict: score[convert_dict[tag]] += 1
        return score / (1.0*np.sum(score))

#############Extracting Premise, hypothesis and Gold_label ##############
  
    def input_out(self, lines):
        scores = []
        premise = [line['sentence1'] for line in lines]
        hypothesis = [line['sentence2'] for line in lines]
        output_label = [line['gold_label'] for line in lines]
        for line in lines:
            scores.append(self.score_setup(line))
        return premise, hypothesis, output_label, np.array(scores)

################Converting words to numbers by tokenizing ###############       
        
    def tokenize(self, inputs):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(inputs)             
        v_size = len(self.tokenizer.word_index) + 1     #vocab_size
        encoded_seq = self.tokenizer.texts_to_sequences(inputs)
        tokenized_input = pad_sequences(encoded_seq, maxlen=300)
        return tokenized_input, v_size 

############Creating Embedding layer using glove weights#################

    def embedding_layer(self, vocab_size):
        embeddings_index = dict()
        f = open(self.embedding_file, encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        
        embedding_matrix = zeros((vocab_size, 50))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
        
####################Computational graph of First NN########################
        
    def model_f(self, network_input, vocab_size):
        #Adding Embedding Layer
        embedding_matrix = self.embedding_layer(vocab_size)
        embed = tf.nn.embedding_lookup(embedding_matrix, network_input)

        #Defining weights and biases
        weights, biases = self.init_weights_n1(in_dim = 50)

        hidden = tf.add(tf.matmul(embed, weights['w1']), biases['b1'])
        hidden = tf.nn.relu(hidden)
        
        out_layer = tf.add(tf.matmul(hidden, weights['w2']), biases['b2'])
        out_layer = tf.nn.relu(out_layer)
        return out_layer

####################Computational graph of Second NN########################        

    def model_g(self, input):
        #Defining weights and biases
        weights, biases = self.init_weights_n2(in_dim = 600)   # 200 if out_dim in f is 100 ; 600 if out_dim in f is 300
        
        hidden = tf.nn.relu(tf.add(tf.matmul(input, weights['w1']), biases['b1']))
        
        out_layer = tf.add(tf.matmul(hidden, weights['w2']), biases['b2'])
        return out_layer

###########################Weight Initialization#############################

    def init_weights_n1(self, in_dim, hidden_n = 500, out_n=300):   #100 or 300
        weights = {
            'w1': tf.Variable(tf.random_normal([in_dim, hidden_n], dtype= tf.float64)),
            'w2': tf.Variable(tf.random_normal([hidden_n, out_n], dtype= tf.float64))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([hidden_n], dtype= tf.float64)),
            'b2': tf.Variable(tf.random_normal([out_n], dtype= tf.float64))
        }
        return weights, biases
        
    def init_weights_n2(self, in_dim, hidden_n = 100, out_n=3):  #50 or 100
        weights = {
            'w1': tf.Variable(tf.random_normal([in_dim, hidden_n], dtype= tf.float64)),
            'w2': tf.Variable(tf.random_normal([hidden_n, out_n], dtype= tf.float64))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([hidden_n], dtype= tf.float64)),
            'b2': tf.Variable(tf.random_normal([out_n], dtype= tf.float64))
        }
        return weights, biases

################## Creating complete Graph Architecture###### ################

def modelnn(premise, hypothesis, p_vocab_size, h_vocab_size):
    premise_rep = infer1.model_f(premise, p_vocab_size)
    hypothesis_rep = infer2.model_f(hypothesis, h_vocab_size)
    net_input = concatenate([premise_rep, hypothesis_rep], axis=1)
    network_2 = infer.model_g(net_input)
    return network_2

##########################Training variables###################################

infer = Textual_Entailment()
infer1 = Textual_Entailment()
infer2 = Textual_Entailment()
premise, hypothesis, output_label, scores = infer.input_out(infer.read_file(FILE_PATH))

######################## word to integers ######################################

premise_input, p_vocab_size = infer1.tokenize(premise)
hypothesis_input, h_vocab_size = infer2.tokenize(hypothesis)

print("Defining Placeholders")
placeholder_p = tf.placeholder(tf.int32, shape=[None])
placeholder_h = tf.placeholder(tf.int32, shape=[None])
placeholder_y = tf.placeholder(tf.int32, shape=[None, 3])

############################### Creating graph ##################################
pred = modelnn(placeholder_p, placeholder_h, p_vocab_size, h_vocab_size)

###################### Cost function and Optimization ###########################
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=(placeholder_y)))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

##############################Testing Variables #################################
inferx= Textual_Entailment()
infer3 = Textual_Entailment()
infer4 = Textual_Entailment()

premise_test, hypothesis_test, output_label_test, scores_test = inferx.input_out(inferx.read_file(TEST_FILE_PATH))
pre_test_input, p_test_vocab_size = infer3.tokenize(premise_test)
hyp_test_input, h_test_vocab_size = infer4.tokenize(hypothesis_test) 

###############################Dev variables ####################################
infer_dev= Textual_Entailment()
infer_dev1 = Textual_Entailment()
infer_dev2 = Textual_Entailment()

premise_dev, hypothesis_dev, output_label_dev, scores_dev = infer_dev.input_out(infer_dev.read_file(DEV_FILE_PATH))
pre_dev_input, p_dev_vocab_size = infer_dev1.tokenize(premise_dev)
hyp_dev_input, h_dev_vocab_size = infer_dev2.tokenize(hypothesis_dev) 

################################# RUN Model ######################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Initialized')
    
    for epoch in range(EPOCHS):
        avg_cost = 0
        total_batch = int(len(premise_input) / BATCH_SIZE)
        batch = np.random.randint(premise_input[0].shape[0], size=BATCH_SIZE)
		
        pr, hy, yc = (premise_input[0][batch],
                      hypothesis_input[0][batch],
                      scores[batch])
        _, c  = sess.run([optimizer, cost], 
                            feed_dict={
                                placeholder_p: pr, placeholder_h: hy, placeholder_y: yc
                            }) 
        avg_cost = c / total_batch
        if epoch % STEPS == 0:
            print("Epoch:{}__cost:{}".format(epoch+1, avg_cost))
    print("Optimization Finished!")
    
############################ Model Evaluation #####################################

########################### Train #################################################
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(placeholder_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={placeholder_p: pr, placeholder_h: hy, placeholder_y: yc})
    print("Training accuracy:{}".format(acc))
            
########################### DEV SET ###############################################
    prdev, hydev, ycdev = (pre_dev_input[0][batch],
                      hyp_dev_input[0][batch],
                      scores_dev[batch])
    dev_acc = sess.run(accuracy, feed_dict={placeholder_p: prdev, placeholder_h: hydev, placeholder_y: ycdev})
    print("Dev accuracy:{}".format(dev_acc)) 

##############################TEST SET#############################################

    prt, hyt, yct = (pre_test_input[0][batch],
                      hyp_test_input[0][batch],
                      scores_test[batch])
    testacc = sess.run(accuracy, feed_dict={placeholder_p: prt, placeholder_h: hyt, placeholder_y: yct})
    print("Test accuracy:{}".format(testacc)) 
