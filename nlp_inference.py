import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from numpy import zeros
import numpy as np
from keras.layers import Embedding

FILE_PATH = r'F:\ASU\sem3\nlp\dev.jsonl' 
EMBEDDING_FILE_PATH = r'F:\ASU\sem3\nlp\glove.6B\glove.6B.50d'

class Inference():
    def __init__(self):
        self.tokenizer = None
        
    def read_file(self, file_path):
        with open(file_path) as f:
            lines = f.read().strip().split("\n")
        lines = [json.loads(line) for line in lines]
        return lines

    def input_out(self, lines):
        premise = [line['sentence1'] for line in lines]
        hypothesis = [line['sentence2'] for line in lines]
        output_label = [line['gold_label'] for line in lines]
        return premise, hypothesis, output_label

    def tokenize(self, inputs):
        self.tokenizer = Tokenizer()#num_words=2000)
        self.tokenizer.fit_on_texts(inputs)       # it converts them into words and numbers
        v_size = len(self.tokenizer.word_index) + 1     #vocab_size
        encoded_seq = self.tokenizer.texts_to_sequences(inputs)
        tokenized_input = pad_sequences(encoded_seq, maxlen=300)
        return tokenized_input, v_size 

    def embedding_layer(self, embedding_file, vocab_size):
        embeddings_index = dict()
        f = open(embedding_file)
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
        
    def model_one(self, network_input, embedding_matrix):
        
        em = Embedding(input_dim=vocab_size+1, output_dim=50, weights=[embedding_matrix], input_length=15, trainable=False)
        embed = tf.nn.embedding_lookup(em, network_input)
        #define weights and biases
        weights, biases = self.init_weights_n1(in_dim = 50)
        hidden = tf.add(tf.matmul(embed, weights['w1']), biases['b1'])
        hidden = tf.nn.relu(hidden)
        out_layer = tf.add(tf.matmul(hidden, weights['w2']), biases['b2'])
        return out_layer
        
    def model_two(self, input):
        #some changes regarding input here
        
        #define weights and biases
        weights, biases = self.init_weights_n2(2 * input.shape[1]) 
        hidden = tf.nn.relu(tf.add(tf.matmul(input, weights['w1']), biases['b1']))
        out_layer = tf.add(tf.matmul(hidden, weights['w2']), biases['b2'])
        return out_layer
        
    def init_weights_n1(in_dim, hidden_n = 500, out_n=100):
        
        weights = {
            'w1': tf.Variable(tf.random_normal([in_dim, hidden_n])),
            'w2': tf.Variable(tf.random_normal([hidden_n, out_n]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([hidden_n])),
            'b2': tf.Variable(tf.random_normal([out_n]))
        }
        
    def init_weights_n2(in_dim, hidden_n = 100, out_n=3):
        
        weights = {
            'w1': tf.Variable(tf.random_normal([in_dim, hidden_n])),
            'w2': tf.Variable(tf.random_normal([hidden_n, out_n]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([hidden_n])),
            'b2': tf.Variable(tf.random_normal([out_n]))
        }

		
def modelnn(premise_input, hypothesis_input, embedding_matrix):
	premise_rep = inference().model_1(premise_input, embedding_matrix)
	hypothesis_rep = inference().model_1(hypothesis_input, embedding_matrix)
	net_input = concatenate(premise_rep, hypothesis_rep)
	network_2 = infer.model_2(net_input)
	return pred
	
infer = Inference()
premise, hypothesis, output_label = infer.input_out(infer.read_file(FILE_PATH))

#word to integers
premise_input, p_vocab_size = infer.tokenize(premise)
hypothesis_input, h_vocab_size = infer.tokenize(hypothesis)
#embedding layes using Glove
embedding_matrix = infer.embedding_layer(EMBEDDING_FILE_PATH, p_vocab_size)

# creating model
pred = modelnn(premise_input, hypothesis_input, embedding_matrix)

epochs = 5000
steps = 1000
batch_size = 128
y = output_label.shape[1]

# x = tf.placeholder("float", [None, n_input])
# y = tf.placeholder("float", [None, n_classes])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=(y)))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

##############running###########

with tf.Session(graph=graph) as sess:
    init.run()
    print('Initialized')
	
	for epoch in range(training_epochs):
		average_loss = 0
		total_batch = int(len(premise_input) / batch_size)
        x_batches = np.array_split(premise_input, total_batch)
        y_batches = np.array_split(premise_rep, total_batch)
		for batch in range(total_batch):
			batch_x, batch_y = x_batches[i], y_batches[i]
            _, c = sess.run([optimizer, cost], 
                            feed_dict={
                                x: batch_x, 
                                y: batch_y
                            })
            avg_cost += c / total_batch
		if epoch % steps == 0:
            print("Epoch:{}__cost:{}".format(epoch+1, avg_cost)
			
	print("Optimization Finished!")
	
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			
