import sys
import collections
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import nltk
import random
import numpy as np
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    def getDict(data):
	    master_list=[]
	    li=[]
	    for wordList in data:
		for word in wordList:
			if word not in stopwords:
				li.append(word)
		for i in set(li):
			master_list.append(i)
		li=[]

	    master_dict={}
	    for i in master_list:
		if i in master_dict:
			master_dict[i]=master_dict[i]+1
		else:
			master_dict[i]=1
	    return master_dict

    train_pos_dict = getDict(train_pos)
    train_neg_dict = getDict(train_neg)
    #test_pos_dict = getDict(test_pos)
    #test_neg_dict = getDict(test_neg)
    print len(train_pos_dict)
    print len(train_neg_dict)
    def removeLessFrequent(word_dict,val):
	    final_dict={}
	    for k,v in word_dict.items():
		if(v>=len(train_pos)*0.01):#'''+test_pos'''
			final_dict[k]=v
	    print len(final_dict)
	    final_list=[]
	    if val==0:
	    	for k,v in final_dict.items():
			if(v>=(2*(train_neg_dict.get(k,0)))):
				final_list.append(k)
	    elif val==1:
	    	for k,v in final_dict.items():
	    		if(v>=(2*(train_pos_dict.get(k,0)))):
				final_list.append(k)

	    return final_list

    train_pos_list = removeLessFrequent(train_pos_dict,0)
    train_neg_list = removeLessFrequent(train_neg_dict,1)
    #test_pos_list = removeLessFrequent(test_pos_dict,0)
    #test_neg_list = removeLessFrequent(test_neg_dict,1)
    print len(train_pos_list)
    print len(train_neg_list)
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    def buildVector(data, featureList):
	    train_pos_vec=[]
	    row_vec=[]
	    for i in data:
		for j in featureList:
			if j in i:
				row_vec.append(1)
			else:
				row_vec.append(0)
		train_pos_vec.append(row_vec)
		row_vec=[]
	    return train_pos_vec

    train_pos_vec=buildVector(train_pos, train_pos_list+train_neg_list)
    train_neg_vec=buildVector(train_neg, train_pos_list+train_neg_list)
    test_pos_vec=buildVector(test_pos, train_pos_list+train_neg_list)
    test_neg_vec=buildVector(test_neg, train_pos_list+train_neg_list)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    labeled_train_pos=[]
    labeled_train_neg=[]
    labeled_test_pos=[]
    labeled_test_neg=[]
    count=0;
    for i in train_pos:
	labeled_train_pos.append(LabeledSentence(i, ['train_pos_'+str(count)]))
	count=count+1
    count=0;
    for i in train_neg:
	labeled_train_neg.append(LabeledSentence(i, ['train_neg_'+str(count)]))
	count=count+1
    count=0;
    for i in test_pos:
	labeled_test_pos.append(LabeledSentence(i, ['test_pos_'+str(count)]))
	count=count+1
    count=0;
    for i in test_neg:
	labeled_test_neg.append(LabeledSentence(i, ['test_neg_'+str(count)]))
	count=count+1

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    #print model.most_similar('bad')
    #model.save('./twitter.d2v')
    
    #model = Doc2Vec.load('./twitter.d2v')    

    #print model.most_similar('good')
    #print model.docvecs['train_neg_0']
    #print model
    train_pos_vec=[]
    train_neg_vec=[]
    test_pos_vec=[]
    test_neg_vec=[]
    
    #print 'Creating Vector for Train Pos'
    for i in range(len(labeled_train_pos)):
	train_pos_vec.append(model.docvecs['train_pos_'+str(i)])
    #print 'Creating Vector for Train Neg'
    for i in range(len(labeled_train_neg)):
	train_neg_vec.append(model.docvecs['train_neg_'+str(i)])
    #print 'Creating Vector for Test Pos'
    for i in range(len(labeled_test_pos)):
	test_pos_vec.append(model.docvecs['test_pos_'+str(i)])
    #print 'Creating Vector for Test Neg'
    for i in range(len(labeled_test_neg)):
	test_neg_vec.append(model.docvecs['test_neg_'+str(i)])
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(train_pos_vec+train_neg_vec, Y)    
    lr_model = LogisticRegression()
    lr_model.fit(train_pos_vec+train_neg_vec, Y)
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    #print Y
    nb_model = GaussianNB()
    nb_model.fit(train_pos_vec+train_neg_vec, Y)
    lr_model = LogisticRegression()
    lr_model.fit(train_pos_vec+train_neg_vec, Y)	    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    tp=0
    tn=0
    fp=0
    fn=0    
    posR = model.predict(test_pos_vec)
    negR = model.predict(test_neg_vec)
    for i in posR:
	if i=='pos':
	   tp=tp+1
        else:
           fn=fn+1
    for i in negR:
	if i=='neg':
	   tn=tn+1
        else:
           fp=fp+1
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    accuracy = (float)(tp+tn)/(tp+fp+tn+fn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
