import re
import json
import random
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

vectorizer = CountVectorizer()
ps = PorterStemmer()

sc_docs = []
nsc_docs = []
label_sc_docs = []
label_nsc_docs = []
join_docs = []
join_labels = []
train_docs = []
train_labels = []

def classifier_prediction(docs,labels,documents):
	classifier_MNB = MultinomialNB().fit(docs, labels)
	classifier_BNB = BernoulliNB().fit(docs, labels)
	classifier_LR = LogisticRegression().fit(docs,labels)
	classifier_SGDC = SGDClassifier().fit(docs,labels)
	classifier_SVC = SVC().fit(docs,labels)
	classifier_LSVC = LinearSVC().fit(docs,labels)
	classifier_NuSVC = NuSVC().fit(docs,labels)

	egs = documents

	egs_count = vectorizer.transform(egs)


#					Classifier Predictions 		 			  #
###############################################################
	predicted_MNB = classifier_MNB.predict(egs_count)
	predicted_BNB = classifier_BNB.predict(egs_count)
	predicted_LR = classifier_LR.predict(egs_count)
	predicted_SGDC = classifier_SGDC.predict(egs_count)
	predicted_SVC = classifier_SVC.predict(egs_count)
	predicted_LSVC = classifier_LSVC.predict(egs_count)
	predicted_NuSVC = classifier_NuSVC.predict(egs_count)
################################################################

#										Classifications																		#
#############################################################################################################################
	print("Multinomial Naive Bayes :- "+str(predicted_MNB))
	print
	print("Bernoulli Naive Bayes :- "+str(predicted_BNB))
	print
	print("Logistic Regression :- "+str(predicted_LR))
	print
	print("SGDC :- "+str(predicted_SGDC))
	print
	print("SVC :- "+str(predicted_SVC))
	print
	print("Linear SVC :- "+str(predicted_LSVC))
	print
	print("Nu SVC :- "+str(predicted_NuSVC))
#############################################################################################################################


def stemming(features):
	after_stem = []
	final_words = []

	for w in features:
		after_stem.append(ps.stem(w))

	for w in after_stem:
		if w not in final_words:
			final_words.append(w)

	return final_words


def feature_selection(train):
	ch2 = SelectKBest(chi2,'all'); 
	x_train = ch2.fit_transform(train,train_labels)
	v = vectorizer.get_feature_names()
	return v

def tdm_training(doc):
	term_doc = vectorizer.fit_transform(doc)
	return term_doc


def shuffle(docs,labels):
	join = list(zip(docs,labels))
	random.shuffle(join)
	join_docs,join_labels = zip(*join)

def label_docs(document1,document2):
	for i in document1:
		label_sc_docs.append('security')
	for i in document2:
		label_nsc_docs.append("non security")

def clean(data,list_name):
	for key in data:
		content = data[key]['content']
		pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
		text = pattern.sub('',content)
		text = re.sub(r'[^\w\s]','',text)
		text = re.sub('\d+','',text)
		list_name.append(text)

def enlist(document1,document2):
	document1 = document1+".json"
	document2 = document2+".json"

	with open(document1) as data_file:
		data = json.load(data_file)

	send = clean(data,sc_docs)
	
	with open(document2) as data_file:
		data = json.load(data_file)

	send = clean(data,nsc_docs)


doc1 = raw_input("Enter Security file name : ")
doc2 = raw_input("Enter Non-Security file name : ")

enlist(doc1,doc2)

label_docs(sc_docs,nsc_docs)

join_docs = sc_docs+nsc_docs
join_labels = label_sc_docs+label_nsc_docs

shuffle(join_docs,join_labels)

#training_set
train_docs = join_docs[100:]
train_labels = join_labels[100:]

term_docs = tdm_training(train_docs)

features = feature_selection(term_docs)

final_features = stemming(features)

documents = ['happy birthday','bomb blast need to plant bomb','murder in train occured during puja vacations','bjp wins election']

print("Documents :- "+str(documents))
print("Classifications :-")

classifier_prediction(term_docs,train_labels,documents)

##################################################################################################################
#											Results :-										   	   			 	 #

#Documents :- ['happy birthday', 'bomb blast need to plant bomb', 'murder in train occured during puja vacations'#
#				, 'bjp wins election']																			 #

#	Multinomial Naive Bayes :- ['non security' 'security' 'security' 'non security']							 #

#	Bernoulli Naive Bayes :- ['security' 'security' 'security' 'security']										 #

#	Logistic Regression :- ['non security' 'security' 'non security' 'non security']							 #

#	SGDC :- ['non security' 'security' 'non security' 'non security']											 #	

#	SVC :- ['security' 'security' 'security' 'security']														 #

#	Linear SVC :- ['non security' 'non security' 'non security' 'non security']									 #

#	Nu SVC :- ['security' 'security' 'security' 'security']														 #
#################################################################################################################