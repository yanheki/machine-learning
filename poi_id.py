#!/usr/bin/python

import sys
import pickle
import numpy
import matplotlib.pyplot as plt

sys.path.append("../tools/")
sys.path.append("../final_project/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# from given 
financial_features=['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] #(all units are in US dollars)

email_features=['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] #(units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI_label=['poi'] #(boolean, represented as integer)

# total features list
total_features_list = POI_label + email_features + financial_features

# create feature list 
def select_features_by_num(data_dict, features_list, threshold = 0):
    num_list=[]
    for feature in features_list:#calculate_available_num_features_list
        ctr = 0
        for name in data_dict: #calculate_available_num_features
            if data_dict[name][feature] != 'NaN':
                ctr += 1 
        num_list.append(ctr)
    selected_features = []
    
    for i in range(len(num_list)):
        if num_list[i] > threshold:
            selected_features.append(features_list[i])
            
    return selected_features
 
    
available_data = 85
selected_features_list = select_features_by_num(data_dict, total_features_list, threshold = available_data)
features_list = selected_features_list


### Task 2: Remove outliers
# Remove the "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" data point (outliers)
data_dict.pop("TOTAL", 0 ) 
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) 

### Task 3: Create new feature(s)

def delete_feature(delete_feature_name):
        features_list.remove(delete_feature_name)
        for name in data_dict:
            data_dict[name].pop(delete_feature_name)

def calculate_feature(feature_one, feature_two, new_name, function = 'add'):
        if function == 'add':
            for name in data_dict:
                if data_dict[name][feature_one] == 'NaN':
                    data_dict[name][feature_one] = 0
                if data_dict[name][feature_two] == 'NaN':
                    data_dict[name][feature_two] = 0
                    
                data_dict[name][new_name] = data_dict[name][feature_one] + data_dict[name][feature_two]
               
        elif function == 'multiply':
            for name in data_dict:
                if data_dict[name][feature_one] == 'NaN':
                    data_dict[name][feature_one] = 0
                if data_dict[name][feature_two] == 'NaN':
                    data_dict[name][feature_two] = 0
                    
                data_dict[name][new_name] = data_dict[name][feature_one] * data_dict[name][feature_two]



calculate_feature("total_stock_value", "exercised_stock_options", "new_features_1", "add")
calculate_feature("total_stock_value", "salary", "new_features_2", "add")
calculate_feature("shared_receipt_with_poi", "total_payments", "new_features_3", "multiply")
delete_feature("total_stock_value")
delete_feature("email_address")

new_features_list=['new_features_1','new_features_2','new_features_3']
features_list=features_list+new_features_list

### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()


# Preprocessing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Using feature selection to select the feature
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
k = 4
selectKB = SelectKBest(f_classif, k = k)
features = selectKB.fit_transform(features, labels)
index = selectKB.get_support().tolist()

new_features_list = []
for i in range(len(index)):
    if index[i]:
        new_features_list.append(features_list[i+1])
        
# Insert poi to the first element
new_features_list.insert(0, "poi")

# Re-run the featureFormat and targetFeatureSplit to remove all zeros data
data = featureFormat(my_dataset, new_features_list)
labels, features = targetFeatureSplit(data)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

skf = StratifiedKFold( labels, n_folds=3 )
accuracies = []
precisions = []
recalls = []


from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

for train_idx, test_idx in skf: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
    
    ### fit the classifier using training set, and test on test set
    

    parameter = {'base_estimator':[None, DecisionTreeClassifier(min_samples_split=5, max_features = None),
                                   RandomForestClassifier(min_samples_split=3, max_features = None)],
                                   'n_estimators':[20, 50, 110]}
                                   
                                   
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
adaBoost = AdaBoostClassifier(learning_rate = 1, random_state = 0, algorithm='SAMME')
clf = GridSearchCV(adaBoost, parameter)
i=0
while i<5:
    clf.fit(features_train, labels_train)
    i=i+1
pred = clf.predict(features_test)
    
accuracy = clf.score(features_test, labels_test) 
labels_test_1 = labels_test


### for each fold, print some metrics
print k
print "Accuracy: %f " %accuracy
print "precision score: ", precision_score( labels_test, pred )
print "recall score: ", recall_score( labels_test, pred )
    
accuracies.append(accuracy)
precisions.append( precision_score(labels_test, pred) )
recalls.append( recall_score(labels_test, pred) )

### aggregate precision and recall over all folds
print "average accuracy: ", sum(accuracies)/3.
print "average precision: ", sum(precisions)/3.
print "average recall: ", sum(recalls)/3.
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
features_list = new_features_list

dump_classifier_and_data(clf, my_dataset, features_list)
#####################################################################
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(my_dataset, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

#
