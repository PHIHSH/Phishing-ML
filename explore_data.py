import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import tree
from sklearn import neural_network

df = pd.read_csv("Training Dataset.txt")
columns = ['having_IP_Address', 'URL_Length', 'Shortening_Service', 'having_At_Symbol', 'double_slash_redirecting', 
			'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port', 
			'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
			'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic',
			'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report', 'Result']

df.columns = columns

phish = df[df['Result'] == 1]
NotPhish = df[df['Result'] == -1]

print(len(phish))
print(len(NotPhish))

def overSample(originalDataset, minorityClassDataset):
	# Use average values from each column to generate
	# artificial data to solve class imbalance.
	
	# Determining how many rows are needed
	numOfRowsNeeded = len(originalDataset) - len(minorityClassDataset) - len(minorityClassDataset)

	# Creating averages dataframe
	columns = originalDataset.columns
	averagesDict = {}
	for column in columns:
		averagesDict[column] = [minorityClassDataset[column].mean()] * numOfRowsNeeded
	averageDF = pd.DataFrame.from_dict(averagesDict)

	balancedDF = pd.concat([originalDataset, averageDF], ignore_index=True)

	return balancedDF

balancedDF = overSample(df, NotPhish)

train, test = train_test_split(balancedDF, test_size=0.3) # Train and Test data with 70 / 30 split

train_features = train[columns].drop(['Result'], axis=1).values
train_labels = train['Result'].values

test_features = test[columns].drop(['Result'], axis=1).values
test_labels = test['Result'].values


# Decision Tree
clf = tree.DecisionTreeClassifier()
model = clf.fit(train_features, train_labels)
predictions = model.predict(test_features)
tree_report = classification_report(test_labels, predictions)
print("\n\nDecision Tree:")
print(tree_report)

