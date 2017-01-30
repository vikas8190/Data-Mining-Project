import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2,f_regression
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import pydot
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# Purpose: Classifier class to use the data to train and create a prediction model.
class DiabeticPredictor():

    #Purpose: Intialize the class varaibles and read data from CSV

    def __init__(self,file_name):
        print("init")
        self.df = pd.read_csv(file_name)
        # To consider only first encounter of a patient : Not sure if rest encounters to be ignored
        #self.df = self.df.groupby('patient_nbr').first().reset_index()
        self.df_bucket = None
        self.df_oversampled=None
        self.nan_columns = []

    # Purpose: PreProcess phase
    # Convert label to required form
    # Drop irrelevant columns based on human expertize of the data set
    # Impute missing values
    # Feature Selection
    def preprocess(self):
        self.preprocess_range_col('age')
        #self.preprocess_range_col('weight')
        self.df.replace(to_replace={'readmitted':{'<30':'Yes','>30':'No'}},inplace=True)
        self.df['readmitted'] = self.df['readmitted'].str.lower()
        self.preprocess_others()
        self.drop_irrelevant_cols()
        self.normalize()
        self.impute_most_frequent()
        self.select_k_best()

    #Purpose: Uses the Mode to replace all the missing values
    def impute_most_frequent(self):
        imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        imputer.fit(self.df)
        self.df = pd.DataFrame.from_records(imputer.transform(self.df),columns=self.df.columns)

	#Purpose: Uses the Median to replace all the missing values
    def impute_missing(self):
        imputer = Imputer(strategy = "median")
        df_y = self.df['readmitted']
        df_imputed = imputer.fit_transform(self.df[self.df.columns[:-1]],df_y)
        self.df = pd.DataFrame.from_records(df_imputed,columns=self.df.columns[:-1])
        self.df['readmitted'] = df_y

    #Purpose: Uses KNN algorithm to replace all the missing values
#    def impute_knn(self):
#		knnImpute = KNN(k=5)
#		list = []	
#		for chunk in np.array_split(self.df, 5):
## 			print chunk
#			list.append(pd.DataFrame(knnImpute.complete(chunk)))		
#		names = self.df.columns.values		
#		self.df = pd.concat(list)
#		self.df.columns =  names
#		self.df.to_csv("NonImputedFile.csv")

    #Purpose: These features have more than 50% missing values and hence it was
    #decided to remove then.
    def drop_irrelevant_cols(self):
        for col in ['encounter_id','patient_nbr','weight','payer_code']:
            self.df.drop(col, axis=1, inplace=True)

    #Purpose: Feature selection is done using F_regression
    def select_k_best(self):
        df_dataset_y = self.df['readmitted']
        df_dataset = self.df[self.df.columns[:-1]]
        sel_k = int(round(self.df.shape[1]*0.60))
        f_reg = SelectKBest(f_regression, k=sel_k)
        df_dataset = f_reg.fit_transform(df_dataset,df_dataset_y)
        selected_cols = np.asarray(self.df.columns[:-1])[f_reg.get_support()]
        selected_cols = np.append(selected_cols,'readmitted')
        self.df = self.df[selected_cols]

    #Purpose: Normalize the data using Z-Score method.
    def normalize(self):
        for col in self.df.columns[:-1]:
            if self.df[col].std(ddof=1) == 0:
                self.df[col] = (self.df[col] - self.df[col].mean())
            else:
                self.df[col] = (self.df[col] - self.df[col].mean())/self.df[col].std(ddof=1)

    #Purpose: Convert the string representation values into categorical buckets
    def replace_diag(self,value):
        try:
            value = float(value)
            if (value >= 390 and value <= 459) or value == 785:
                return 0
            if (value >= 460 and value <= 520) or value == 786:
                return 1
            if (value >= 520 and value <= 579) or value == 787:
                return 2
            if (value >= 250 and value <= 250.99):
                return 3
            if (value >= 800 and value <= 999):
                return 4
            if (value >= 710 and value <= 739):
                return 5
            if (value >= 580 and value <= 629) or value == 788:
                return 6
            if (value >= 140 and value <= 239) or (value >= 790 and value <= 799) or (value >= 240 and value <= 249) or \
                    (value >= 251 and value <= 279) or (value >= 680 and value <= 709) or (value >= 1 and value <= 139) or \
                    (value in [780,781,784,782]):
                return 7
            if (value=="?"):
                return np.nan
            else:
                return 8
        except ValueError:
            return 8
 
    # Purpose: Make sure every column is formatted to a numerical meaningful value.
    # ? in label are re labled
    # Cateorical string values are num encoded.
    def preprocess_others(self):
        le = LabelEncoder()
        classes = {}
        string_cols = ['race','gender','weight','medical_specialty','max_glu_serum','A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride', 'payer_code','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','change','diabetesMed']
        for col in self.df.columns.values:
            if col == 'diag_1' or col == 'diag_2' or col == 'diag_3':
                self.df[col] = self.df[col].apply(self.replace_diag)
            elif col in string_cols:
                le.fit(self.df[col].values)
                classes[col] = list(le.classes_)
                self.df[col] = le.transform(self.df[col])
                if "?" in classes[col]:
                    encoded_to=le.transform(["?"])
                    self.df[col].replace(encoded_to, np.nan ,inplace=True)
            elif col == 'readmitted':
                self.df[col].replace(['yes', 'no'], [1, 0], inplace=True)

    #Purpose: Convert range columns to mean
    def preprocess_range_col(self,col):
        self.df[col] = self.df[col].str.extract('\[(\d+)\-(\d+)\)',expand=True).astype(int).mean(axis=1)

    #Purpose: Convert ? in to Nan which can be processed later.
    def set_nan_cols(self):
        for col in self.df.columns:
            temp = self.df[col].unique()
            if '?' in temp:
                self.nan_columns.append(col)

        for col in self.nan_columns:
            print("Percentage of ? in col {} = {}".format(col,(self.df.groupby(by=[col]).size()*100/len(self.df))['?']))

    #Purpose: Create
    def create_df_bucket(self):
        df_grp = self.df.groupby(by=['readmitted'])
        df_grp_yes = df_grp.get_group(1)
        df_grp_no = df_grp.get_group(0)
        df_grp_no = shuffle(df_grp_no)
        self.df_bucket = np.array_split(df_grp_no,9)
        for ind in range(0,len(self.df_bucket)):
            self.df_bucket[ind] = pd.concat([self.df_bucket[ind],df_grp_yes])

    #Purpose: Read data from disk
    def spool_df(self):
        self.df.to_csv("converted_data.csv",index=False)

    #Purpose: Oversampling implemented to balance the data.
    def oversampling(self):
        readmitted_yes = self.df['readmitted'] == 1
        df_yes = self.df[readmitted_yes]
        self.df_oversampled = self.df.append([df_yes] * 8, ignore_index=True)


    #Purpose: Implements the decision tree classifier
    def run_decision_tree(self):
        Y = self.df_oversampled["readmitted"]
        X = self.df_oversampled[self.df_oversampled.columns[:-1]]
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, Y)
        scores = cross_val_score(dt, X, Y, cv=10)
        print("Decision Tree Score for bucket is: {}".format(scores))
        #self.visualize_tree(dt, self.df.columns[:-1])

    #Purpose: Implements the SVM classifier
    def run_svm(self):
        Y = self.df_oversampled["readmitted"]
        X = self.df_oversampled[self.df_oversampled.columns[:-1]]
        dt = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr',penalty='l1', random_state=None, tol=0.001, verbose=0)
        dt.fit(X, Y)
        scores = cross_val_score(dt, X, Y, cv=10)
        print("SVM Score for bucket is: {}".format(scores))
        #self.visualize_tree(dt, self.df.columns[:-1])

    #Purpose: Implement voting classifier
    def run_voting_classifier(self):
        clf1 = DecisionTreeClassifier(random_state=0)
        clf2 = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr',penalty='l2', random_state=None, tol=0.001, verbose=0)
        #clf2 = SVC(gamma=0.001, C=100.)
        clf3 = RandomForestClassifier(n_estimators = 50)
        eclf = VotingClassifier(estimators=[('dr', clf1), ('lsvm', clf2), ('rf', clf3)], voting='hard')

        from sklearn.model_selection import train_test_split

        train, test = train_test_split(self.df_oversampled, test_size = 0.3)
        yTrain = train["readmitted"]
        xTrain = train[train.columns[:-1]]

        yTest = test["readmitted"]
        xTest = test[train.columns[:-1]]

        # eclf.fit(xTrain,yTrain)
        for clf, label in zip([clf1, clf2, clf3, eclf], ['Decision Tree', 'Linear SVM', 'Random Forest', 'Ensemble']):
            clf.fit(xTrain,yTrain)
            trainScores = cross_val_score(clf, xTrain, yTrain, cv=5, scoring='accuracy')
            y_trainPred = clf.predict(xTrain)
            trainAccuracy = accuracy_score(yTrain, y_trainPred)
            print("Train Accuracy: %0.2f [%s]" % (trainAccuracy, label))
            print("Train Accuracy (5 Fold CV): %0.2f (+/- %0.2f) [%s]" % (trainScores.mean(), trainScores.std(), label))
            y_pred = clf.predict(xTest)
            testScores = accuracy_score(yTest, y_pred)
            # testScores = cross_val_score(clf, xTest, yTest, cv=5, scoring='accuracy')
            print("Test Accuracy: %0.2f [%s]" % (testScores, label))


    #Purpose: Print the tree generated from the decision tree classifier.
    def visualize_tree(self, tree, feature_names):
        with open("dt.dot", 'w') as f:
            export_graphviz(tree, out_file=f,
                            feature_names=feature_names)
        (graph,) = pydot.graph_from_dot_file("dt.dot")
        graph.write_png("abc.png")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-f","--file_name",type=str,help="Data file name")
    args=parser.parse_args()
    file_name = args.file_name
    predictor = DiabeticPredictor(file_name)
    predictor.set_nan_cols()
    predictor.preprocess()
    predictor.oversampling()
    predictor.spool_df()
    #predictor.run_svm()
    predictor.create_df_bucket()
    #predictor.run_decision_tree()
    predictor.run_voting_classifier()
if __name__ == "__main__":
    main()
