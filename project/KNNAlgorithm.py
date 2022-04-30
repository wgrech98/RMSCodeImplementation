import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class KNN_Algorithm():

    def __init__(self):
        """
        Constructor: set initial variables
        """

        self.cols = ['methodology', 'requirements_volatility',
                     'requirements_clarity', 'dev_time', 'project_size', 'team_size',
                     'prod_complexity', 'testing_intensity', 'risk_analysis', 'user_participation',
                     'team_expertise', 'dev_expertise', 'doc_needed', 'fund_avail', 'delivery_speed', 'task_visualisation', 'prototyping']

        self.num_cols = [6, 8, 9, 10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        self.num_cols1 = [24, 26, 27, 28, 29, 30, 31,
                          32, 33, 34, 35, 36, 37, 38, 39, 40, 41]

        self.dataset = None

    def main(self):
        self.read_csv()
        self.perform_KNN()

    def read_csv(self):
        """ 
        Function to split the CSV file into two dataframes, one for records representing the first methodology that participants
        rated while the second represents the records for those participant who responded to use a second methodology.
        The two dataframes are then merged together and the final dataframe is returned.
        """

        csv_path = 'dataset/survey_dataset.csv'
        df = pd.read_csv(csv_path, names=self.cols,
                         usecols=self.num_cols, header=0)
        df1 = pd.read_csv(csv_path, names=self.cols,
                          usecols=self.num_cols1, header=0)

        # Deleting rows with null values
        df = df.dropna()
        df1 = df1.dropna()

        # Merging the two datasets
        self.dataset = df.append(df1, ignore_index=True)

        return self.dataset

    def convert_to_vectors(self, df):
        """ 
        Convert characteristics to vectors to enable machine learning processing.

        Returns discrete dataset.
        """

        df['risk_analysis'] = df['risk_analysis'].map(
            dict(Low=1, Medium=2, High=3))
        df['user_participation'] = df['user_participation'].map(
            dict(Low=1, Medium=2, High=3))
        df['team_expertise'] = df['team_expertise'].map(
            dict(Low=1, Medium=2, High=3))
        df['dev_expertise'] = df['dev_expertise'].map(
            dict(Low=1, Medium=2, High=3))
        df['doc_needed'] = df['doc_needed'].map(dict(Low=1, Medium=2, High=3))
        df['fund_avail'] = df['fund_avail'].map(dict(Low=1, Medium=2, High=3))
        df['delivery_speed'] = df['delivery_speed'].map(
            dict(Low=1, Medium=2, High=3))
        df['task_visualisation'] = df['task_visualisation'].map(
            dict(Low=1, Medium=2, High=3))
        df['prototyping'] = df['prototyping'].map(
            dict(Low=1, Medium=2, High=3))

        requirements_volatility = {'Changing': 1, 'Fixed': 2}
        requirements_clarity = {
            'unknown/defined later in the lifecycle': 1, 'understandable/early defined': 2}
        dev_time = {'Intensive': 1, 'Non-Intensive': 2}
        project_size = {'Small': 1, 'Medium': 2, 'Large': 3}
        team_size = {'Small (1-5)': 1, 'Medium (6-15)': 2, 'Large (16....)': 3}
        prod_complexity = {'Simple': 1, 'Complex': 2}
        testing_intensity = {
            'After each cycle (Intensive testing)': 1, 'After development is done (Non-intensive testing)': 2}

        # df.project_type = [project_type[item] for item in df.project_type]
        df.requirements_volatility = [
            requirements_volatility[item] for item in df.requirements_volatility]
        df.requirements_clarity = [requirements_clarity[item]
                                   for item in df.requirements_clarity]
        df.dev_time = [dev_time[item] for item in df.dev_time]
        df.project_size = [project_size[item] for item in df.project_size]
        df.team_size = [team_size[item] for item in df.team_size]
        df.prod_complexity = [prod_complexity[item]
                              for item in df.prod_complexity]
        df.testing_intensity = [testing_intensity[item]
                                for item in df.testing_intensity]

        return df

    def perform_KNN(self):
        """
        Function which performs the K-nearest neighbour with the cross validation technique.

        Returns the KNN predictions, the y_test values (the actual class of the test records), and the accuracy score
        of each fold. The average accuracy achieved by all the fold
        """
        # Converts string data to discrete variables
        conv = self.convert_to_vectors(self.dataset)

        # Set X and Y
        X = conv.drop('methodology', axis=1)
        y = conv[['methodology']]

        k = 3
        kf = KFold(n_splits=k, random_state=None)
        acc_score = []

        # for-loop to perform KNN with Kfold cross-validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            neigh = KNeighborsClassifier(n_neighbors=3)

            neigh.fit(X_train, y_train)
            y_predict = neigh.predict(X_test)

            # Accuracy score for each fold
            acc = accuracy_score(y_test, y_predict)
            acc_score.append(acc)
            print(y_predict, y_test)

        # Average accuracy score from all folds
        avg_acc_score = sum(acc_score)/k

        print_fold_accuracy = print(
            'accuracy of each fold - {}'.format(acc_score))
        print_avg_accuracy = print('Avg accuracy : {}'.format(avg_acc_score))
        print(y_predict)


if __name__ == '__main__':
    knn_class = KNN_Algorithm()
    knn_class.main()
