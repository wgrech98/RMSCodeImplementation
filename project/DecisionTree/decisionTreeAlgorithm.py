import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz


class D3_Algorithm():

    def __init__(self):
        """
        setting initial variables
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

        self.f_names = ['requirements_volatility',
                        'requirements_clarity', 'dev_time', 'project_size', 'team_size',
                        'prod_complexity', 'testing_intensity', 'risk_analysis', 'user_participation',
                        'team_expertise', 'dev_expertise', 'doc_needed', 'fund_avail', 'delivery_speed', 'task_visualisation', 'prototyping']
        self.c_names = ['Waterfall', 'Scrum', 'Kanban', 'Hybrid: Scrum and Kanban',
                        'Hybrid: Scrum and Waterfall', 'Spiral', 'RAD']

    def main(self):
        self.read_csv()
        self.perform_D3()

    def read_csv(self):
        """ 
        Function to split the CSV file into two dataframes, one for records representing the first methodology that participants
        rated while the second represents the records for those participant who responded to use a second methodology.
        The two dataframes are then merged together and the final dataframe is returned.
        """

        csv_path = '../dataset/survey_dataset.csv'
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

    def perform_D3(self):
        """
        Function which creates a decision tree model and generate predictions based on the model
        with the Kfold cross validation technique.

        Returns the decision tree prediction.
        """

        dataset = self.dataset

        # convert characteristics to vectors
        cl_dataset = self.convert_to_vectors(dataset)

        # Set X and Y variables
        X = cl_dataset.drop('methodology', axis=1)
        y = cl_dataset[['methodology']]
        k = 3

        # for-loop to build a decision tree model with Kfold cross-validation
        kf = KFold(n_splits=k, random_state=None)
        acc_score = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.model = DecisionTreeClassifier(
                criterion="gini", min_samples_leaf=1)
            self.model.fit(X_train, y_train)
            target = list(self.dataset['methodology'].unique())

            # plot tree
            dot_data = tree.export_graphviz(self.model, out_file=None,
                                            feature_names=self.f_names,
                                            class_names=target,
                                            filled=True)
            graph = graphviz.Source(dot_data, format="png")
            graph.render("decision_tree_graph/decision_tree_graphivz")
            'decision_tree_graph/decision_tree_graphivz.png'

            # Getting prediction
            y_predict = self.model.predict(X_test)

            # print accuracy score for each fold
            acc = accuracy_score(y_test, y_predict)
            acc_score.append(acc)
            print(y_predict, y_test)

        avg_acc_score = sum(acc_score)/k

        # printing accuracy of each fold + average accuracy
        print_fold_accuracy = print(
            'accuracy of each fold - {}'.format(acc_score))
        print_avg_accuracy = print('Avg accuracy : {}'.format(avg_acc_score))
        print(print_avg_accuracy, print_fold_accuracy)


if __name__ == '__main__':
    D3_model = D3_Algorithm()
    D3_model.main()
