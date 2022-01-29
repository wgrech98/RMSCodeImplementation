import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
from sklearn.tree import export_text

class D3_RMS():

    def __init__(self):
            """
                constructor: partitions data into train and test sets, sets the minimum accepted significance value
                and maximum star size which limits the number of complexes considered for specialisation.
                """
            self.cols = ['methodology', 'requirements_volatility', 
                    'requirements_clarity', 'dev_time', 'project_size', 'team_size', 
                    'prod_complexity', 'testing_intensity', 'risk_analysis', 'user_participation',
                    'team_expertise', 'dev_expertise', 'doc_needed', 'fund_avail', 'delivery_speed','task_visualisation']
                    
            self.num_cols = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

            self.dataset = None

            self.model = None

            self.f_names = ['requirements_volatility', 
                        'requirements_clarity', 'dev_time', 'project_size', 'team_size', 
                        'prod_complexity', 'testing_intensity', 'risk_analysis', 'user_participation',
                        'team_expertise', 'dev_expertise', 'doc_needed', 'fund_avail', 'delivery_speed', 'task_visualisation']
            self.c_names = ['Waterfall', 'Scrum', 'Kanban', 'Hybrid: Scrum and Kanban', 'Hybrid: Scrum and Waterfall', 'Spiral', 'RAD' ]

    def main(self):
        self.read_csv()
        # self.perform_D3()
        self.perform_D31()

   
    def read_csv(self):
        csv_path = 'SDLC2.csv'
        self.dataset = pd.read_csv(csv_path, names = self.cols, usecols=self.num_cols, header = 0)

        return self.dataset 

    def convert_to_vectors(self,df):

        df['risk_analysis'] = df['risk_analysis'].map(dict(Low=1, Medium=2,High=3))
        df['user_participation'] = df['user_participation'].map(dict(Low=1, Medium=2,High=3))
        df['team_expertise'] = df['team_expertise'].map(dict(Low=1, Medium=2,High=3))
        df['dev_expertise'] = df['dev_expertise'].map(dict(Low=1, Medium=2,High=3))
        df['doc_needed'] = df['doc_needed'].map(dict(Low=1, Medium=2,High=3))
        df['fund_avail'] = df['fund_avail'].map(dict(Low=1, Medium=2,High=3))
        df['delivery_speed'] = df['delivery_speed'].map(dict(Low=1, Medium=2,High=3))
        df['task_visualisation'] = df['task_visualisation'].map(dict(Low=1, Medium=2,High=3))

        # project_type = {'Application (everything else)': 1,'System (sits between the hardware and the application software e.g. OSs)': 2,
        #                 'Utility (performs specific tasks to keep the computer running e.g. antivirus)':3}
        requirements_volatility = {'Changing': 1,'Fixed': 2}
        requirements_clarity = {'unknown/defined later in the lifecycle': 1,'understandable/early defined': 2}
        dev_time = {'Intensive':1, 'Non-Intensive':2}
        project_size = {'Small':1 , 'Medium':2, 'Large':3}
        team_size = {'Small (1-5)':1, 'Medium (6-15)':2, 'Large (16....)':3}
        prod_complexity = {'Simple':1, 'Complex':2}
        testing_intensity = {'After each cycle (Intensive testing)':1, 'After development is done (Non-intensive testing)':2}


        # df.project_type = [project_type[item] for item in df.project_type]
        df.requirements_volatility = [requirements_volatility[item] for item in df.requirements_volatility]
        df.requirements_clarity = [requirements_clarity[item] for item in df.requirements_clarity]
        df.dev_time = [dev_time[item] for item in df.dev_time]
        df.project_size = [project_size[item] for item in df.project_size]
        df.team_size = [team_size[item] for item in df.team_size]
        df.prod_complexity = [prod_complexity[item] for item in df.prod_complexity]
        df.testing_intensity = [testing_intensity[item] for item in df.testing_intensity]

        return df


    def perform_D3(self):
        dataset = self.dataset
        cl_dataset = self.convert_to_vectors(dataset)    
        X = cl_dataset.drop('methodology',axis=1)
        y = cl_dataset[['methodology']]
        k = 5
        kf = KFold(n_splits=k, random_state=None)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=42)
        acc_score = []
        for train_index , test_index in kf.split(X):
            X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train , y_test = y.iloc[train_index] , y.iloc[test_index]
            self.model = DecisionTreeClassifier(criterion="entropy", random_state=42,max_depth=17, min_samples_leaf=1)  

            self.model.fit(X_train,y_train)
            y_predict = self.model.predict(X_test)

            acc = accuracy_score(y_test,y_predict)
            acc_score.append(acc)
        avg_acc_score = sum(acc_score)/k

        print_fold_accuracy = print('accuracy of each fold - {}'.format(acc_score))
        print_avg_accuracy = print('Avg accuracy : {}'.format(avg_acc_score))
        ro = export_text(self.model, feature_names=self.f_names)
        rules = self.get_rules(self.model, self.f_names, self.c_names)
        print(y_predict)

    def perform_D31(self):

        X_test = ['Changing', 'unknown/defined later in the lifecycle', 
            'Intensive', 'Large', 'Small (1-5)', 'Complex', 'After each cycle (Intensive testing)', 'High', 'Low','High', 'High', 
            'Low','Low','Medium', 'Low']

        y_test = ['Spiral']
        
        X_test = pd.DataFrame(X_test, columns=self.f_names)

        X_test = self.convert_to_vectors(X_test)

        y_predict = self.model.predict(X_test)

        acc = accuracy_score(y_test,y_predict)

        print(y_predict, acc) 


    def get_rules(self, tree, feature_names, class_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []

        def recurse(node, path, paths):
            
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    p1, p2 = list(path), list(path)
                    p1 += [f"({name} <= {np.round(threshold, 3)})"]
                    recurse(tree_.children_left[node], p1, paths)
                    p2 += [f"({name} > {np.round(threshold, 3)})"]
                    recurse(tree_.children_right[node], p2, paths)
                else:
                    path += [(tree_.value[node], tree_.n_node_samples[node])]
                    paths += [path]
                    
                recurse(0, path, paths)

                # sort by samples count
                samples_count = [p[-1][1] for p in paths]
                ii = list(np.argsort(samples_count))
                paths = [paths[i] for i in reversed(ii)]
                
                rules = []
                for path in paths:
                    rule = "if "
                    
                    for p in path[:-1]:
                        if rule != "if ":
                            rule += " and "
                        rule += str(p)
                    rule += " then "
                    if class_names is None:
                        rule += "response: "+str(np.round(path[-1][0][0][0],3))
                    else:
                        classes = path[-1][0][0]
                        l = np.argmax(classes)
                        rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
                    rule += f" | based on {path[-1][1]:,} samples"
                    rules += [rule]
                    
                return rules



if __name__ == '__main__':
    D3_model =  D3_RMS()
    D3_model.main()
