import pandas as pd

class Test():

    def __init__(self):
        self.remaining_examples = None
        self.cols = ['methodology', 'project_type', 'requirements_volatility', 
                'requirements_clarity', 'dev_time', 'project_size', 'team_size', 
                'prod_complexity', 'testing_intensity', 'risk_analysis', 'user_participation',
                'team_expertise', 'dev_expertise', 'doc_needed', 'fund_avail', 'delivery_speed']
        self.num_cols = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        self.rule_list = pd.read_csv('rule_list.csv')
        self.df = pd.read_csv('SDLC2.csv', names = self.cols, usecols=self.num_cols, header = 0)

    def main(self):
        self.test_model(self.rule_list, self.testo())
        
        
    def test_model(self, rule_list, data_set = 'default'):
        """
        Test rule list returned by fit_CN2_Model function on test data(or manually supplied data)
        returns a dataframe that contains the rule, rule acc, num of examples covered.
        Also return general accuracy as average of each rule accuracy
        """

        self.remaining_examples = data_set
        list_of_row_dicts = []


        for rule in rule_list:
            rule_coverage_indexes,rule_coverage_dataframe = self.complex_coverage(rule[0], self.remaining_examples)
            #check for zero coverage due to noise(lense data too small)
            if len(rule_coverage_dataframe) == 0:
                row_dictionary = {'rule':rule,'pred_class':'zero coverage','rule_acc':0,
                            'num_examples':0,'num_correct':0,
                            'num_wrong':0}
                list_of_row_dicts.append(row_dictionary)			   
            #otherwise generate statistics about rule then save and remove examples from the data and test next rule.
            else:				   
                class_of_covered_examples = rule_coverage_dataframe['methodology']
                #import ipdb;ipdb.set_trace(context=8)
                class_counts = class_of_covered_examples.value_counts()
                rule_accuracy = class_counts.values[0]/sum(class_counts)
                num_correctly_classified_examples = class_counts.values[0]
                num_incorrectly_classified_examples = sum(class_counts.values) - num_correctly_classified_examples

                row_dictionary = {'rule':rule,'pred_class':rule[1],'rule_acc':rule_accuracy,
                                'num_examples':len(rule_coverage_indexes),'num_correct':num_correctly_classified_examples,
                                'num_wrong':num_incorrectly_classified_examples}
                list_of_row_dicts.append(row_dictionary) 

                remaining_examples = remaining_examples.drop(rule_coverage_indexes)

        results = pd.DataFrame(list_of_row_dicts)
        overall_accuracy = sum(results['rule_acc'])/len([r for r in results['rule_acc'] if r !=0])
        return results, overall_accuracy


    def testo(self):
        testSet = [['Waterfall', 'Fixed', 'understandable/early defined', 
                    'Non-Intensive', 'Small', 'Small (1-5)', 'Simple', 'After development is done (Non-intensive testing)', 'Low', 'High','High', 'High', 
                    'Medium','Low','Low', 'Low'],

                        ['RAD', 'Fixed', 'understandable/early defined', 
                                    'Intensive', 'Large', 'Large (16....)', 'Complex', 'After each cycle (Intensive testing)', 'High', 'High','High', 'High', 
                                    'Medium','Low','Low', 'Low'],

                        ['Scrum',  'Changing', 'unknown/defined later in the lifecycle', 
                                    'Intensive', 'Large', 'Medium (6-15)', 'Simple', 'After each cycle (Intensive testing)', 'Low', 'High','High', 'High', 
                                    'Low','High','Medium', 'Low'],

                        ['Kanban', 'Changing', 'understandable/early defined', 
                                    'Non-Intensive', 'Medium', 'Large (16....)', 'Complex', 'After each cycle (Intensive testing)', 'Medium', 'Medium','High', 'High', 
                                    'Medium','High','Medium', 'High'],

                        ['Hybrid: Scrum and Waterfall', 'Changing', 'understandable/early defined', 
                                    'Intensive', 'Large', 'Large (16....)', 'Complex', 'After each cycle (Intensive testing)', 'High', 'High','High', 'High', 
                                    'High','High','High', 'Low'],

                        ['Spiral',  'Changing', 'unknown/defined later in the lifecycle', 
                                    'Intensive', 'Large', 'Small (1-5)', 'Complex', 'After each cycle (Intensive testing)', 'High', 'Low','High', 'High', 
                                    'Low','Low','Medium', 'Low'],

                        ['Kanban', 'Changing', 'understandable/early defined', 
                                    'Non-Intensive', 'Medium', 'Medium (6-15)', 'Simple', 'After each cycle (Intensive testing)', 'Medium', 'Medium','High', 'High', 
                                    'Medium','High','Medium', 'High']]

        testo = pd.DataFrame(testSet, columns=self.cols)
        return testo

    def complex_coverage(self, passed_complex, data_set = 'default'):
        """ Returns set of instances of the data 
            which complex(rule) covers as a dataframe.
        """
        # if type(data_set) == str: 
        #     data_set = self.train
        # coverage = []

        rule = self.build_rule(passed_complex)
        if rule == False:
            return [],[]

        mask = data_set.isin(rule).all(axis=1)
        rule_coverage_indexes = data_set[mask].index.values
        rule_coverage_dataframe = data_set[mask]
    
        # #iterate over dataframe rows
        # for index,row in data_set.iterrows():
        # 	if self.check_rule_datapoint(row, complex):
        # 		coverage.append(index)
        

        #import ipdb;ipdb.set_trace(context=8)
        return rule_coverage_indexes, rule_coverage_dataframe

    def build_rule(self,passed_complex):
        """
        build a rule in dict format where target attributes have a single value and non-target attributes
        have a list of all possible values. Checks if there are repetitions in the attributes used, if so
        it returns False
        """
        atts_used_in_rule = []
        for selector in passed_complex:
            atts_used_in_rule.append(selector[0])
        set_of_atts_used_in_rule = set(atts_used_in_rule)
        
        if len(set_of_atts_used_in_rule) < len(atts_used_in_rule):
            return False


        rule = {}
        attributes = self.df.columns.values.tolist() 	
        for att in attributes:
            rule[att] = list(set(self.df[att]))

        for att_val_pair in passed_complex:
            att = att_val_pair[0]
            val = att_val_pair[1]
            rule[att] = [val]
        return rule	


if __name__ == '__main__':
    test_data_class = Test()
    test_data_class.main()
