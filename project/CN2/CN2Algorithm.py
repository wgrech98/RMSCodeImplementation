import pandas as pd
import numpy as np
import copy
import collections as col
import pickle
from sklearn.model_selection import KFold


class CN2algorithm():

    def __init__(self):
        """
        constructor: Provides heading names for the dataset,  specifies 
        the columns needed for the dataset, sets name for the rules pkl 
        file, sets the minimum accepted significance_of_rules value and
        maximum star size which limits the number of complexes considered
        for specialisation.
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

        self.test = None

        self.train = None

        self.rule_list = None

        self.filename = "rules/rules.pkl"

        self.min_significance_of_rules = 0.5

        self.max_size_of_star = 5

    def main(self):
        self.read_csv()
        self.perform_CN2()

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
        df = df.dropna()
        df1 = df1.dropna()
        self.dataset = df.append(df1, ignore_index=True)

    def perform_CN2(self):
        """ 
        Function to perform CN2 algorithm with the k-fold cross validation technique. 
        Returns a rule list and the results (predictions) for each fold fold.
        """
        k = 3
        kf = KFold(n_splits=k, random_state=None)
        df = self.dataset
        numbers = range(0, 2)

        for train_index, test_index in kf.split(df):
            self.train, self.test = df.iloc[train_index,
                                            :], df.iloc[test_index, :]
            self.fit_CN2(self.train)
            predict = self.test_fitted_model(self.test)
            result = pd.DataFrame(predict)
            # rule = pd.DataFrame(rule_list)
            # rule.to_csv(r"rules_{}.csv".format())
            result.to_csv(r"results/results_{}.csv".format(test_index))

        print(predict)

    def fit_CN2(self, train):
        """
        Function to fit the CN2 model and produce a rule-set.

        Returns a rule-list. 
        """

        selectors = self.attribute_value_pairs()
        remaining_records = train
        rule_list = []
        # iterate until end of dataset.
        while len(remaining_records) >= 1:
            best_new_calculate_significance_of_rules = 1
            rules_to_specialise = []
            existing_rules = pd.DataFrame()
            # search rule space until rule best_new_rule_significance_of_rules = 1significance_of_rules is lower than user set boundary(0.5 for testing)
            while best_new_calculate_significance_of_rules > self.min_significance_of_rules:
                # calls statement if its first iteration of loop
                if len(rules_to_specialise) == 0:
                    ordered_rule_results = self.apply_statistics_and_order(
                        selectors, remaining_records)
                    trimmed_rules = ordered_rule_results[0:self.max_size_of_star]
                elif len(rules_to_specialise) != 0:
                    specialised_rules = self.specialisation_process(
                        rules_to_specialise, selectors)
                    ordered_rule_results = self.apply_statistics_and_order(
                        specialised_rules, remaining_records)
                    trimmed_rules = ordered_rule_results[0:self.max_size_of_star]
                # append newly discovered rules to existing ones, order them and then take best X(3 for testing)
                existing_rules = existing_rules.append(
                    trimmed_rules)
                existing_rules = self.order_rule_list(
                    existing_rules).iloc[0:2]
                # update 'rules to specialise' and significance_of_rules value of best new rule
                rules_to_specialise = trimmed_rules['rule']
                best_new_calculate_significance_of_rules = trimmed_rules[
                    'significance_of_rules'].values[0]

            # Get best rule for instance
            best_rule = (existing_rules['rule'].iloc[0], existing_rules['predicted_class'].iloc[0],
                         existing_rules['num_records_covered'].iloc[0])
            best_rule_coverage_index, best_rule_coverage_df = self.complex_coverage(
                best_rule[0], remaining_records)
            rule_list.append(best_rule)

            # Save rules into PKL file
            open_file = open(self.filename, "wb")
            pickle.dump(rule_list, open_file)
            open_file.close()

            self.rule_list = rule_list

            remaining_records = remaining_records.drop(
                best_rule_coverage_index)

        return self.rule_list

    def test_fitted_model(self, data):
        """
        Test rule list returned by fit_CN2 function on test data(or manually supplied data).
        Returns a dictionary that contains the original class of the instance, the rule, the
        predicted class, the num of examples covered by the rule, the number of correct predictions,
        and the rule acc.
        """

        remaining_records = data
        list_of_rule_dicts = []

        for rule in self.rule_list:
            indexes_of_rules_covered, rule_coverage_dataset = self.complex_coverage(
                rule[0], remaining_records)
            # If a rule isn't trigger with the testing records
            if len(rule_coverage_dataset) == 0:
                rule_dictionary = {'original_class': class_of_covered_instances, 'rule': rule, 'predicted_class': 'zero coverage', 'rule_acc': 0,
                                   'num_examples': 0, 'num_correct': 0,
                                   'num_wrong': 0}
                list_of_rule_dicts.append(rule_dictionary)
            # otherwise generate statistics about rule then save and remove examples from the data and test next rule.
            else:
                class_of_covered_instances = rule_coverage_dataset['methodology']
                class_tally = class_of_covered_instances.value_counts()
                rule_accuracy = class_tally.values[0]/sum(class_tally)
                correctly_classified_examples_tally = class_tally.values[0]
                misclassified_examples_count = sum(
                    class_tally.values) - correctly_classified_examples_tally

                rule_dictionary = {'original_class': class_of_covered_instances, 'rule': rule, 'predicted_class': rule[1], 'rule_acc': rule_accuracy,
                                   'num_examples': len(indexes_of_rules_covered), 'num_correct': correctly_classified_examples_tally,
                                   'num_wrong': misclassified_examples_count}
                list_of_rule_dicts.append(rule_dictionary)

                remaining_records = remaining_records.drop(
                    indexes_of_rules_covered)

        return list_of_rule_dicts

    def apply_statistics_and_order(self, list_of_complexes, data):
        """
        A function which takes a list of complexes/rules and returns a dataframe
        with the complex, the entropy, the significance_of_rules, the number of selectors,
        the number of examples covered, the length of the rule and the predicted class of the rule. 
        The input parameter complexes should be a list of lists of tuples.
        """
        # build a dictionary for each rule with statistics
        list_of_rule_dicts = []
        for row in list_of_complexes:
            rule_coverage = self.complex_coverage(row, data)[1]
            length_of_rule = len(row)
            # test if rule covers 0 examples
            if len(rule_coverage) == 0:

                rule_dictionary = {'rule': row, 'predict_class': 'dud rule',
                                   'entropy': 10, 'laplace_accuracy': 0,
                                   'significance_of_rules': 0, 'length': length_of_rule,
                                   'num_records_covered': 0, 'specificity': 0}
                list_of_rule_dicts.append(rule_dictionary)

            # calculate statistics for rules with coverage
            else:

                num_examples_covered = len(rule_coverage)
                rule_entropy = self.calculate_entropy(rule_coverage)
                rule_significance = self.calculate_significance_of_rules(
                    rule_coverage)
                laplace_accuracy_of_rule = self.calculate_laplace_accuracy(
                    rule_coverage)
                class_attrib = rule_coverage['methodology']
                class_tally = class_attrib.value_counts()
                majority_class = class_tally.axes[0][0]
                rule_specificity = class_tally.values[0]/sum(class_tally)
                rule_dictionary = {'rule': row, 'predicted_class': majority_class,
                                   'entropy': rule_entropy, 'laplace_accuracy': laplace_accuracy_of_rule,
                                   'significance_of_rules': rule_significance, 'length': length_of_rule,
                                   'num_records_covered': num_examples_covered, 'specificity': rule_specificity}
                list_of_rule_dicts.append(rule_dictionary)

        # put dictionaries into dataframe and order them according to laplace accuracy, length
        rules_and_stats = pd.DataFrame(list_of_rule_dicts)
        ordered_rules_and_stats = self.order_rule_list(rules_and_stats)

        return ordered_rules_and_stats

    def order_rule_list(self, rules_dataframe):
        """
        Function to order a dataframe of rules and stats according to laplace acc and length.

        The dataframe is then reindexed.
        """
        # sort the dataframe in ascending order according to laplace acc and length
        ordered_rules_and_stats = rules_dataframe.sort_values(['entropy', 'length',
                                                               'num_records_covered'], ascending=[True, True, False])

        # Reindex
        ordered_rules_and_stats = ordered_rules_and_stats.reset_index(
            drop=True)

        return ordered_rules_and_stats

    def attribute_value_pairs(self):
        """
        Function to return the initial set 
        of complexes
        """

        # get attribute names
        attributes = self.train.columns.values.tolist()

        # remove class from features list
        del attributes[0]

        # get possible values for attributes
        possAttribVals = {}
        for attribute in attributes:
            possAttribVals[attribute] = set(self.train[attribute])

        # get list of attribute,value pairs
        # from possAttribVals dictionary
        attribute_value_pairs = []
        for key in possAttribVals.keys():
            for possVal in possAttribVals[key]:
                attribute_value_pairs.append([(key, possVal)])

        return attribute_value_pairs

    def specialisation_process(self, target_complexes, selectors):
        """
        Function which expects a complex (a list of tuples) as input and 
        performs the CN2 specialisation process to specialise the complexes 
        in the "star". In the process, it adds ddtional conjunctions using 
        all the possible selectors. 

        Returns a list of new, specialised complexes.
        """

        provisional_specialised_rules = []
        for targ_complex in target_complexes:
            for selector in selectors:
                # check to see if target complex is a single tuple otherwise assume list of tuples
                if type(targ_complex) == tuple:
                    comp_to_specialise = [copy.copy(targ_complex)]
                else:
                    comp_to_specialise = copy.copy(targ_complex)

                comp_to_specialise.append(selector[0])

                # count if any selector is duplicated, append rule if not
                number_of_selectors_in_complex = col.Counter(
                    comp_to_specialise)
                flag = True
                for count in number_of_selectors_in_complex.values():
                    if count > 1:
                        flag = False

                if flag == True:
                    provisional_specialised_rules.append(comp_to_specialise)

        return provisional_specialised_rules

    def create_rule(self, passed_complex):
        """
        Function which builds a rule in dict format where target attributes have a single value and non-target attributes
        have a list of all possible values. Checks if there are repetitions in the attributes used, if so,
        it returns False
        """
        rule_atts = []
        for selector in passed_complex:
            rule_atts.append(selector[0])
        set_of_rule_atts = set(rule_atts)

        if len(set_of_rule_atts) < len(rule_atts):
            return False

        rule = {}
        attributes = self.train.columns.values.tolist()
        for attribute in attributes:
            rule[attribute] = list(set(self.train[attribute]))

        for att_val_pair in passed_complex:
            attribute = att_val_pair[0]
            value = att_val_pair[1]
            rule[attribute] = [value]
        return rule

    def complex_coverage(self, passed_complex, data):
        """ 
        Returns records which complex(rule) 
        covers as a dataframe.
        """

        rule = self.create_rule(passed_complex)
        if rule == False:
            return [], []

        covered_instances = data.isin(rule).all(axis=1)
        indexes_of_rules_covered = data[covered_instances].index.values
        rule_coverage_dataset = data[covered_instances]

        return indexes_of_rules_covered, rule_coverage_dataset

    def check_rule_data_instance(self, data_instance, complex):
        """
        Function to check if a given record satisfies
        the conditions of a given complex. The record
        should be a pandas series. Complex should be a
        tuple or a list of tuples where each tuple is of
        the form ('Attribute', 'Value').
        """
        if type(complex) == tuple:
            if data_instance[complex[0]] == complex[1]:
                return True
            else:
                return False

        if type(complex) == list:
            result = True
            for selector in complex:
                if data_instance[selector[0]] != selector[1]:
                    result = False

            return result

    def calculate_entropy(self, covered_data):
        """
        Function which takes the covered records by the rule.

        Returns the Shannon entropy of the rule according to the 
        records it covered. 
        """
        class_series = covered_data['methodology']
        num_of_records = len(class_series)
        class_tally = class_series.value_counts()
        class_probabilities = class_tally.divide(num_of_records)
        log2_of_classprobs = np.log2(class_probabilities)
        plog2p = class_probabilities.multiply(log2_of_classprobs)
        entropy = plog2p.sum()*-1

        return entropy

    def calculate_significance_of_rules(self, covered_data):
        """
        Function to check the rule significance using the 
        likelihood ratio statistical test where the observed frequency 
        of the class in the coverage of the rule is compared to the 
        observed frequencies of the classes in the training data.
        """
        covered_classes = covered_data['methodology']
        covered_num_of_records = len(covered_classes)
        covered_counts = covered_classes.value_counts()
        covered_probs = covered_counts.divide(covered_num_of_records)

        train_classes = self.train['methodology']
        train_num_of_records = len(train_classes)
        train_counts = train_classes.value_counts()
        train_probs = train_counts.divide(train_num_of_records)

        significance_of_rules = covered_probs.multiply(
            np.log(covered_probs.divide(train_probs))).sum()*2

        return significance_of_rules

    def calculate_laplace_accuracy(self, covered_data):
        """
        Function to calculate laplace accuracy of a rule
        """

        class_series = covered_data['methodology']
        class_tally = class_series.value_counts()
        num_of_records = len(class_series)
        num_classes = len(class_tally)
        num_predicted_class = class_tally.iloc[0]
        laplace_accuracy = (num_of_records + num_classes -
                            num_predicted_class - 1)/(num_of_records + num_classes)
        return laplace_accuracy


if __name__ == '__main__':
    CN2_class = CN2algorithm()
    CN2_class.main()
