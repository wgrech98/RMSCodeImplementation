# A NOVEL RECOMMENDER SYSTEM TO DETERMINE THE OPTIMAL SOFTWARE DEVELOPMENT LIFECYCLE MODEL IN SOFTWARE PROJECTS 
This is a readme file for the implementation of the algorithms for my dissertation project

## Background & Objectives
Software Development Lifecycle (SDLC) methodologies provide the software development community with a structural framework to develop software products in an efficient and qualitative manner. Several recommender tools have been developed to facilitate the SDLC selection process; however, literature evidenced that the existing solutions in the market lack transparency when considering the algorithmic process of how the recommender tools arrive to a recommendation. This dissertation addresses the issue by developing a knowledge-based recommender system (RMS) that predicts the ideal SDLC methodology for different project scenarios. Explainable Artificial Intelligence techniques were utilised to facilitate user understanding of the predictions and the algorithmic process of the RMS.
 
## Dataset
The dataset is composed of the characteristics related to the choice of a SDLC methodology. Overall, the dataset  amounted to 143 records.

## Models
The repository hosts three python algorithms, namely the KNN algorithm, the decision tree algorithm, and the CN2 algorithm. 

# CN2 Algorithm 
The CN2 algorithm is a rule-based technique that supersedes its predecessors, namely the ID3 algorithm, and AQ by accounting for the presence of noise in the domain data.

A rule list produced with the CN2 algorithm follows the following notation:

	if Condition_1 then Class_A
	else if Condition_2 then Class_B
	...
	else Default_Class.

The algorithm generates a rule list according to the approach followed by for the dataset described above. Then, the cross validation exercise is utilised to test the algorithm's accuracy and correctness in predicting a SDLC methodology. The results are returned to the user in CSV format. To run the algorithm:

1. Download the entire repository 
2. Unzip all files
3. cd CN2
4. python3 CN2Algorithm.py


# KNN Algorithm 
In KNN, the trained data is compared with test data and distances are calculated using Euclidean distance. It then classifies an instance by finding its nearest neighbors and recommend the top n nearest neighbor SDLC Methodologies. The algorithm runs the ‘KNeighborsClassifier'() with the cross validation technqiue. The predictions are returned with the accuracy achieved for each fold. To run the algorithm:

1. Download the entire repository 
2. Unzip all files
4. python3 KNNalgorithm.py

# Decision Tree Algorithm 
Decision Trees (DTs) are another important rule-based technique represented portrayed as a tree structure. The algorithm splits the dataset into smaller classes according to the most important predictor in the predictor space with an if-then rule set. When running the decision tree algorithm, The algorithm runs the ‘DecisionTreeClassifier() with the cross validation technqiue. The predictions are returned with the accuracy achieved for each fold. A decision tree graph is also returned to the user. To run the algorithm:

1. Download the entire repository 
2. Unzip all files
3. cd DecsionTree
4. python3 decisionTreeAlgorithm.py
