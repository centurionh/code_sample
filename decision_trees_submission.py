from __future__ import division

import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.

    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.

    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the provided data.

    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = DecisionNode(None, None, lambda feature: feature[0] ==1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)

    D1 = DecisionNode(None, None, lambda feature: feature[0]==1)
    
    D2 = DecisionNode(None, None, lambda feature: feature[1]==1)
    D2.left = DecisionNode(None, None, None, 0)
    D2.right = DecisionNode(None, None, None, 1)

    D3 = DecisionNode(None, None, lambda feature: feature[2]==1)
    D3.left = DecisionNode(None, None, None, 0)
    D3.right = DecisionNode(None, None, None, 1)

    D4 = DecisionNode(None, None, lambda feature: feature[3]==1)
    D4.left = D2
    D4.right = D3

    decision_tree_root.right = D4

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """
    true_positive = sum(list(map(lambda pred_y, y: 1 if (pred_y==1 and y==1) else 0, classifier_output, true_labels))) 
    false_negative = sum(list(map(lambda pred_y, y: 1 if (pred_y==0 and y==1) else 0, classifier_output, true_labels)))
    
    true_negative = sum(list(map(lambda pred_y, y: 1 if (pred_y==0 and y==0) else 0, classifier_output, true_labels))) 
    false_positive = sum(list(map(lambda pred_y, y: 1 if (pred_y==1 and y==0) else 0, classifier_output, true_labels)))

    Output = [[true_positive, false_negative], [false_positive, true_negative]]
    return Output


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.

    Precision is measured as:
        true_positive/ (true_positive + false_positive)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    true_positive = sum(list(map(lambda pred_y, y: 1 if (pred_y==1 and y==1) else 0, classifier_output, true_labels)))
    false_positive = sum(list(map(lambda pred_y, y: 1 if (pred_y==1 and y==0) else 0, classifier_output, true_labels)))
    return true_positive / (true_positive + false_positive)


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.

    Recall is measured as:
        true_positive/ (true_positive + false_negative)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    true_positive = sum(list(map(lambda pred_y, y: pred_y==y if y==1 else False, classifier_output, true_labels))) 
    false_negative = sum(list(map(lambda pred_y, y: 1 if (pred_y==0 and y==1) else 0, classifier_output, true_labels)))
    return true_positive / (true_positive + false_negative)


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.
    true_positive = sum(list(map(lambda pred_y, y: 1 if (pred_y==1 and y==1) else 0, classifier_output, true_labels))) 
    false_negative = sum(list(map(lambda pred_y, y: 1 if (pred_y==0 and y==1) else 0, classifier_output, true_labels)))
    
    true_negative = sum(list(map(lambda pred_y, y: 1 if (pred_y==0 and y==0) else 0, classifier_output, true_labels))) 
    false_positive = sum(list(map(lambda pred_y, y: 1 if (pred_y==1 and y==0) else 0, classifier_output, true_labels)))
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.

    Returns:
        Floating point number representing the gini impurity.
    """
    if len(class_vector) > 0:
        p = sum(class_vector) / len(class_vector)
    else: 
        p = 0
    gini = 1 - p**2 - (1.0-p)**2
    return gini


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    gini0 = gini_impurity(previous_classes)
    gini1 = 0 
    N = sum(list(map(lambda x: len(x), current_classes))) 
    for c in current_classes:
        gini1 += len(c) / N * gini_impurity(c)
    return gini0 - gini1

class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """
        # Method 1
        class_map = dict(Counter(classes))
        if len(class_map) == 1:
            root = DecisionNode(None, None, None, classes[0])
            return root
        elif depth == self.depth_limit:
            label = None
            ct = 0
            for key, value in class_map.items():
                if value >= ct:
                    ct = value
                    label = key
            root = DecisionNode(None, None, None, label)
            return root

        else:
            alpha_best = -1
            alpha_best_ig = -9999
            alpha_best_threshold = -9999

            for idx in range(len(features[0])):
                alpha = features.transpose()[idx]
                alpha_min = min(alpha)
                alpha_max = max(alpha)
                if alpha_max == alpha_min:
                    for key, value in class_map.items():
                        label = None
                        ct = 0
                        if value >= ct:
                            ct = value
                            label = key
                    return DecisionNode(None,None,None,label)

                step = 100
                best_threshold = alpha_min
                best_gain = -9999

                for threshold in np.arange(alpha_min, alpha_max, (alpha_max-alpha_min)/step):
                    left = classes[np.where(alpha < threshold)]
                    right = classes[np.where(alpha >= threshold )]

                    gain = gini_gain(classes,[left, right])

                    if gain > best_gain:
                        best_gain = gain
                        best_threshold = threshold
                            
                if best_gain > alpha_best_ig:
                    alpha_best_ig = best_gain
                    alpha_best = idx
                    alpha_best_threshold = best_threshold
            
            idx = np.where(features[:,alpha_best] > alpha_best_threshold)
            features1 = features[idx]
            classes1 = classes[idx]
            left = self.__build_tree__(features1, classes1, depth+1)

            idx = np.where(features[:,alpha_best] <= alpha_best_threshold)
            features2 = features[idx]
            classes2 = classes[idx]
            right = self.__build_tree__(features2, classes2, depth+1)

            root = DecisionNode(left, right, lambda feature: feature[alpha_best] > alpha_best_threshold)
            return root

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.

        Args:
            features (list(list(int)): List of features.

        Return:
            A list of class labels.
        """

        class_labels = [self.root.decide(feature) for feature in features]
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """
    n_features = len(dataset[0][0])
    dataset = np.concatenate((dataset[0], dataset[1].reshape(len(dataset[0]),1)), axis=1)
    np.random.shuffle(dataset)

    output = []
    for i in range(0, k):
        test_start = i * (dataset.shape[0]//k)
        test_end = (i+1) * (dataset.shape[0]//k)

        test_features = dataset[test_start:test_end, :n_features]
        test_classes = dataset[test_start:test_end, -1]

        train_features = np.concatenate((dataset[0:test_start, :n_features], dataset[test_end:, :n_features]), axis=0)
        train_classes = np.concatenate((dataset[0:test_start, -1], dataset[test_end:, -1]), axis=0)
        
        output.append(([train_features,train_classes],[test_features,test_classes]))
    
    return output


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.features = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.

            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """
        for i in range(self.num_trees):
            data_idx = np.random.choice(features.shape[0], size=int(self.example_subsample_rate * features.shape[0]), replace=True)
            feature_idx = np.random.choice(features.shape[1], size=int(self.attr_subsample_rate * features.shape[1]), replace=False)
            self.features.append(feature_idx)

            features_subset = features[data_idx][:,feature_idx]
            classes_subset = classes[data_idx]
            
            DT = DecisionTree(self.depth_limit)
            DT.fit(features_subset,classes_subset)
            self.trees.append(DT)
            
    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (list(list(int)): List of features.
        """

        output = []
        # for index, tree in enumerate(self.trees):
        for i in range(len(self.trees)):
            tree = self.trees[i]
            ret = tree.classify(features[:,self.features[i]])
            output.append(ret)

        output2 =  np.array(output).transpose()
        result = []
        for i in range(output2.shape[0]):
            class_map = dict(Counter(output2[i,:]))
            label = None
            ct = 0
            for key, value in class_map.items():
                if value >= ct:
                    ct = value
                    label = key
            result.append(label)
        return result

class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, num_trees=10, depth_limit=4, example_subsample_rate=0.5, attr_subsample_rate=0.8):
        """Create challenge classifier.

        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        self.trees = []
        self.features = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build the underlying tree(s).

            Fit your model to the provided features.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        for i in range(self.num_trees):
            data_idx = np.random.choice(features.shape[0], size=int(self.example_subsample_rate * features.shape[0]), replace=True)
            feature_idx = np.random.choice(features.shape[1], size=int(self.attr_subsample_rate * features.shape[1]), replace=False)
            self.features.append(feature_idx)

            features_subset = features[data_idx][:,feature_idx]
            classes_subset = classes[data_idx]
            
            DT = DecisionTree(self.depth_limit)
            DT.fit(features_subset,classes_subset)
            self.trees.append(DT)

    def classify(self, features):
        """Classify a list of features.

        Classify each feature in features as either 0 or 1.

        Args:
            features (list(list(int)): List of features.

        Returns:
            A list of class labels.
        """
        output = []
        # for index, tree in enumerate(self.trees):
        for i in range(len(self.trees)):
            tree = self.trees[i]
            ret = tree.classify(features[:,self.features[i]])
            output.append(ret)

        output2 =  np.array(output).transpose()
        result = []
        for i in range(output2.shape[0]):
            class_map = dict(Counter(output2[i,:]))
            label = None
            ct = 0
            for key, value in class_map.items():
                if value >= ct:
                    ct = value
                    label = key
            result.append(label)
        return result


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Args:
            data: data to be added to array.

        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Bonnie time to beat: 0.09 seconds.

        Args:
            data: data to be sliced and summed.

        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        out = data * data + data
        return out

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Args:
            data: data to be added to array.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Bonnie time to beat: 0.07 seconds

        Args:
            data: data to be sliced and summed.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        row_sum = np.sum(data[:100], axis=1)
        idx = np.argmax(row_sum)
        return (max(row_sum), idx)

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Bonnie time to beat: 15 seconds

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.
        data = data.flatten()
        data = data[data>0]
        data = dict(Counter(data))
        return data.items()

def return_your_name():
    # return your name
    # TODO: finish this
    return "Zhengyang He"
