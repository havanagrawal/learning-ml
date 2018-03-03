from collections import Counter, namedtuple
import pandas as pd


class DecisionTreeNode(object):
    def __init__(self, feature):
        self.feature = feature
        self.children = {}
        self.label = None

    def add_child(self, feature_value, child_node):
        self.children[feature_value] = child_node

    def print_tree(self, tab=0):
        indent = "\t" * tab
        print(indent + "TreeNode({})".format(self.feature))
        for feature_value in self.children:
            print(indent + feature_value)
            self.children[feature_value].print_tree(tab + 1)


class DecisionTreeLeafNode(DecisionTreeNode):
    def __init__(self, label):
        self.label = label

    def print_tree(self, tab=0):
        indent = "\t" * tab
        print(indent + self.label)


class DecisionTree(object):

    def __init__(self, data, target, features):
        self.root = self._build_tree(data, target, features)

    def _build_tree(self, data, target, features):
        # No more features left, predict majority
        if not features:
            return DecisionTreeLeafNode(self.majority(data, target))

        # If this data has a single output label, simply predict that
        if len(data[target].unique()) == 1:
            return DecisionTreeLeafNode(data[target].iloc[0])

        features = set(features)

        best_split_feature = self.find_best_feature(data, target, features)

        features.remove(best_split_feature)

        unique_values = data[best_split_feature].unique()

        tree_node = DecisionTreeNode(best_split_feature)

        for v in unique_values:
            data_slice = data[data[best_split_feature] == v]
            child_tree = self._build_tree(data_slice, target, features)
            tree_node.add_child(v, child_tree)

        return tree_node

    def predict(self, data_row):

        def predict_helper(node, data_row):
            if node.label is not None:
                return node.label

            feature_value = data_row[node.feature]
            child = node.children[feature_value]
            return predict_helper(child, data_row)

        return predict_helper(self.root, data_row)

    def visualize(self):
        self.root.print_tree()


    def find_best_feature(self, data, target, features):
        BestScore = namedtuple('BestScore', ['feature', 'score'])
        best_score = BestScore(None, 0)

        for feature in features:
            unique_values = data[feature].unique()

            correct_guesses = 0

            for v in unique_values:
                data_slice = data[data[feature] == v]

                # Get the majority label for this feature's value slice
                guess = self.majority(data_slice, target)

                # How many guesses were correct
                correct_guesses += sum(data_slice[target] == guess)

            if correct_guesses > best_score.score:
                best_score = BestScore(feature, correct_guesses)

        return best_score.feature


    def majority(self, data, target):
        labels = data[target]
        frequencies = Counter(labels)
        most_common_pair = frequencies.most_common(1)[0]

        # Return the key that has the maximum frequency
        return most_common_pair[0]

def majority_test():

    data = {
        'dummy_labels_1': [1, 5, 1, 1, 1, 3, 3, 3, 4],
        'dummy_labels_2': [1, 1, 5, 5, 5],
        'dummy_labels_3': [1]
    }

    dummy_tree = DecisionTree(data, 'dummy_labels_1', None)

    assert(dummy_tree.majority(data, 'dummy_labels_1') == 1)
    assert(dummy_tree.majority(data, 'dummy_labels_2') == 5)
    assert(dummy_tree.majority(data, 'dummy_labels_3') == 1)

def test_tree_prediction():
    data = {
        'color': ['red', 'red', 'red', 'orange', 'orange', 'red', 'green', 'green', 'green', 'green'],
        'size': ['medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'big', 'small'],
        'fruit': ['apple', 'apple', 'apple', 'orange', 'orange', 'apple', 'apple', 'apple', 'watermelon', 'grape']
    }
    df = pd.DataFrame(data)
    tree = DecisionTree(df, 'fruit', ['size', 'color'])
    tree.visualize()

    assert tree.predict({'color': 'red', 'size': 'medium'}) == 'apple'
    assert tree.predict({'color': 'green', 'size': 'small'}) == 'grape'
    assert tree.predict({'color': 'green', 'size': 'medium'}) == 'apple'
    assert tree.predict({'color': 'green', 'size': 'big'}) == 'watermelon'
    assert tree.predict({'color': 'orange'}) == 'orange'


def test():
    majority_test()
    test_tree_prediction()

if __name__ == "__main__":
    test()
