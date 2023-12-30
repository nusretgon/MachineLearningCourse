import pandas as pd
import numpy as np
import pydot
from PIL import Image
import io

train_data = pd.read_excel("trainDATA.xlsx")
test_data = pd.read_excel("testDATA.xlsx")

def calculate_variable_impurity(data, target_column):
    if len(data) == 0:
        return 0
    counts = data[target_column].value_counts()
    probabilities = counts / counts.sum()
    variable_impurity = 1 - np.sum(np.square(probabilities))
    return variable_impurity

def calculate_variable_for_split(data, split_attribute, target_column):
    unique_values = data[split_attribute].unique()
    weighted_variable = 0
    for value in unique_values:
        subset = data[data[split_attribute] == value]
        variable_impurity = calculate_variable_impurity(subset, target_column)
        weighted_variable += (len(subset) / len(data)) * variable_impurity
    return weighted_variable

def find_best_split(data, target_column):
    attributes = data.columns.drop(target_column)
    min_variable = 1
    best_attribute = None
    for attribute in attributes:
        variable = calculate_variable_for_split(data, attribute, target_column)
        if variable < min_variable:
            min_variable = variable
            best_attribute = attribute
    return best_attribute

def build_decision_tree(data, target_column, max_depth=6, depth=0):
    if len(data[target_column].unique()) == 1 or (max_depth is not None and depth >= max_depth):
        return data[target_column].iloc[0]

    best_attribute = find_best_split(data, target_column)
    if best_attribute is None:
        return data[target_column].mode()[0]

    tree = {}
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        tree[(best_attribute, value)] = build_decision_tree(subset, target_column, max_depth, depth + 1)

    return tree

def predict(instance, tree):
    if not isinstance(tree, dict):
        return tree
    for key, subtree in tree.items():
        attribute, value = key
        if instance[attribute] == value:
            return predict(instance, subtree)

decision_tree = build_decision_tree(train_data, 'Car Acceptibility')

def visualize_decision_tree(tree, parent_name, graph, counter=[0]):
    if not isinstance(tree, dict):
        leaf_name = f"leaf_{counter[0]}"
        graph.add_node(pydot.Node(leaf_name, label=str(tree), shape="ellipse"))
        graph.add_edge(pydot.Edge(parent_name, leaf_name))
        counter[0] += 1
    else:
        for key, subtree in tree.items():
            attribute, value = key
            child_name = f"{attribute}_{value}_{counter[0]}"
            graph.add_node(pydot.Node(child_name, label=f"{attribute}\n={value}", shape="box"))
            graph.add_edge(pydot.Edge(parent_name, child_name))
            counter[0] += 1
            visualize_decision_tree(subtree, child_name, graph, counter)


test_data['Predicted Values'] = test_data.apply(predict, axis=1, tree=decision_tree)
accuracy = np.mean(test_data['Predicted Values'] == test_data['Car Acceptibility'])
accuracy = accuracy * 100
print("Accuracy:",accuracy)

graph = pydot.Dot(graph_type="digraph")

# Visualize the decision tree
visualize_decision_tree(decision_tree, "root", graph)

# Save the graph to a PNG file
graph_png = graph.create_png(prog='dot')

# Convert the PNG byte stream to an image
decision_tree_image = Image.open(io.BytesIO(graph_png))
decision_tree_image.show()
