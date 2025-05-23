import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import dec_tree_func as dt

#os.system('cls')

#file_name = "lenses\lenses.txt"
file_name = 'car/car.txt'
#file_name = 'tic/tic.txt'
#file_name = 'letters1.txt'
#file_name = 'letters2.txt'
#file_name = 'letters3.txt'
#file_name = 'letters5.txt'

num_of_random_features = 0

number_of_experiments = 10


# part of the examples for building a tree in percentage (0-100%) other axamples can be used 
# for pruning as validation subset:
part_for_constr = 100 # (..... you can choose the proportions for tree construction and validation ) 



print("\n\n\nbuilding tree for " + file_name)
# loading data from file and dividing it into training and test sets:
# examples with odd numbers (1,3,5,...) for training
# examples with even numbers for test

example_set = dt.load_data(file_name) 
# Adding worthless information:
if num_of_random_features > 0:
    example_set = np.concatenate(\
        (np.array(np.ceil(np.random.rand(len(example_set),num_of_random_features)*12),dtype=int),example_set), axis = 1) 
#print(" examples = \n" + str(examples))

[num_of_examples, num_of_columns] = example_set.shape
num_of_training_examples = num_of_examples//2
num_of_test_examples = num_of_examples - num_of_training_examples
number_for_constr = int(np.ceil(num_of_training_examples*part_for_constr/100))   # number of examples for tree construction 
print("number of examples: for tree construction = "+ str(number_for_constr) + ", for validation = "+ \
      str(num_of_training_examples-number_for_constr) + ", for test = " + str(num_of_test_examples))
#print("num_of_training_examples = " + str(num_of_training_examples)+ " num_of_test_examples = "+str(num_of_test_examples))

mean_test_error = 0
mean_test_error_prun = 0



for experiment in range(number_of_experiments):

    np.random.shuffle(example_set)       # random permutation of example set
    #print(" examples after shuffle = \n" + str(examples))

    train_vectors = example_set[0:num_of_training_examples,0:num_of_columns-1]
    train_classes = example_set[0:num_of_training_examples,num_of_columns-1]
    test_vectors = example_set[num_of_training_examples:num_of_examples,0:num_of_columns-1]
    test_classes = example_set[num_of_training_examples:num_of_examples,num_of_columns-1]

    #print("train_vectors = \n" + str(train_vectors))
    #print("train_classes = \n" + str(train_classes))
    #print("test_vectors = \n" + str(test_vectors))
    #print("test_classes = \n" + str(test_classes))

    tree = dt.build_tree(train_vectors, train_classes)

    #print("final tree = \n" + str(tree))
    if experiment == 0: dt.print_tree("tree_"+str(experiment)+".txt",tree)

    dist = dt.distribution(train_vectors,train_classes,tree)
    #print("distribution = \n" + str(dist))
    d =  dt.depth(tree)
    #print("depth = " + str(d))

    num_of_training_errors = dt.calc_error(train_vectors,train_classes,tree)
    num_of_test_errors = dt.calc_error(test_vectors,test_classes,tree)

    mean_test_error += num_of_test_errors/num_of_test_examples/number_of_experiments 


    # pruning - training example set division for tree construction and validation: 
    train_vectors_constr = train_vectors[:number_for_constr]
    train_classes_constr = train_classes[:number_for_constr]
    train_vectors_valid =  train_vectors[number_for_constr:]
    train_classes_valid =  train_classes[number_for_constr:]


    tree_pruned = dt.build_tree(train_vectors_constr, train_classes_constr)
    [num_of_rows, num_of_nodes] = tree_pruned.shape

    # pruning:
    # a place for your algorithm here!
    # Do not use a test set for training (tree construction or pruning)!!!!!!!!!!!!!!!
    # tree_pruned is already built using train_vectors_constr, train_classes_constr

    if len(train_vectors_valid) > 0:
        print("Performing Reduced Error Pruning using validation set.")
        current_best_tree = np.copy(tree_pruned)
        # Calculate initial error on validation set using the tree built on the construction set
        best_validation_error = dt.calc_error(train_vectors_valid, train_classes_valid, current_best_tree)

        # Get initial class distributions on the construction set for determining majority class
        # This distribution helps in deciding the class of a new leaf node [cite: 27, 30]
        distributions_on_construction_set = dt.distribution(train_vectors_constr, train_classes_constr,
                                                            current_best_tree)

        while True:
            candidate_tree_for_this_pass = None
            error_of_candidate_tree_this_pass = best_validation_error

            # Iterate over all nodes to find the best single prune
            _, num_nodes = current_best_tree.shape

            # distributions_on_construction_set should be updated if current_best_tree changes,
            # or calculated based on the specific node's examples from construction set.
            # For simplicity, we can re-calculate it for the current_best_tree before checking nodes.
            distributions_on_construction_set = dt.distribution(train_vectors_constr, train_classes_constr,
                                                                current_best_tree)

            for node_idx in range(num_nodes):
                # Check if it's an internal node (has children) [cite: 10]
                if np.sum(current_best_tree[:-1, node_idx]) != 0:
                    temp_pruned_tree = np.copy(current_best_tree)

                    # Determine majority class for this node using construction set examples that reach it
                    node_class_counts = distributions_on_construction_set[:, node_idx]

                    majority_class = 0
                    if np.sum(node_class_counts) > 0:
                        majority_class = np.argmax(node_class_counts) + 1  # Class numbers are 1-based [cite: 3]
                    else:
                        # Fallback: if no construction examples reach this node (unlikely for a fresh tree)
                        # assign the most frequent class from the entire construction set or a default.
                        if len(train_classes_constr) > 0:
                            unique_classes_constr, counts_constr = np.unique(train_classes_constr, return_counts=True)
                            majority_class = unique_classes_constr[np.argmax(counts_constr)]
                        else:
                            majority_class = 1  # Default if construction set is empty

                    # Prune: convert node_idx to a leaf [cite: 9, 10]
                    temp_pruned_tree[:-1, node_idx] = 0  # Set all child pointers to zero
                    temp_pruned_tree[-1, node_idx] = majority_class  # Set node to be a leaf of majority_class

                    # Evaluate this pruned tree on the validation set
                    current_temp_validation_error = dt.calc_error(train_vectors_valid, train_classes_valid,
                                                                  temp_pruned_tree)

                    # If this prune results in a better or equal error, it's a candidate
                    if current_temp_validation_error <= error_of_candidate_tree_this_pass:
                        error_of_candidate_tree_this_pass = current_temp_validation_error
                        candidate_tree_for_this_pass = np.copy(temp_pruned_tree)

            # If a prune was found in this pass that improved or maintained validation error
            if candidate_tree_for_this_pass is not None and error_of_candidate_tree_this_pass < best_validation_error:  # Strictly better
                current_best_tree = np.copy(candidate_tree_for_this_pass)
                best_validation_error = error_of_candidate_tree_this_pass
                # Continue to the next iteration of pruning
            else:
                # No single prune in this pass improved the validation error, so stop
                break

        tree_pruned = np.copy(current_best_tree)
        # (Continued from above)
    else:
        print("No validation set available. Applying a simple heuristic pruning method.")
        # Heuristic: Prune internal nodes that cover fewer than a threshold number of examples
        # from the construction set.
        MIN_SAMPLES_THRESHOLD = 5  # Define a heuristic threshold

        # Calculate distributions using the construction set and the initially built tree
        distributions_heuristic = dt.distribution(train_vectors_constr, train_classes_constr, tree_pruned)

        _, num_nodes_heuristic = tree_pruned.shape
        # Iterate backwards (from deeper nodes potentially)
        for node_idx in range(num_nodes_heuristic - 1, -1, -1):
            # Check if it's currently an internal node
            if np.sum(tree_pruned[:-1, node_idx]) != 0:
                num_samples_at_node = np.sum(distributions_heuristic[:, node_idx])

                # If node covers few samples (but more than 0)
                if 0 < num_samples_at_node < MIN_SAMPLES_THRESHOLD:
                    # Determine majority class for this node
                    majority_class_node = np.argmax(
                        distributions_heuristic[:, node_idx]) + 1  # Class numbers are 1-based

                    # Convert this node to a leaf
                    tree_pruned[:-1, node_idx] = 0
                    tree_pruned[-1, node_idx] = majority_class_node
                    # print(f"Heuristically pruned node {node_idx} to class {majority_class_node} (samples: {num_samples_at_node})")
    # ...............................


    
    if experiment == 0: dt.print_tree("tree_pruned_"+str(experiment)+".txt",tree_pruned)
    num_of_training_errors_prun = dt.calc_error(train_vectors,train_classes,tree_pruned)
    num_of_test_errors_prun = dt.calc_error(test_vectors,test_classes,tree_pruned)

    print("error: train = " + str(np.round(num_of_training_errors/num_of_training_examples,4)) + " test = " + 
    str(np.round(num_of_test_errors/num_of_test_examples,4)) + "  after pruning: train = " + 
    str(np.round(num_of_training_errors_prun/num_of_training_examples,4)) + " test = " + str(np.round(num_of_test_errors_prun/num_of_test_examples,4)))

    mean_test_error_prun += num_of_test_errors_prun/num_of_test_examples/number_of_experiments 

print("mean test error: before pruning = " + str(np.round(mean_test_error,4)) + " after pruning = " + str(np.round(mean_test_error_prun,4)) +
"\nreduction of average test error = " + str(100*(mean_test_error-mean_test_error_prun)/mean_test_error) + "%")


