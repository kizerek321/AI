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
    # ...............................
    # ...............................
    # ...............................
    # ...............................
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


