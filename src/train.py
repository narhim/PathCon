import tensorflow as tf
import numpy as np
from collections import defaultdict
from model import PathCon
from utils import sparse_to_tuple


args = None

def decoder(triplets, dict_entities, dict_relations): #TMS
    '''To reverse encoding of entities.'''
    for triplet in triplets:
        head = {i for i in dict_entities if dict_entities[i]==triplet[0]}
        tail = {i for i in dict_entities if dict_entities[i]==triplet[1]}
        relation = {i for i in dict_relations if dict_relations[i]==triplet[2]}
        decoded.append((head,relation,tail))

def write_list_to_file(name,list_triples): #TMS
    with open(name,"w") as f:
        f.write("HEAD\tRELATION\tTAIL\n")
        for triplet in list_triples:
            f.write(str(triplet[0]) + "\t" + str(triplet[1]) + "\t" + str(triplet[2]) + "\n")


def train(model_args, data):
    global args, model, sess, decoded
    decoded = []
    args = model_args

    # extract data
    triplets, paths, n_relations, neighbor_params, path_params, entity_dict, relation_dict = data

    train_triplets, valid_triplets, test_triplets = triplets
    train_edges = np.array(range(len(train_triplets)), np.int32)
    train_entity_pairs = np.array([[triplet[0], triplet[1]] for triplet in train_triplets], np.int32)
    valid_entity_pairs = np.array([[triplet[0], triplet[1]] for triplet in valid_triplets], np.int32)
    test_entity_pairs = np.array([[triplet[0], triplet[1]] for triplet in test_triplets], np.int32)
    train_paths, valid_paths, test_paths = paths

    train_labels = np.array([triplet[2] for triplet in train_triplets], np.int32)
    valid_labels = np.array([triplet[2] for triplet in valid_triplets], np.int32)
    test_labels = np.array([triplet[2] for triplet in test_triplets], np.int32)

    # define the model
    model = PathCon(args, n_relations, neighbor_params, path_params)
    saver = tf.train.Saver()

    # prepare for top-k evaluation
    true_relations = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_relations[(head, tail)].add(relation)
    best_valid_acc = 0.0
    final_res = None  # acc, mrr, mr, hit1, hit3, hit5

    with tf.Session() as sess:
        print('start training ...')
        sess.run(tf.global_variables_initializer()) #The variables are called earlier, but we need to initialize them in order to allocate memory for them and set its initial values.

        for step in range(args.epoch): #For each epoch

            # shuffle training data
            index = np.arange(len(train_labels)) #Give me evenly spaced in intervals btw 0 and length of train labels.
            np.random.shuffle(index) #Shuffle the index.
            if args.use_context:
                train_entity_pairs = train_entity_pairs[index] #Subset entity pairs.
                train_edges = train_edges[index] #Subset number train edges
            if args.use_path:
                train_paths = train_paths[index] #Subset train paths
            train_labels = train_labels[index] #Gold labels

            # training
            s = 0
            while s + args.batch_size <= len(train_labels):
                _, loss = model.train(sess, get_feed_dict(
                    train_entity_pairs, train_edges, train_paths, train_labels, s, s + args.batch_size))
                s += args.batch_size

            # evaluation
            print('epoch %2d   ' % step, end='')

            train_acc, _ = evaluate(train_entity_pairs, train_paths, train_labels)
            valid_acc, _ = evaluate(valid_entity_pairs, valid_paths, valid_labels)
            test_acc, test_scores = evaluate(test_entity_pairs, test_paths, test_labels)

            # show evaluation result for current epoch
            current_res = 'acc: %.4f' % test_acc
            print('train acc: %.4f   valid acc: %.4f   test acc: %.4f' % (train_acc, valid_acc, test_acc))
            mrr, mr, hit1, hit3, hit5 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
            current_res += '   mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5)
            print('           mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5))
            print()
            # update final results according to validation accuracy
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                final_res = current_res

        # show final evaluation result
        print('final results\n%s' % final_res)
        # output test
        decoder(test_triplets, entity_dict,relation_dict) #TMS
        write_list_to_file("predictions_fb15k.tsv",decoded) #TMS


def get_feed_dict(entity_pairs, train_edges, paths, labels, start, end):
    feed_dict = {}

    if args.use_context:
        feed_dict[model.entity_pairs] = entity_pairs[start:end]
        if train_edges is not None:
            feed_dict[model.train_edges] = train_edges[start:end]
        else:
            # for evaluation no edges should be masked out
            feed_dict[model.train_edges] = np.array([-1] * (end - start), np.int32)

    if args.use_path:
        if args.path_type == 'embedding':
            feed_dict[model.path_features] = sparse_to_tuple(paths[start:end])
        elif args.path_type == 'rnn':
            feed_dict[model.path_ids] = paths[start:end]

    feed_dict[model.labels] = labels[start:end]

    return feed_dict


def evaluate(entity_pairs, paths, labels):
    acc_list = []
    scores_list = []

    s = 0
    while s + args.batch_size <= len(labels):
        acc, scores = model.eval(sess, get_feed_dict(
            entity_pairs, None, paths, labels, s, s + args.batch_size))
        acc_list.append(acc)
        scores_list.extend(scores)
        s += args.batch_size

    return float(np.mean(acc_list)), np.array(scores_list)


def calculate_ranking_metrics(triplets, scores, true_relations):

    for i in range(scores.shape[0]):
        head, tail, relation = triplets[i]
        for j in true_relations[head, tail] - {relation}:
            scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1)
    relations = np.array(triplets)[0:scores.shape[0], 2]
    sorted_indices -= np.expand_dims(relations, 1)
    zero_coordinates = np.argwhere(sorted_indices == 0)
    rankings = zero_coordinates[:, 1] + 1

    mrr = float(np.mean(1 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit5 = float(np.mean(rankings <= 5))

    return mrr, mr, hit1, hit3, hit5
