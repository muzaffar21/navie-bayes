__author__ = 'muzaffar'
import json


def get_class_probabilities(label_map, total_vectors):
    for k, v in label_map.items():
        label_map[k] = float(v) / float(total_vectors);
    return label_map


def get_mean_variance_per_label_per_feature(label_feature_map, label_map):
    """
    label_feature_map contains list of feature weights per feature per label
    label_map contains each label an it occurrence number in training data.

    """
    per_label_per_feature_mv_map = {}
    for label in label_feature_map.keys():
        f_map = label_feature_map[label]
        # now f_map is map of features and their weight list.
        per_feature_m_v_map = {}
        for feature in f_map.keys():
            lst = f_map[feature]
            f_weight_sum = sum(lst)
            label_count = label_map[label]
            mean = f_weight_sum / label_count
            var = 0.0
            for feature_weight in lst:
                variance = (feature_weight - mean) ** 2
                var += variance

            # remaining no. of training examples in which the current feature is not present
            remaining = label_count - len(lst)
            for i in range(1, remaining):
                variance = (0 - mean) ** 2
                var += variance
            var = var / label_count
            mean_var_map = {"mean": mean, "variance": var}
            per_feature_m_v_map[feature] = mean_var_map

        per_label_per_feature_mv_map[label] = per_feature_m_v_map

    return per_label_per_feature_mv_map


def get_feature_and_label_maps(vector_list, label_map, label_feature_map):
    # the format of vector is assumed to be light svm type starting with label number and then pair of feature
    # dimension and weight separated by ':' and each pair separated by space ex : 1 1:2.19 2:3.24 5:4.14. here
    # starting 1 is label or class  number, 1:2.19 means dimension 1 with weight 2.19 similarly we can create n
    # dimensions. the vector can be sparse.
    for vector in vector_list:
        list = vector.split(" ")
        label = list[0]
        if not label_map.__contains__(label):
            label_map[label] = 1
            feature_map = {}
            label_feature_map[label] = feature_map
        else:
            label_map[label] = label_map[label] + 1

        temp_map = label_feature_map[label]
        for i in range(1, len(list)):
            vector_point = list[i].split(":")
            feature = vector_point[0]
            weight = float(vector_point[1])
            if not temp_map.__contains__(feature):
                lst = []
                lst.append(weight)
                temp_map[feature] = lst
            else:
                lst = temp_map[feature]
                lst.append(weight)
                temp_map[feature] = lst
        label_feature_map[label] = temp_map


# End Of Method

def train_navie_bayes(training_file, model_file):
    label_map = {}
    label_feature_map = {}
    with open(training_file, 'r') as f:
        vector_list = f.readlines()
    total_vectors = len(vector_list)
    print "started training Navie Bayes for " + str(total_vectors) + "from vector file " + training_file

    get_feature_and_label_maps(vector_list, label_map, label_feature_map)
    f_map = get_mean_variance_per_label_per_feature(label_feature_map, label_map)
    get_class_probabilities(label_map, total_vectors)
    final_model_map = {"label_prob": label_map, "mv_per_label_per_feature": f_map}
    model = json.dumps(final_model_map)
    writer = open(model_file, 'w')
    writer.write(model)
    writer.close()
    print "finished training , Model at " + model_file
