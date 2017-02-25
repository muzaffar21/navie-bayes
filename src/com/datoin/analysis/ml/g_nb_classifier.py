__author__ = 'muzaffar'
import json
import math


def get_gaussian_prob_dist(weight, mean, variance):
    var = 2.0 * math.pi
    part_one = 1 / math.sqrt(var * variance)
    zSq = (weight - mean) ** 2
    var3 = (2.0 * variance)
    div = (zSq / var3)
    part_two = math.e ** -div
    gaussian_prob_dist = part_one * part_two
    return gaussian_prob_dist


class classifier():
    label_prob_map = {}
    per_class_per_feature_mv_map = {}

    def __init__(self, model):
        with open(model, "r") as f:
            obj = f.readlines()[0]
        model_json = json.loads(obj)
        self.label_prob_map = model_json["label_prob"]
        self.per_class_per_feature_mv_map = model_json["mv_per_label_per_feature"]

    def get_per_class_probability(self, vector):
        lst = vector.split()
        print  lst
        result_map = {}
        for label in self.label_prob_map:
            prob = self.get_class_prob(label, lst)
            result_map[label] = prob
        return result_map

    def get_class_prob(self, label, vector):
        class_prob = self.label_prob_map[label]

        per_feature_mv_map = self.per_class_per_feature_mv_map[label]
        print per_feature_mv_map
        for i in range(1, len(vector)):
            vec = vector[i].split(":")
            feature = vec[0]
            print feature
            weight = float(vec[1])
            feature_mv = per_feature_mv_map[feature]
            variance = float(feature_mv["variance"])
            # todo handle this condition in better way
            if variance == 0.000:
                variance = 0.000001
            mean = float(feature_mv["mean"])
            print variance
            gaus_prob_dist = get_gaussian_prob_dist(weight, mean, variance)
            class_prob *= gaus_prob_dist

        return class_prob


def main(model, vector):
    obj = classifier(model)
    result = obj.get_per_class_probability(vector)
    print  result


if __name__ == '__main__':
    main("/disk2/home/muzaffar/PycharmProjects/navie-bayes/resources/nb_model.json",
         "1:0.08999999999999997 2:0 3:1.0 4:0 5:1 6:0 7:0.3224754874136049 8:0.08999999999999998 9:0 10:0")
