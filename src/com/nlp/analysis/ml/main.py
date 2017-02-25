__author__ = 'muzaffar'

from g_nb_trainer import train_navie_bayes


def create_model(root, training_file):
    if not root.endswith("/"):
        root += "/"
    training_file = root + training_file
    model_file = root + 'nb_model.json'
    train_navie_bayes(training_file, model_file)


def main(root, training_file):
    create_model(root, training_file)


if __name__ == '__main__':
    main('/disk2/home/muzaffar/PycharmProjects/navie-bayes/resources', 'vector.txt')
