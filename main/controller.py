from main import feature_program as fp
from main import classify_program as cp

if __name__ == '__main__':
    fp.init_feature_set()
    fp.learn_vocabulary()
    cp.train_classifier()
    cp.classify()
