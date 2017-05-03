from main import feature_program as fp
from main import classify_program as cp

if __name__ == '__main__':
    fp.init_feature_set('brisk')
    fp.learn_vocabulary('brisk')
    cp.train_classifier('brisk')
    cp.classify('brisk', 'svc')
