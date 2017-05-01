from main import feature_program as fp
from main import classify_program as cp

if __name__ == '__main__':
    fp.init_feature_set('sift')
    fp.learn_vocabulary('sift')
    cp.train_classifier('sift')
    cp.classify('sift ', 'svc')
