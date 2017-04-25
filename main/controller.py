from main import feature_program as fp
from main import classify_program as cp

if __name__ == '__main__':
    fp.initFeatureSet()
    fp.learnVocabulary()
    cp.trainClassifier()
    cp.classify()
