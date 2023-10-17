

from sklearn.metrics import accuracy_score
import sklearn.linear_model as sklm

class train_model:
    @staticmethod
    def getModel(features_train, features_test, labels_train, labels_test, eta0, max_iter):
        LR = sklm.SGDClassifier(
            random_state=0, learning_rate='constant', eta0=eta0, max_iter=max_iter)
        LR.fit(features_train, labels_train)
        prediction = LR.predict(features_test)
        return accuracy_score(labels_test, prediction)