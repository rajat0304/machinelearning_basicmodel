from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
cancer=load_breast_cancer()
X=cancer.data[:,:2]
Y=cancer.target
svm=SVC(kernel='linear',C=1)
svm.fit(X,Y)
DecisionBoundaryDisplay.from_estimator(
    svm,
    X,
    response_method='predict',
    cmap="Pastel1",
    xlabel=cancer.feature_names[0],
    ylabel=cancer.feature_names[1],
)
#matplot lib to visiualize data
plt.scatter(X[:,0],X[:,1],
            c=Y,
            s=20,
            edgecolor="k")
plt.show()

