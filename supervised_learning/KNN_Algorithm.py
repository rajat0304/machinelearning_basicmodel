import numpy as np
from collections import Counter
#calculation of distance
def euclidean_distance(point1,point2):
  return np.sqrt(np.sum((np.array(point1)-np.array(point2))**2))
#prediction of function
def knn_prediction(training_data,training_labels,test_point,k):
  distance_list=[]
  for i in range(len(training_data)):
    dist = euclidean_distance(test_point,training_data[i])
    distance_list.append((dist, training_labels[i]))

  distance_list.sort(key=lambda x:x[0])
  k_nearest_labels=[label for _,label in distance_list[:k]]
  return Counter(k_nearest_labels).most_common(1)[0][0]

training_data=[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]]
training_labels=['A','A','A','A','B','B','C','C']
test_point=[4,5]
k=3
prediction=knn_prediction(training_data,training_labels,test_point,k)
print(prediction)

