import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.io import loadmat
import scipy.signal
import pandas as pd
import seaborn as sns

mat_data = loadmat('usps_benchmark.mat')
maindata = mat_data['benchmarkdata']

def extractfeatures(digdata):
    # Count the number of non-zero pixels in each row
    x = np.count_nonzero(digdata)
    return x

def extractmyfeatures(digdata): #function which applies a 5x5 convolution max pooling and a 3x3 convolution
    image_matrix = digdata.reshape((16, 16))
    kernel_conv1 = np.ones((5, 5)) / 25 
    conv1_result = scipy.signal.convolve2d(image_matrix, kernel_conv1, mode='valid')
    pooled_result = scipy.signal.convolve2d(conv1_result, np.ones((2, 2)), mode='valid')[::2, ::2]
    kernel_conv2 = np.ones((3, 3)) / 9  
    conv2_result = scipy.signal.convolve2d(pooled_result, kernel_conv2, mode='valid')
    flattened_result = conv2_result.flatten()
    return flattened_result

def Calculate_distance(instance1, instance2):
    dist = np.sqrt(np.sum((instance1 - instance2) ** 2))
    return dist

def find_neighbors(k, instance, x_train, y_train):
    distances = []
    for i in range(len(x_train)):
        dist = Calculate_distance(instance, x_train[i])
        distances.append((i, dist))
    distances = sorted(distances, key=lambda x: x[1])
    neighbors_indices = [index for index, _ in distances[:k]]
    return y_train[neighbors_indices]

def get_response(arr):
    return np.bincount(arr).argmax()

# Number of samples
num_samples = 1500
samples = np.zeros((num_samples, 256))
labels = np.zeros(num_samples)

# Create 100 samples of digit '3'
for i in range(500):
    random_index = np.random.randint(maindata.shape[1])  # Choose a random index from maindata
    samples[i] = maindata[:, random_index, 2].reshape((256,))  #digit '3' is at index 2
    labels[i] = 0  # 0 represents the digit '3'

# Create 100 samples of digit '8'
for i in range(500, 1000):
    random_index = np.random.randint(maindata.shape[1])  # Choose a random index from maindata
    samples[i] = maindata[:, random_index, 7].reshape((256,))  #digit '8' is at index 7
    labels[i] = 1  # 1 represents the digit '8'

for i in range(1000, 1500):
    random_index = np.random.randint(maindata.shape[1])  # Choose a random index from maindata
    samples[i] = maindata[:, random_index, 5].reshape((256,))  #digit '6' is at index 5
    labels[i] = 2  # 2 represents the digit '6'

# Shuffle the samples and labels
indices = np.random.permutation(num_samples)
samples = samples[indices]
labels = labels[indices]

x_all = samples.astype(int)
y_all = labels.astype(int)

x_all_features = np.apply_along_axis(extractmyfeatures, axis=1, arr=x_all)

print("Training samples shape:", x_all_features.shape)
print("Training labels shape:", y_all.shape)

train_accuracies = []
ks = list(range(1, 21))

for k in ks:
    predictions = [get_response(find_neighbors(k, instance, x_all_features, y_all)) for instance in x_all_features]
    accuracy = accuracy_score(y_all, predictions)
    train_accuracies.append(accuracy)
    print(f"Training Accuracy for k={k}: {accuracy}")

# Plot the training accuracies for different values of k
plt.plot(ks, train_accuracies, marker='o')
plt.title("Training Accuracy vs. k")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Training Accuracy")
plt.show()
plt.pause(0.001)
input("Press Enter to close the plot...")
num_splits = 2
testing_accuracies_all_splits = []

for split in range(num_splits):
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_all_features, y_all, test_size=0.5, random_state=split)
    x_train = x_train.astype(int)
    x_test = x_test.astype(int)
    y_test = y_test.astype(int)
    y_train = y_train.astype(int)
    testing_accuracies = []
    
    for k in ks:
        predictions = [get_response(find_neighbors(k, instance, x_train, y_train)) for instance in x_test]
        accuracy = accuracy_score(y_test, predictions)
        testing_accuracies.append(accuracy)
        print(f"Testing Accuracy for k={k} (Split {split+1}): {accuracy}")

    testing_accuracies_all_splits.append(testing_accuracies)

testing_accuracies_all_splits = np.array(testing_accuracies_all_splits)
average_accuracies = np.mean(testing_accuracies_all_splits, axis=0)
std_accuracies = np.std(testing_accuracies_all_splits, axis=0)

plt.errorbar(ks, average_accuracies, yerr=std_accuracies, fmt='o', capsize=5)
plt.title("Testing Accuracy vs. k (Multiple Splits)")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Testing Accuracy")
plt.show()
plt.pause(0.001)
input("Press Enter to close the plot...")

k = 3
x_train, x_test, y_train, y_test = train_test_split(x_all_features, y_all, test_size=0.5, random_state=split)
x_train = x_train.astype(int)
x_test = x_test.astype(int)
y_test = y_test.astype(int)
y_train = y_train.astype(int)
predictions_k3 = [get_response(find_neighbors(k, instance, x_train, y_train)) for instance in x_test]
accuracy_k3 = accuracy_score(y_test, predictions_k3)
conf_matrix_k3 = confusion_matrix(y_test, predictions_k3)

cm_df = pd.DataFrame(conf_matrix_k3, index=np.unique(y_train), columns=np.unique(y_train))
# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

# Print the confusion matrix
print('Confusion Matrix:')
print(cm_df)
accuracy_k3 = np.sum(np.diag(conf_matrix_k3)) / np.sum(conf_matrix_k3)
print('Accuracy:', accuracy_k3)
