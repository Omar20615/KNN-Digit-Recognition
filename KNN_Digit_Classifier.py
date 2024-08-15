import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.io import loadmat

# Load MATLAB .mat file
mat_data = loadmat('usps_main.mat')
maindata = mat_data['maindata']

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
num_samples = 300
samples = np.zeros((num_samples, 256))
labels = np.zeros(num_samples)

# Create 100 samples of digit '3'
for i in range(100):
    random_index = np.random.randint(maindata.shape[1])  # Choose a random index from maindata
    samples[i] = maindata[:, random_index, 2].reshape((256,))  #digit '3' is at index 2
    labels[i] = 0  # 0 represents the digit '3'

# Create 100 samples of digit '8'
for i in range(100, 200):
    random_index = np.random.randint(maindata.shape[1])  # Choose a random index from maindata
    samples[i] = maindata[:, random_index, 7].reshape((256,))  #digit '8' is at index 7
    labels[i] = 1  # 1 represents the digit '8'

for i in range(200, 300):
    random_index = np.random.randint(maindata.shape[1])  # Choose a random index from maindata
    samples[i] = maindata[:, random_index, 5].reshape((256,))  #digit '6' is at index 5
    labels[i] = 2  # 2 represents the digit '6'

# Shuffle the samples and labels
indices = np.random.permutation(num_samples)
samples = samples[indices]
labels = labels[indices]

# Assign the samples and labels arrays to x_train and y_train, respectively
x_train = samples.astype(int)
y_train = labels.astype(int)

print("Training samples shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

train_accuracies = []
ks = list(range(1, 21))

for k in ks:
    predictions = [get_response(find_neighbors(k, instance, x_train, y_train)) for instance in x_train]
    accuracy = accuracy_score(y_train, predictions)
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
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.5, random_state=split)
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

# Task 5
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


def showdata(data, labels, guess=None):
    num_examples = data.shape[0]

    # Sort data and labels
    sort_order = np.argsort(labels)
    data = data[sort_order, :]
    labels = labels[sort_order]

    # Check if predictions are provided
    testing = True if guess is not None else False
    if testing:
        # Flatten guess if it's a multi-dimensional array
        guess = np.ravel(guess)
        guess = guess[sort_order]

    # Calculate the number of digits to put in the square
    side = int(np.ceil(np.sqrt(num_examples)))

    # Set up border parameters
    border = 3
    frame_width = 16 + 2 * border

    # Initialize the main matrix to display
    m = np.zeros((side * frame_width, side * frame_width))

    n = 0
    mistakes = 0

    for row in range(0, side * frame_width, frame_width):
        for col in range(0, side * frame_width, frame_width):
            # Check if we reached the end of examples
            if n >= num_examples:
                break

            # Retrieve the digit pixels
            digit = data[n, :].reshape((16, 16))

            # Put a black border around it
            frame = np.zeros((frame_width, frame_width))
            frame[border:border+16, border:border+16] = digit
            digit = frame

            # Draw a further white border around the digit if there's a mistake
            if testing and labels[n] != guess[n]:
                digit[border, border:frame_width-border] = 255  # top of white 'mistake' box
                digit[frame_width-border, border:frame_width-border] = 255  # bottom
                digit[border:frame_width-border, border] = 255  # left
                digit[border:frame_width-border, frame_width-border] = 255  # right
                mistakes += 1

            # Put it in the main matrix
            m[row:row+frame_width, col:col+frame_width] = digit

            # Increment which example we're dealing with
            n += 1

    

    plt.imshow(m, cmap='gray')
    plt.axis('off')
    plt.title(f"{mistakes} errors from {num_examples} ({(mistakes / num_examples) * 100:.2f}%)", fontsize=16)
    plt.show()
    plt.pause(0.001)
    input("Press Enter to close the plot...")

chosen_k = 3
x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.5, random_state=split)
x_train = x_train.astype(int)
x_test = x_test.astype(int)
y_test = y_test.astype(int)
y_train = y_train.astype(int)
testing_accuracies = []


predictions = [get_response(find_neighbors(chosen_k, instance, x_train, y_train)) for instance in x_test]
showdata(x_test, y_test, predictions)
