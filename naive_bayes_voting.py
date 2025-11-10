import csv
import random
from collections import defaultdict, Counter

DATA_PATH = "data/house-votes-84.data" # 1. Load the dataset
NUM_FOLDS = 10   # Number of folds for cross-validation



def load_dataset(filepath):
    dataset = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 0:
                continue 
            dataset.append(row)
    return dataset


''' Preprocessing(handling missing values('?') by replace
  them with the most frequesnt value(mode) for that attribute 
  *within the same class* (democrat or republican))'''
def handle_missing_values(dataset):
   
    # Separate data by class
    class_groups = defaultdict(list)
    for row in dataset:
        label = row[0]
        class_groups[label].append(row)

    # Compute most common (mode) vote per column within each class
    class_modes = {}
    for label, rows in class_groups.items():
        columns = list(zip(*rows)) 
        modes = []
        for col_idx, col in enumerate(columns[1:], start=1):  
            votes = [v for v in col if v != '?']
            mode = Counter(votes).most_common(1)[0][0] if votes else 'y'
            modes.append(mode)
        class_modes[label] = modes

    # Replace '?' with the class-specific mode for that column
    cleaned_data = []
    for row in dataset:
        label = row[0]
        fixed_row = [label]
        for i, v in enumerate(row[1:]):
            if v == '?':
                fixed_row.append(class_modes[label][i])
            else:
                fixed_row.append(v)
        cleaned_data.append(fixed_row)

    return cleaned_data


# 3. Naive Bayes Classifier Implementation 
class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.conditional_probs = {}

    def fit(self, data):
        """
        Trains the Naive Bayes model on categorical data and it
        uses frequency counts with Laplace smoothing to handle unseen values.
        """
        total_samples = len(data)
        label_counts = Counter([row[0] for row in data])
        self.class_priors = {cls: count / total_samples for cls, count in label_counts.items()}
        self.conditional_probs = {cls: [defaultdict(float) for _ in range(len(data[0]) - 1)]
                                  for cls in label_counts.keys()}

        # Compute probabilities per class per attribute
        for cls in label_counts.keys():
            class_rows = [r for r in data if r[0] == cls]
            for col_idx in range(1, len(data[0])):
                col_values = [r[col_idx] for r in class_rows]
                value_counts = Counter(col_values)
                total = len(col_values)
                unique_values = len(set(col_values))
                for value in value_counts:
                    self.conditional_probs[cls][col_idx - 1][value] = \
                        (value_counts[value] + 1) / (total + unique_values)

    def predict(self, row):
      # Predicts the class of a given record based on computed probabilities.
     
        posteriors = {}
        for cls in self.class_priors:
            prob = self.class_priors[cls]
            for i, attr in enumerate(row[1:]):
                prob *= self.conditional_probs[cls][i].get(attr, 1e-6)  
            posteriors[cls] = prob
        return max(posteriors, key=posteriors.get)

    def score(self, test_data):

        # Evaluates model accuracy on test data.
        correct = 0
        for row in test_data:
            if self.predict(row) == row[0]:
                correct += 1
        return correct / len(test_data)


# 4. 10-Fold Cross-Validation
def cross_validation(data, k=NUM_FOLDS):
    """
    This function performs k-fold cross-validation.
    If dataset size isn't perfectly divisible by k,
    extra samples are distributed among the first few folds.
    """
    random.shuffle(data)
    n = len(data)
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1

    folds, start = [], 0
    for size in fold_sizes:
        folds.append(data[start:start + size])
        start += size

    accuracies = []
    for i in range(k):
        test = folds[i]
        train = [row for j, fold in enumerate(folds) if j != i for row in fold]
        model = NaiveBayesClassifier()
        model.fit(train)
        acc = model.score(test)
        accuracies.append(acc)
        print(f"Fold {i + 1} Accuracy: {acc:.3f}")

    mean_accuracy = sum(accuracies) / k
    print(f"\nAverage Accuracy over {k} folds: {mean_accuracy:.3f}")
    return accuracies, mean_accuracy


# 5. Main Execution
if __name__ == "__main__":
    raw_data = load_dataset(DATA_PATH)
    print(f"Loaded {len(raw_data)} records from dataset.")
    cleaned_data = handle_missing_values(raw_data)
    print("Missing values handled using class-wise mode replacement.\n")
    accuracies, mean_acc = cross_validation(cleaned_data, k=10)
