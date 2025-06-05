from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression as myLR
import numpy as np

# 3.1 Προετοιμασία δεδομένων
print("3.1 Προετοιμασία δεδομένων")

housing = fetch_california_housing()
x, y = housing.data, housing.target

# Διαχωρισμός σε train/test (70% train και 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Δημιουργία του μοντέλου
model = myLR()

# Εκπαίδευση στο training set
model.fit(x_train, y_train)

# Αξιολόγηση σε training set
y_pred_train, mse_train = model.evaluate(x_train, y_train)
rmse_train = np.sqrt(mse_train)

# Αξιολόγηση σε test set
y_pred_test, mse_test = model.evaluate(x_test, y_test)
rmse_test = np.sqrt(mse_test)

print(f"Train RMSE: {rmse_train:.4f}")
print(f"Test RMSE: {rmse_test:.4f}")

# 3.2 Μέση τιμή και διασπορά σφάλματος
print("-"*50,"\n3.2 Μέση τιμή και διασπορά σφάλματος")
rmse_scores = []

for i in range(20):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    # διαφορετικά σύνολα εκπαίδευσης = i

    # Εκπαίδευση μοντέλου
    model = myLR()
    model.fit(x_train, y_train)

    # Αξιολόγηση στο test set
    _, mse_test = model.evaluate(x_test, y_test)
    rmse_test = np.sqrt(mse_test)

    # Αποθήκευση της RMSE
    rmse_scores.append(rmse_test)

# Υπολογισμός μέσου όρου rmse και τυπικής απόκλισης rmse
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print("\nΜετά από 20 επαναλήψεις:")
print(f"Μέση τιμή RMSE: {mean_rmse:.4f}")
print(f"Τυπική απόκλιση RMSE: {std_rmse:.4f}")

# 3.3 Έλεγχος έναντι των αντίστοιχων συναρτήσεων του scikit-learn
print("-"*50,"\n3.3 Έλεγχος έναντι των αντίστοιχων συναρτήσεων του scikit-learn")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

rmse_list = []

# 20 επαναλήψεις με διαφορετικό random_state
for i in range(20):
    # Διαχωρισμός σε train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)

    # Δημιουργία και εκπαίδευση μοντέλου scikit-learn
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Πρόβλεψη και υπολογισμός RMSE στο test set
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmse_list.append(rmse)

# Υπολογισμός μέσου όρου και τυπικής απόκλισης
mean_rmse = np.mean(rmse_list)
std_rmse = np.std(rmse_list)

print("\nΑποτελέσματα μετά από 20 επαναλήψεις με την κλάση LinearRegression του scikit-learn:")
print(f"Μέση τιμή RMSE: {mean_rmse:.4f}")
print(f"Τυπική απόκλιση RMSE: {std_rmse:.4f}")
