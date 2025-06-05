import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self):
        # Αρχικοποίηση βαρών και όρο μεροληψίας (b)
        self.w = None
        self.b = None

    def fit(self, x, y):
        # Έλεγχος ότι είναι numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)

        # Ελεγχος διαστάσεων
        if x.ndim != 2 or y.ndim != 1:
            raise ValueError("x πρέπει να είναι 2D και y πρέπει να είναι 1D array")

        # Ελεγχος ότι οι δύο πίνακες έχουν το ίδιο πλήθος γραμμών
        N, p = x.shape
        if y.shape[0] != N:
            raise ValueError("Το πλήθος γραμμών του x και το μέγεθος του y δεν ταιριάζουν")

        # Προσθήκη στήλης με 1 για τον όρο μεροληψίας-b (p+1)
        X_augmented = np.hstack([x, np.ones((N, 1))])

        # Κανονικές εξισώσεις: θ = (X^T*X)^-1*X^T*y

        XtX = np.dot(X_augmented.T, X_augmented)
        # Υπολογίζει το X^T * X

        XtX_inv = np.linalg.inv(XtX) # Υπολογίζει το (X^T*X)^-1
        # Υπολογίζει το αντίστροφο του X^T * X => (X^T * X)^-1

        Xty = np.dot(X_augmented.T, y)
        # Υπολογίζει το X^T * y

        theta = np.dot(XtX_inv, Xty)
        # Υπολογίζει τη λύση του συστήματος: θ = (X^T * X)^-1 * X^T * y

        # Χωρίζουμε τα αποτελέσματα
        self.w = theta[:-1]  # πρώτα p στοιχεία (8)
        self.b = theta[-1]   # τελευταίο στοιχείο (b)

    def predict(self, x):

        if self.w is None or self.b is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί ακόμα")

        x = np.asarray(x) # Μετατρέπει το x σε NumPy array, αν δεν είναι ήδη.

        if x.ndim != 2 or x.shape[1] != self.w.shape[0]:
            raise ValueError("Το x πρέπει να έχει τις ίδιες διαστάσεις με τα εκπαιδευτικά δεδομένα")

        return np.dot(x, self.w) + self.b # y_hat = Xw + b

    def evaluate(self, x, y):
        y_hat = self.predict(x)
        y = np.asarray(y) # Μετατρέπει το x σε NumPy array, αν δεν είναι ήδη.

        if y.shape[0] != y_hat.shape[0]:
            raise ValueError("Μη συμβατές διαστάσεις μεταξύ προβλέψεων και πραγματικών τιμών")

        # MSE = (1/N) * (ŷ - y)^T (ŷ - y)
        mse = np.mean((y_hat - y) ** 2)
        return y_hat, mse
