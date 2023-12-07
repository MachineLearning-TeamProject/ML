import numpy as np
import pandas as pd
from scipy.linalg import svd
import time

def singular_value_decomposition(table, user_id, n = 1000):
    print("Start SVD")
    start_time = time.time()

    ## https://lsjsj92.tistory.com/m/569
    ## https://maxtime1004.tistory.com/m/91
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
    # items as vectors of latent features
    U, Sigma, Vt = svd(table, full_matrices=True)
    Sigma_mat = np.diag(Sigma)

    # SVD reduces the dimensionality of the ratings matrix.
    result = np.round(np.dot(U[:, :n], np.dot(Sigma_mat[:n,:n], Vt[:n, :])), 1)
    result = pd.DataFrame(result, index=table.index, columns=table.columns).T[user_id]
    print("End SVD ", time.time() - start_time, " sec")
    return result

## https://yamalab.tistory.com/92
class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param learning_rate: alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs
        :param verbose: print status
        """
        self._R = np.array(R)
        self._columns = R.columns
        self._index = R.index
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose

        # init latent features
        self.user_latent = np.random.normal(size=(self._num_users, self._k)) #(2307, 3)
        self.item_latent = np.random.normal(size=(self._num_items, self._k)) #(2810, 3)

        # init biases
        self._bias_user = np.zeros(self._num_users)   # (2810,)
        self._bias_item = np.zeros(self._num_items)   # (2307,)
        self._bias = np.mean(self._R[np.where(self._R != 0)])

    def fit(self):
        print("Start Matrix Factorization")
        start_time = time.time()
        """
        training Matrix Factorization : Update matrix latent weight and bias

        self._b:
        - global bias: Use the average value of the rated rating in the input R as the global bias
        - Normalization.
        Instead of having a negative number in the final rating, allow the late feature to include a negative number.

        :return: training_process
        """
        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):
            self.gradient_descent(np.where(self._R > 0)[0],np.where(self._R>0)[1], self._R)
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))
        print("End Matrix Factorization ", round(time.time() - start_time, 3), " sec")

    def cost(self):
        """
        compute root mean square error
        :return: rmse cost
        """
        xi, yi = self._R.nonzero()
        predicted = self.get_complete_matrix()
        cost = np.sum(pow(self._R[xi, yi] - predicted[xi, yi], 2))
        return np.sqrt(cost) / len(xi)


    def gradient(self, error, i, j):
        """
        gradient of latent feature for GD

        :param error: rating - prediction error
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple
        """
        diff_user = (np.stack((error, error, error), axis=1) * self.item_latent[j, :]) - (self._reg_param * self.user_latent[i, :])
        diff_item = (np.stack((error, error, error), axis=1) * self.user_latent[i, :]) - (self._reg_param * self.item_latent[j, :])
        return diff_user, diff_item


    def gradient_descent(self, i, j, rating):
        """
        graident descent function

        :param i: user index of matrix
        :param j: item index of matrix
        :param rating: rating of (i,j)
        """
        # get error
        prediction = self.get_prediction(i, j)
        error = rating[i, j] - prediction

        # update biases
        temp_user = self._learning_rate * (error - self._reg_param * self._bias_user[i])
        temp_item = self._learning_rate * (error - self._reg_param * self._bias_item[j])
        dp, dq = self.gradient(error, i, j)

        for num, idx in enumerate(zip(i, j)):
            self._bias_user[idx[0]] += temp_user[num]
            self._bias_item[idx[1]] += temp_item[num]

            # update latent feature
            self.user_latent[idx[0], :] += self._learning_rate * dp[num]
            self.item_latent[idx[1], :] += self._learning_rate * dq[num]

    def get_prediction(self, i, j):
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return self._bias + self._bias_user[i] + self._bias_item[j]+ self.user_latent.dot(self.item_latent.T)[i, j]

    def get_complete_matrix(self):
        """
        computer complete matrix PXQ + P.bias + Q.bias + global bias

        - PXQ Matrix
        - Add self._bias_user : adding bias to each user
        - Add self._bias_item : adding bias to each item(visit area)
        - Add self._bias : Adding bias to each element

        :return: complete matrix R^
        """
        return self._bias + self._bias_user[:, np.newaxis] + self._bias_item[np.newaxis:, ] + self.user_latent.dot(self.item_latent.T)

    def test(self, user_id):
        # Returns the result after performing Matrix Factorization.
        predicted = self.get_complete_matrix()
        predicted = np.round(predicted, 2)
        predicted = np.where(predicted > 16.5, 0.0, predicted)
        result = pd.DataFrame(predicted, index=self._index, columns=self._columns).T[user_id]
        return result

    def save_array(self):
        # Save each bias and latent matrix
        np.save('parameter/bias', self._bias)
        np.save('parameter/bias_user', self._bias_user)
        np.save('parameter/bias_item', self._bias_item)
        np.save('parameter/user_latent', self.user_latent)
        np.save('parameter/item_latent', self.item_latent)

    def load_array(self):
        # load the saved bias and late feature.
        self._bias = np.load('parameter/bias.npy')

        bias_user = np.load('parameter/bias_user.npy')
        self._bias_user[:bias_user.shape[0]] = bias_user

        bias_item = np.load('parameter/bias_item.npy')
        self._bias_item[:bias_item.shape[0]] = bias_item

        user_latent = np.load('parameter/user_latent.npy')
        self.user_latent[:user_latent.shape[0], :] = user_latent

        item_latent = np.load('parameter/item_latent.npy')
        self.item_latent[:item_latent.shape[0], :] = item_latent