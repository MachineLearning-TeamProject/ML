import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from numpy import linalg as la

## 확인필요
# def get_k(sigma,percentage):
#     sigma_sqr=sigma**2
#     sum_sigma_sqr=sum(sigma_sqr)
#     k_sum_sigma=0
#     k=0
#     for i in sigma:
#         k_sum_sigma+=i**2
#         k+=1
#         if k_sum_sigma>=sum_sigma_sqr*percentage:
#             return k
#
# def ecludSim(inA,inB):
#     return 1.0/(1.0+la.norm(inA-inB))

def singular_value_decomposition(table):
    # hyperparameter n_components
    ## https://lsjsj92.tistory.com/m/569
    ## https://maxtime1004.tistory.com/m/91
    # SVD = TruncatedSVD(n_components=100)
    # matrix = SVD.fit_transform(table)
    # print(matrix)
    U, Sigma, Vt = svd(table)

    # U 행렬의 경우는 Sigma와 내적을 수행하므로 Sigma의 앞 2행에 대응되는 앞 2열만 추출
    U_ = U[:, :]
    print(Sigma)
    Sigma_ = np.diag(Sigma[:2])
    # V 전치 행렬의 경우는 앞 2행만 추출
    Vt_ = Vt[:2]
    print(U_.shape, Sigma_.shape, Vt_.shape)
    # U, Sigma, Vt의 내적을 수행하며, 다시 원본 행렬 복원
    print('U_ matrix:\n', np.round(U_, 3))
    print(Sigma_)
    print('V_ transpose matrix:\n', np.round(Vt_, 3))
    a_ = np.dot(np.dot(U_, Sigma_), Vt_)
    print('\n', np.round(a_, 3))

    print()
    exit()
    # k = get_k(sigma, 0.9)
    # # Construct the diagonal matrix
    # sigma_k = np.diag(sigma[:k])
    # # Convert the original data to k-dimensional space (lower dimension)
    # formed_items = np.around(np.dot(np.dot(u[:, :k], sigma_k), vt[:k, :]), decimals=3)
    # 방문지

    # for j in table.columns:
    #     user_rating = table[j]['a000012']
    #     if user_rating == 0 or j == item: continue
    #     # the similarity between item and item j
    #     similarity = simMeas(formed_items[item, :].T, formed_items[j, :].T)
    #     sim_total += similarity
    #     # product of similarity and the rating of user to item j, then sum
    #     rat_sim_total += similarity * user_rating
    #     if sim_total == 0:
    #         return 0
    #     else:
    #         return np.round(rat_sim_total / sim_total, decimals=3)
    # visit id 값 넣는 건데 이건 나중에 userid 받고
    # 거기서 유저가 방문한 곳 중에 만족도가 높은 곳만 추출해서 넣으면 됄 듯
    # coffey_hands = list(visit_list).index(5)
    # # visit 장소에 대해 상관계수가 0.9보다 높은 지역들만 출력
    # print(list(visit_list[(corr[coffey_hands]>=0.9)]))
    # exit()
    # 추후에 좀 많이 다듬어야할 것 같음
    # return 0

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
        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose


    def fit(self):
        """
        training Matrix Factorization : Update matrix latent weight and bias

        참고: self._b에 대한 설명
        - global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
        - 정규화 기능. 최종 rating에 음수가 들어가는 것 대신 latent feature에 음수가 포함되도록 해줌.

        :return: training_process
        """

        # init latent features
        self.user_latent = np.random.normal(size=(self._num_users, self._k)) #(2307, 3)
        self.item_latent = np.random.normal(size=(self._num_items, self._k)) #(2810, 3)

        # init biases
        self._bias_user = np.zeros(self._num_users)   # (2810,)
        self._bias_item = np.zeros(self._num_items)   # (2307,)
        self._bias = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):
            self.gradient_descent(np.where(self._R > 0)[0],np.where(self._R>0)[1], self._R)
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))

    def cost(self):
        """
        compute root mean square error
        :return: rmse cost
        """

        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        # 참고: http://codepractice.tistory.com/90
        xi, yi = self._R.nonzero()
        predicted = self.get_complete_matrix()
        cost = 0
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
        # return self._bias + self._bias_user[i] + self._bias_item[j] + self.user_latent[i, :].dot(self.item_latent[j, :].T)


    def get_complete_matrix(self):
        """
        computer complete matrix PXQ + P.bias + Q.bias + global bias

        - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._bias + self._bias_user[:, np.newaxis] + self._bias_item[np.newaxis:, ] + self.user_latent.dot(self.item_latent.T)


    def print_results(self):
        """
        print fit results
        """

        print("User Latent P:")
        print(self.user_latent)
        print("Item Latent Q:")
        print(self.item_latent.T)
        print("P x Q:")
        print(self.user_latent.dot(self.item_latent.T))
        print("bias:")
        print(self._bias)
        print("User Latent bias:")
        print(self._bias_user)
        print("Item Latent bias:")
        print(self._bias_item)
        print("Final R matrix:")
        print(self.get_complete_matrix())
        print("Final RMSE:")
        print(self._training_process[self._epochs-1][1])