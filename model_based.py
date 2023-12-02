import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from numpy import linalg as la

## 다시해봐야할 듯
def get_k(sigma,percentage):
    sigma_sqr=sigma**2
    sum_sigma_sqr=sum(sigma_sqr)
    k_sum_sigma=0
    k=0
    for i in sigma:
        k_sum_sigma+=i**2
        k+=1
        if k_sum_sigma>=sum_sigma_sqr*percentage:
            return k

def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

def singular_value_decomposition(table):
    ## https://lsjsj92.tistory.com/m/569
    ## https://maxtime1004.tistory.com/m/91
    # SVD = TruncatedSVD(n_components=100)
    # matrix = SVD.fit_transform(table)
    n = table.shape[1]
    user_rating = table[1]['a000012']
    print(user_rating)
    exit()
    u, sigma, vt = svd(table)
    print(u.shape)
    print(sigma.shape)
    print(vt.shape)
    k = get_k(sigma, 0.9)
    # Construct the diagonal matrix
    sigma_k = np.diag(sigma[:k])
    # Convert the original data to k-dimensional space (lower dimension)
    formed_items = np.around(np.dot(np.dot(u[:, :k], sigma_k), vt[:k, :]), decimals=3)
    for j in table.columns:
        user_rating = table['a000012', j]
        print(user_rating)
        # if user_rating == 0 or j == item: continue
        # # the similarity between item and item j
        # similarity = simMeas(formed_items[item, :].T, formed_items[j, :].T)
        # sim_total += similarity
        # # product of similarity and the rating of user to item j, then sum
        # rat_sim_total += similarity * user_rating
        # if sim_total == 0:
        #     return 0
        # else:
        #     return np.round(rat_sim_total / sim_total, decimals=3)
    exit()
    # corr = np.corrcoef(matrix)
    # visit_list = table.index
    ## visit id 값 넣는 건데 이건 나중에 userid 받고
    ## 거기서 유저가 방문한 곳 중에 만족도가 높은 곳만 추출해서 넣으면 됄 듯
    # coffey_hands = list(visit_list).index(1)
    # # visit 장소에 대해 상관계수가 0.9보다 높은 지역들만 출력
    # print(list(visit_list[(corr[coffey_hands]>=0.9)]))
    # exit()

    # 추후에 좀 많이 다듬어야할 것 같음
    # return 0

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
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # init biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):

            # rating이 존재하는 index를 기준으로 training
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j])
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
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - predicted[x, y], 2)
        return np.sqrt(cost) / len(xi)


    def gradient(self, error, i, j):
        """
        gradient of latent feature for GD

        :param error: rating - prediction error
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple
        """

        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq


    def gradient_descent(self, i, j, rating):
        """
        graident descent function

        :param i: user index of matrix
        :param j: item index of matrix
        :param rating: rating of (i,j)
        """

        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        # update latent feature
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq


    def get_prediction(self, i, j):
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)


    def get_complete_matrix(self):
        """
        computer complete matrix PXQ + P.bias + Q.bias + global bias

        - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)


    def print_results(self):
        """
        print fit results
        """

        print("User Latent P:")
        print(self._P)
        print("Item Latent Q:")
        print(self._Q.T)
        print("P x Q:")
        print(self._P.dot(self._Q.T))
        print("bias:")
        print(self._b)
        print("User Latent bias:")
        print(self._b_P)
        print("Item Latent bias:")
        print(self._b_Q)
        print("Final R matrix:")
        print(self.get_complete_matrix())
        print("Final RMSE:")
        print(self._training_process[self._epochs-1][1])


# run example
if __name__ == "__main__":
    # rating matrix - User X Item : (7 X 5)
    R = np.array([
        [1, 0, 0, 1, 3],
        [2, 0, 3, 1, 1],
        [1, 2, 0, 5, 0],
        [1, 0, 0, 4, 4],
        [2, 1, 5, 4, 0],
        [5, 1, 5, 4, 0],
        [0, 0, 0, 1, 0],
    ])

    # P, Q is (7 X k), (k X 5) matrix
