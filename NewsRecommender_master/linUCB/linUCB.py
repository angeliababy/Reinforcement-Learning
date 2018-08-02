# linUCB算法
# dataset.txt数据集里面：第一列为推荐系统推荐的文章号，第二列为回报，第三列及之后都为文章特征
import numpy as np
import matplotlib.pyplot as plt

# 初始化
def init(numFeatures, numArms):
    # 初始化A，b，θ，p
    A = np.array([np.eye(numFeatures).tolist()]*numArms)
    b = np.zeros((numArms, numFeatures,1))

    theta = np.zeros((numArms, numFeatures,1))
    p = np.zeros(numArms)
    return A, b, theta, p

# 训练
def train(data_array):
    # CTR指数（点击率）
    ctr_list = []
    # 时间
    T = []

    # 可选的臂（根据数据）
    numArms = 10
    # 历史数据数
    trials = data_array.shape[0]
    # 臂特征数（根据数据）
    numFeatures = data_array.shape[1] - 2

    # 总回报
    total_payoff = 0
    count = 0

    # 初始化
    A, b, theta, p = init(numFeatures, numArms)

    for t in range(0, trials):
        # 每行数据
        row = data_array[t]
        # 第几个臂（数据第一列）
        arm = row[0] - 1
        # 回报（数据第二列）
        payoff = row[1]
        # 特征（数据第三列及之后）
        x_t = np.expand_dims(row[2:], axis=1)

        # alpha为探索程度，几种情况
        # Best Setting for Strategy 1
        alpha = 0

        # Best Setting for Strategy 2
        # i = 0.05
        # alpha = float(i) / np.sqrt(t + 1)

        # Best Setting for Strategy 3. Another best setting is i = 0.4
        # i = 0.1
        # alpha = float(i) / (t + 1)

        # 求每个臂的p
        for a in range(0, numArms):
            # 求逆
            A_inv = np.linalg.inv(A[a])
            # 相乘
            theta[a] = np.matmul(A_inv, b[a])
            # 求臂的p
            p[a] = np.matmul(theta[a].T, x_t) + alpha * np.sqrt(np.matmul(np.matmul(x_t.T, A_inv), x_t))

        # 更新Aat，bat
        A[arm] = A[arm] + np.matmul(x_t, x_t.T)
        b[arm] = b[arm] + payoff * x_t

        # 选择p最大的那个臂作为推荐
        best_predicted_arm = np.argmax(p)
        # 推荐的臂与所给的相同，将参与CTR点击率计算
        if best_predicted_arm == arm:
            total_payoff = total_payoff + payoff
            count = count + 1
            ctr_list.append(total_payoff / count)
            T.append(t + 1)

    # CTR趋势画图
    plt.xlabel("T")
    plt.ylabel("CTR")
    plt.plot(T, ctr_list)
    # 存入路径
    plt.savefig('../data/ctrVsT/Strategy1.png')

if __name__ == "__main__":
    # 获取数据
    data_array = np.loadtxt('../data/dataset.txt', dtype=int)
    # 训练
    train(data_array)