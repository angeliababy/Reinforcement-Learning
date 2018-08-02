from __future__ import print_function
import tensorflow as tf
import cv2
import sys
import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

# 游戏名
GAME = 'bird'
# 动作（跳、不跳（动))
ACTIONS = 2
# 系数
GAMMA = 0.99
# 观察期
OBSERVE = 10000
# 探索期
EXPLORE = 2000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
# 训练每批样本数
BATCH_SIZE = 32
FRAME_PER_ACTION = 1

# 网络权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

# 网络偏置项
def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

# 卷积
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = 'SAME')

# 池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],strides=[1, 2, 2, 1],padding = 'SAME')

'''
是一个cnn网络结构
作用是返回每次训练可选动作Q值列表
'''
def createNetwork():
    # 定义网络每一层的参数维度
    W_conv1 = weight_variable([8, 8, 4, 32])# 卷积核为8×8、输入维度为4、输出维度为32
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([4, 4, 32, 64])# 卷积核为4×4、输入维度为32、输出维度为64
    b_conv2 = bias_variable([64])
    W_conv3 = weight_variable([3, 3, 64, 64])# 卷积核为3×3、输入维度为64、输出维度为64
    b_conv3 = bias_variable([64])
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    # 前向传播
    state = tf.placeholder('float', [None, 80, 80 ,4]) # 输入state tensor(80*80灰度图，4帧)
    h_conv1 = tf.nn.relu(conv2d(state, W_conv1, 4) + b_conv1) # 卷积
    h_pool1 = max_pool_2x2(h_conv1) # 池化
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2) #卷积
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3) #卷积
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600]) # 展平
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1) # 全连接层
    Qs = tf.matmul(h_fc1, W_fc2) + b_fc2 # 输出层，输出可选动作q值
	#返回每次训练可选动作Q值列表
    return state, Qs

'''
调用游戏的API生成状态，作为最初curent_state的输入数据
'''
def get_game_state():
    game_state = game.GameState()
    a_file = open('logs_' + GAME + "/readout.txt", 'w')
    h_file = open('logs_' + GAME + "/hidden.txt", 'w')
    # 初始化
    # 将图像转化为80*80*4 的矩阵
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t,r_0,terminal = game_state.frame_step(do_nothing)
    # 将图像转换成80*80，并进行灰度化
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_RGBA2GRAY)
    # 对图像进行二值化
    ret,x_t=cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    # 将图像处理成4通道
    s_current=np.stack((x_t, x_t, x_t, x_t), axis = 2)
    return s_current, game_state

'''
生成样本，保存入经验池
'''
def data(D, game_state, a_t, s_current):
    # 其次，执行选择的动作，并保存返回的游戏状态、得分，游戏是否结束
    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)  # 通过当前程序自行决策出来的动作，计算出状态、得分
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_RGB2GRAY)  # 将图像转换成80*80，并进行灰度化
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)  # 图像二值化
    x_t1 = np.reshape(x_t1, (80, 80, 1))  # 改变形状
    # 加上前3帧，得到马尔科夫决策过程下一个状态s_next
    s_next = np.append(x_t1, s_current[:, :, :3], axis=2)
    # 如果经验池超过最大长度 则弹出最早的经验数据
    if len(D) > REPLAY_MEMORY:
        D.popleft()
        # 经验池保存的是以一个马尔科夫序列于D中
        # (s_current, a_t, r_t, s_next, terminal)分别表示
        # t时的状态s_current，
        # 执行的动作a_t，
        # 得到的反馈r_t，
        # 得到的下一步的状态s_current1
        # 游戏是否结束的标志terminal
    D.append((s_current, a_t, r_t, s_next, terminal))
    return D, s_next, r_t

'''
动作选择
'''
def action(Qs, curent_state, s_current, t, epsilon):
    Qs_current = Qs.eval(feed_dict={curent_state:[s_current]})[0]
    a_t = np.zeros([ACTIONS])
    action_index = 0
    if t % FRAME_PER_ACTION == 0:
        # 根据epsilon策略选择动作
        if random.random() <= epsilon:
            print('random actions')
            action_index = random.randrange(ACTIONS)
            a_t[random.randrange(ACTIONS)] = 1
        else: # 学习动作（Q值最大的那个）
            action_index = np.argmax(Qs_current)
            a_t[action_index] = 1
    else:
        a_t[0] = 1
    return a_t, action_index, Qs_current

'''
@功能：进行训练
使targetQ与evalQ不断接近
'''
def trainNetwork(curent_state, Qs, sess):
    # 动作tensor
    actions = tf.placeholder('float', [None, ACTIONS])
    # targetQ值tensor
    targetQ = tf.placeholder('float', [None])
    # loss
    evalQ = tf.reduce_sum(tf.multiply(Qs, actions), reduction_indices = 1) # 根据actions查询evalQ
    cost = tf.reduce_mean(tf.square(targetQ - evalQ)) # 定义损失函数
    global_step = tf.Variable(0, trainable = False)  # 训练步数
    loss = tf.train.AdamOptimizer(1e-6).minimize(cost,global_step = global_step) # 最小化损失函数

    # 最初当前状态，游戏状态, 经验池
    s_current, game_state = get_game_state()

    # 保存和载入网络
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state('saved_networks')
    # 检查checkpoint文件
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print('successed')
    else:
        print('could not find old network weights')

    # epsilon初始值
    epsilon = INITIAL_EPSILON
    # 经验池
    D = deque()
    t=0
    while 'flappy bird'!='angry bird':
        # ---------start:进行动作选择：当前环境输入到网络中得到evalQ值-----------
        a_t, action_index, Qs_current = action(Qs, curent_state, s_current, t, epsilon)
        # ---------end:进行动作选择：当前环境输入到网络中得到evalQ值-----------

        # epsilon衰减
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 保存样本至经验池
        D, s_next, r_t = data(D, game_state, a_t, s_current)

        # 待达到一定迭代次数，才开始训练网络参数
        if t > OBSERVE:
            minibatch=random.sample(D,BATCH_SIZE)
            # 从经验池D中随机提取马尔科夫序列
            s_current_batch=[d[0] for d in minibatch]
            a_batch=[d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_next_batch = [d[3] for d in minibatch]

            targetQ_batch=[]
            # 输入网络得到下一时刻的Q值
            Q_next_batch=Qs.eval(feed_dict={curent_state:s_next_batch})
            # 求targetQ
            for i in range(0,len(minibatch)):
                terminal=minibatch[i][4]
                # 求目标Q值，若游戏结束则只加上当前回报
                if terminal:
                    targetQ_batch.append(r_batch[i])
                else:
                    targetQ_batch.append(r_batch[i]+GAMMA*np.max(Q_next_batch[i]))
            # 训练
            _,step=sess.run([loss,global_step],feed_dict={
                targetQ:targetQ_batch,
                actions:a_batch,
                curent_state:s_current_batch
            })

            # 保存模型
            if step % 1000 == 0:
                saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = step)

        # 下一状态等于当前状态，便于循环
        s_current = s_next
        t += 1

        # 记录状态
        if t < OBSERVE:
            state = 'observe'
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = 'explore'
        else:
            state = 'train'

        print('TIMESTEP', t, '/STATE', state, '/EPSILON', epsilon, '/ACTION', action_index, '/REWARD', r_t, '/Q_MAX%e'%np.max(Qs_current))

def main():
    # 能让你在运行图的时候，插入一些计算图
    sess = tf.InteractiveSession()
    # DQN深度学习网络结构
    curent_state, Qs = createNetwork()
    # 训练网络全过程
    trainNetwork(curent_state, Qs ,sess)

if __name__ == '__main__':
    main()
