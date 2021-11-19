import time
import numpy as np
from numpy import linalg as LA
import torch
from utils import Sincnet
import soundfile as sf
import os
start_learning_rate = 1.0
MAX_ITER = 1000




def quad_solver(Q, b):
    """
    Solve min_a  0.5*aQa + b^T a s.t. a>=0
    """
    K = Q.shape[0]
    alpha = np.zeros((K,))
    g = b
    Qdiag = np.diag(Q)
    for i in range(20000):
        delta = np.maximum(alpha - g / Qdiag, 0) - alpha
        idx = np.argmax(abs(delta))
        val = delta[idx]
        if abs(val) < 1e-7:
            break
        g = g + val * Q[:, idx]
        alpha[idx] += val
    return alpha


def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign == 0] = 1
    return y_sign


class OPT_attack_sign_SGD(object):
    def __init__(self, model,save_dir,k=80,file_name="record.txt",):
        self.save_file=file_name
        self.model = model
        self.k = k
        self.log = torch.ones(MAX_ITER, 2)
        self.save_dir=save_dir
    def predict_label(self,data):
        pred_real, pred_pro = Sincnet.sentence_test(self.model,
                                                    torch.from_numpy(data).float().cuda())
        return pred_real

    def get_log(self):
        return self.log

    def attack_untargeted(self, x0, y0, alpha=0.2, beta=0.001, iterations=300, query_limit=20000,
                          distortion=None, seed=None, svm=False, momentum=0.0, stopping=0.1):


        model = self.model
        y0 = y0
        query_count = 0
        ls_total = 0

        if self.predict_label(x0) != y0:
            print("Fail to classify the image. No need to attack.")
            return x0

        if seed is not None:
            np.random.seed(seed)

        # Calculate a good starting point.
        num_directions = 80
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        # 猜测扰动方向的做法
        for i in range(num_directions):
            print("i:",i)
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if self.predict_label(x0 + theta) != y0:
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:  # 如果当前的扰动最大值，比起之前的还要小，那么就更新
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)

        """
        上面是使用Lmax范数去限制
        如果经过上面几次试探没有遇到被分类不正确的就使用L2范数去限制

        """
        if g_theta == float('inf'):
            num_directions = 100
            best_theta, g_theta = None, float('inf')
            print("Searching for the initial direction on %d random directions: " % (num_directions))
            timestart = time.time()
            for i in range(num_directions):
                query_count += 1
                theta = np.random.randn(*x0.shape)
                if self.predict_label(x0 + theta) != y0:
                    initial_lbd = LA.norm(theta)
                    theta /= initial_lbd
                    lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                    query_count += count
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        print("--------> Found distortion %.4f" % g_theta)
        timeend = time.time()
        """
        如果还不成功--则视为失败
        """
        if g_theta == float('inf'):
            print("Couldn't find valid initial, failed")
            return x0
        print("==========> Found best distortion %.4f in %.4f seconds "
              "using %d queries" % (g_theta, timeend - timestart, query_count))
        """
        下面的是一已经成功扰动的
        如果扰动成功那么best_theta 和g_theta都会有正常值
        """
        self.log[0][0], self.log[0][1] = g_theta, query_count
        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        vg = np.zeros_like(xg)
        learning_rate = start_learning_rate
        prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            print(i)
            """
            下面是用于估计梯度的sign_grad对应了算法一
            """
            if svm == True:
                sign_gradient, grad_queries = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta)
            else:
                print("SIGN_GRAD_v1")
                sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)

            # if False:
            #     # Compare cosine distance with numerical gradient.
            #     gradient, _ = self.eval_grad(model, x0, y0, xg, initial_lbd=gg, tol=beta/500, h=0.01)
            #     print("    Numerical - Sign gradient cosine distance: ",scipy.spatial.distance.cosine(gradient.flatten(), sign_gradient.flatten()))

            # Line search
            ls_count = 0
            """
            xg=best_theta  扰动
            gg=g_theta     扰动比率\alpha
            vg = np.zeros_like(xg) 和扰动形状一样的玩意
            """
            min_theta = xg
            min_g2 = gg
            min_vg = vg
            for _ in range(15):
                """
                如果动量大于0 新扰动的更新就会带有动量
                """
                if momentum > 0:
                    #                     # Nesterov
                    #                     vg_prev = vg
                    #                     new_vg = momentum*vg - alpha*sign_gradient
                    #                     new_theta = xg + vg*(1 + momentum) - vg_prev*momentum
                    new_vg = momentum * vg - alpha * sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient

                new_theta /= LA.norm(new_theta)
                """
                下面这个函数为新扰动确定新的扰动比率
                如果攻击不成功new_g2 会返回inf
                """
                new_g2, count = self.fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd=min_g2, tol=beta / 500)
                ls_count += count
                # alpha是梯度更新的学习率
                alpha = alpha * 2
                """
                只有当新扰动比例小于当前最小比例时候才会更新
                更新的同时会更新扰动和扰动比
                """
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    """
                    new_vg的来源  new_vg是带动量的梯度更新量
                    new_vg = momentum*vg - alpha*sign_gradient
                    """
                    if momentum > 0:
                        min_vg = new_vg
                else:
                    break
            """
            这边只可能等于不可能大于 大于估计是瞎写的
            如果等于的话说明上面都没更新，也就是更新梯度后的扰动攻击都失败了
            失败了说明学习率太大了 ，要缩小 下面就是缩小学习率的过程
            """
            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    if momentum > 0:
                        #                         # Nesterov
                        #                         vg_prev = vg
                        #                         new_vg = momentum*vg - alpha*sign_gradient
                        #                         new_theta = xg + vg*(1 + momentum) - vg_prev*momentum
                        new_vg = momentum * vg - alpha * sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(
                        model, x0, y0, new_theta, initial_lbd=min_g2, tol=beta / 500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        if momentum > 0:
                            min_vg = new_vg
                        break
            """
            如果学习率太小了，移动就太小，
            所以要放大学习率，缩小扰动
            如果扰动被缩小得太多，没办法了 break

            """
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta * 0.1
                if (beta < 1e-8):
                    break
            """
            上面得代码
            xg=best_theta  扰动
            gg=g_theta     扰动比率alpha
            vg = np.zeros_like(xg) 和扰动形状一样的玩意用于记录带动量的梯度更新量

            min_theta = xg  最小扰动
            min_g2 = gg     最小扰动比率
            min_vg = vg     如果动量为0那么min_vg=zeros()  如果动量不是0 那么min_vg就是
            带动量的梯度更新量
            """
            xg, gg = min_theta, min_g2
            vg = min_vg

            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)

            if query_count > query_limit:
                break

            if (i + 1) % 10 == 0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i + 1, gg, query_count))
            self.log[i + 1][0], self.log[i + 1][1] = gg, query_count

        target = self.predict_label(x0 + gg * xg)
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target"
              " %d queries %d \nTime: %.4f seconds" % (gg, target, query_count, timeend - timestart))

        self.log[i + 1:, 0] = gg
        self.log[i + 1:, 1] = query_count
        # print(self.log)
        # print("Distortions: ", distortions)
        return x0 + gg * xg

    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k
        sign_grad = np.zeros(theta.shape)
        queries = 0
        ### USe orthogonal transform
        # dim = np.prod(sign_grad.shape)
        # H = np.random.randn(dim, K)
        # Q, R = qr(H, mode='economic')
        preds = []
        """
        随机扰动 计算梯度
        类似基础的梯度估计
        """
        for iii in range(K):
            #             # Code for reduced dimension gradient
            #             u = np.random.randn(N_d,N_d)
            #             u = u.repeat(D, axis=0).repeat(D, axis=1)
            #             u /= LA.norm(u)
            #             u = u.reshape([1,1,N,N])

            u = np.random.randn(*theta.shape)
            # u = Q[:,iii].reshape(sign_grad.shape)
            u /= LA.norm(u)

            sign = 1
            new_theta = theta + h * u
            new_theta /= LA.norm(new_theta)
            """
            如果攻击成功说明超出界限，所以梯度为负
            """
            # Targeted case.
            if (target is not None and
                    self.predict_label(x0 + initial_lbd * new_theta) == target):
                sign = -1

            # Untargeted case
            preds.append(
                self.predict_label(x0 + initial_lbd * new_theta))
            if (target is None and
                    self.predict_label(x0 + initial_lbd * new_theta) != y0):
                sign = -1
            queries += 1
            sign_grad += u * sign
        # 估计梯度求均值
        sign_grad /= K

        #         sign_grad_u = sign_grad/LA.norm(sign_grad)
        #         new_theta = theta + h*sign_grad_u
        #         new_theta /= LA.norm(new_theta)
        #         fxph, q1 = self.fine_grained_binary_search_local(self.model, x0, y0, new_theta, initial_lbd=initial_lbd, tol=h/500)
        #         delta = (fxph - initial_lbd)/h
        #         queries += q1
        #         sign_grad *= 0.5*delta

        return sign_grad, queries

    def sign_grad_v2(self, x0, y0, theta, initial_lbd, h=0.001, K=200):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        for _ in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)

            ss = -1
            new_theta = theta + h * u
            new_theta /= LA.norm(new_theta)
            if self.predict_label(x0 +initial_lbd * new_theta) == y0:
                ss = 1
            queries += 1
            sign_grad += sign(u) * ss
        sign_grad /= K
        return sign_grad, queries

    def sign_grad_svm(self, x0, y0, theta, initial_lbd, h=0.001, K=100, lr=5.0, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        # 计算维度总数
        # np.prod用于把一个numpy相乘
        dim = np.prod(theta.shape)
        X = np.zeros((dim, K))  # 用于记录
        for iii in range(K):
            # 随机生成新扰动并归一化
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)

            sign = 1
            # 为原扰动添加少许新扰动并归一化
            new_theta = theta + h * u
            new_theta /= LA.norm(new_theta)

            """
            如果新的扰动成功了说明 这个方向存在可以减少的梯度 就往这个方向减少
            """
            # Targeted case.
            if (target is not None and
                    self.predict_label(x0 + initial_lbd * new_theta) == target):
                sign = -1

            # Untargeted case
            if (target is None and
                    self.predict_label(x0 + initial_lbd * new_theta) != y0):
                sign = -1
            queries += 1
            X[:, iii] = sign * u.reshape((dim,))  # 添加到记录矩阵
        # X^T*X
        Q = X.transpose().dot(X)  # shape=(K,K)
        q = -1 * np.ones((K,))  # shape=(K,)
        # np.diag变成对角矩阵，对角为-1，其他未0
        G = np.diag(-1 * np.ones((K,)))
        h = np.zeros((K,))
        ### Use quad_qp solver
        # alpha = solve_qp(Q, q, G, h)
        ### Use coordinate descent solver written by myself, avoid non-positive definite cases
        alpha = quad_solver(Q, q)
        sign_grad = (X.dot(alpha)).reshape(theta.shape)

        return sign_grad, queries

    """
    theta是更新后的扰动   initial_ldb是先前计算出来的扰动比率
    """

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        """"
        如果更新后的扰动攻击失败
            更新上下界
            while 循环提升上界 直到上界超出20倍的时候，跳出循环 返回'inf'
        成功的话就不停的缩小下界直到成功
        """
        if self.predict_label(x0 + lbd * theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while self.predict_label(x0 + lbd_hi * theta) == y0:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while self.predict_label(x0 + lbd_lo * theta) != y0:
                lbd_lo = lbd_lo * 0.99
                nquery += 1
        """
        二分精确查找
        """
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self.predict_label(x0 +lbd_mid * theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    """
    这个函数是仅仅当原x0加上扰动后不等于 原y0时候才会进入这个函数
    就是说进来这个函数必然是已经不正确了 
    """

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        """
        之前最好：current_best

        如果当前的最大扰动值，比起 之前最好的 还要大
            乘上之前攻击成功的扰动
                -没攻击成功就返回inf 以及 查询次数 这里的inf也没啥用，后面也不会更新
                -若攻击成功：就把之前最好的 设置为lbd的上界
        如果当前扰动最大值比起 之前最好要小，那么就把当前扰动最大设置为 lbd上界
        """
        if initial_lbd > current_best:
            if self.predict_label(x0 + current_best * theta) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0
        """
        下界永远为0
        当上下界不够精确的时候进行二分搜索

        """
        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self.predict_label(x0 + lbd_mid * theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def eval_grad(self, model, x0, y0, theta, initial_lbd, tol=1e-5, h=0.001, sign=False):
        # print("Finding gradient")
        fx = initial_lbd  # evaluate function value at original point
        grad = np.zeros_like(theta)
        x = theta
        # iterate over all indexes in x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        queries = 0
        while not it.finished:
            # evaluate function at x+h
            ix = it.multi_index
            oldval = x[ix]
            x[ix] = oldval + h  # increment by h
            unit_x = x / LA.norm(x)
            if sign:
                if self.predict_label(x0 + initial_lbd * unit_x) == y0:
                    g = 1
                else:
                    g = -1
                q1 = 1
            else:
                fxph, q1 = self.fine_grained_binary_search_local(model, x0, y0, unit_x, initial_lbd=initial_lbd,
                                                                 tol=h / 500)
                g = (fxph - fx) / (h)

            queries += q1
            # x[ix] = oldval - h
            # fxmh, q2 = self.fine_grained_binary_search_local(model, x0, y0, x, initial_lbd = initial_lbd, tol=h/500)
            x[ix] = oldval  # restore

            # compute the partial derivative with centered formula
            grad[ix] = g
            it.iternext()  # step to next dimension

        # print("Found gradient")
        return grad, queries

    def attack_targeted(self, x0, y0, target,target_audio, alpha=0.2, beta=0.001, iterations=600, query_limit=25000,
                        distortion=None, seed=None, svm=False, stopping=0.1):

        model = self.model
        print("Targeted attack - Source: {0} and Target: {1}".format(y0, target))

        if (self.predict_label(x0) == target):
            print("Image already target. No need to attack.")
            return x0


        if seed is not None:
            np.random.seed(seed)

        num_samples = 80
        best_theta, g_theta = None, float('inf')
        query_count = 0
        ls_total = 0
        sample_count = 0
        print("Searching for the initial direction on %d samples: " % (num_samples))
        timestart = time.time()


        """
        根据提供的目标说话人代音频 找到扰动方向
        """
        theta = target_audio - x0
        initial_lbd = LA.norm(theta)
        theta /= initial_lbd
        lbd, count = self.fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd, g_theta)
        query_count += count
        if lbd < g_theta:
            best_theta, g_theta = theta, lbd
            print("--------> Found distortion " , g_theta)



        timeend = time.time()
        if g_theta == np.inf:
            return x0, float('inf')
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" %
              (g_theta, timeend - timestart, query_count))

        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        learning_rate = start_learning_rate
        #prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            if svm == True:
                sign_gradient, grad_queries = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta, target=target)
            else:
                sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta, target=target)


            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local_targeted(
                    model, x0, y0, target, new_theta, initial_lbd=min_g2, tol=beta / 500)
                ls_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.5
                    new_theta = xg - alpha * sign_gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local_targeted(
                        model, x0, y0, target, new_theta, initial_lbd=min_g2, tol=beta / 500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break


            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta * 0.1
                if (beta < 1e-8):
                    break

            xg, gg = min_theta, min_g2

            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)

            if query_count > query_limit:
                break

            with open(os.path.join(self.save_dir,self.save_file),'a+', encoding='utf-8') as f:
                f.write("{}\t{}\n".format(query_count, gg))
            if i % 5 == 0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i + 1, gg, query_count))
            #sf.write(r"F:\SR-ATK\exppath\fake_{}_{}.wav".format(query_count,np.linalg.norm(xg*gg)),x0+xg*gg,16000)



        adv_target = self.predict_label(x0 + gg * xg)
        if (adv_target == target):
            timeend = time.time()
            print("\nAdversarial Example Found Successfully: distortion %.4f target"
                  " %d queries %d LS queries %d \nTime: %.4f seconds" % (gg, target, query_count, ls_total, timeend - timestart))

            return x0 + gg * xg
        else:
            print("Failed to find targeted adversarial example.")
            return x0 + gg * xg

    def fine_grained_binary_search_local_targeted(self, model, x0, y0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if self.predict_label(x0+lbd * theta) != t:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while self.predict_label(x0 + lbd_hi * theta) != t:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 100:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while self.predict_label(x0 + lbd_lo * theta) == t:
                lbd_lo = lbd_lo * 0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self.predict_label(x0 + lbd_mid * theta) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

        #         temp_theta = np.abs(lbd_hi*theta)
        #         temp_theta = np.clip(temp_theta - 0.15, 0.0, None)
        #         loss = np.sum(np.square(temp_theta))
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, model, x0, y0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            if self.predict_label(x0 + current_best * theta) != t:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self.predict_label(x0 + lbd_mid * theta) != t:
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
        return lbd_hi, nquery

    def __call__(self, input_xi, label_or_target, target=None,target_datas=None,distortion=None, seed=None,
                 svm=False, query_limit=25000, momentum=0.3, stopping=0.01, TARGETED=False, epsilon=None):
        if target is not None:
            adv = self.attack_targeted(input_xi, label_or_target, target,target_datas, distortion=distortion,
                                       seed=seed, svm=svm, query_limit=query_limit, stopping=stopping)
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, distortion=distortion,
                                         seed=seed, svm=svm, query_limit=query_limit, momentum=momentum,
                                         stopping=stopping)
        return adv


