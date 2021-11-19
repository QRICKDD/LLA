import time
import numpy as np
from numpy import linalg as LA
import random
from utils import Sincnet
import torch

class OPT_attack_lf(object):
    def __init__(self, MODE,savepath,limit_count=25000):
        self.querynum=0
        self.model = Sincnet.get_speaker_model(MODE)
        self.savepath=savepath
        self.limit_count=limit_count
    #预测单个
    def predict_one_label(self, data):
        data=data.squeeze()
        pred_real, pred_pro = Sincnet.sentence_test(self.model,
                                                   torch.from_numpy(data).float().cuda())

        return pred_real
    #批量预测[1,121,0,34,121.....]
    def predict_more_label(self,datas):
        preds=[]
        for item in datas:
            preds.append(self.predict_one_label(item))
        return preds

    def attack_targeted(self, initial_xi, x0, target, alpha=0.2, beta=0.001, iterations=5000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model

        num_samples = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        sample_count = 0
        # print("Searching for the initial direction on %d samples: " % (num_samples))
        timestart = time.time()

        xi = initial_xi
        # initial residual
        theta = xi - x0
        # initial lmax
        initial_lbd = LA.norm(theta.flatten())
        with open(self.savepath, 'a+', encoding='utf-8') as f:
            f.write("{}\t{}\n".format(query_count, LA.norm(theta.flatten())))
        # normlized with theta
        theta /= initial_lbd  # might have problem on the defination of direction
        # loss, query, lbd high
        lbd, count, lbd_g2 = self.fine_grained_binary_search_local_targeted(x0, target, theta)
        query_count += count
        if lbd < g_theta:
            best_theta, g_theta = theta, lbd
            print("--------> Found distortion %.4f" % g_theta)
        timeend = time.time()
        if g_theta == np.inf:
            return "NA", float('inf'), 0
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (
        g_theta, timeend - timestart, query_count))
        with open(self.savepath, 'a+', encoding='utf-8') as f:
            f.write("{}\t{}\n".format(query_count, np.linalg.norm((best_theta*g_theta).flatten())))

        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        # *replace* opt_count = 0
        opt_count = query_count
        stopping = 1e-8
        prev_obj = 1000000
        for i in range(iterations):
            if g2 == 0.0:
                break
            gradient = np.zeros(theta.shape)
            q = 20
            min_g1 = float('inf')
            min_lbd = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= LA.norm(u.flatten())
                ttt = theta + beta * u
                ttt /= LA.norm(ttt.flatten())
                g1, count, lbd_hi = self.fine_grained_binary_search_local_targeted(x0, target, ttt,
                                                                                   initial_lbd=lbd_g2, tol=beta / 500)
                # g1, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt)
                opt_count += count
                gradient += (g1 - g2) / beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
                    min_lbd_1 = lbd_hi
            gradient = 1.0 / q * gradient

            if i % 1 == 0:
                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f "
                      "distortion %.4f num_queries %d" % (
                i + 1, g1, g2, LA.norm((lbd_g2 * theta).flatten()), opt_count))
                # reach target with tiny perturbation
                if g2 > prev_obj - stopping:
                    print("stopping")
                    break
                prev_obj = g2



            min_theta = theta
            min_g2 = g2
            min_lbd = lbd_g2

            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= LA.norm(new_theta.flatten())
                new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(x0, target, new_theta,
                                                                                       initial_lbd=min_lbd,
                                                                                       tol=beta / 500)
                # new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    min_lbd = lbd_hi
                else:
                    break

            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta /= LA.norm(new_theta.flatten())
                    new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted( x0, target,
                                                                                           new_theta,
                                                                                           initial_lbd=min_lbd,
                                                                                           tol=beta / 500)
                    # new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta
                        min_g2 = new_g2
                        min_lbd = lbd_hi
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
                lbd_g2 = min_lbd
            else:
                theta, g2 = min_ttt, min_g1
                lbd_g2 = min_lbd_1
            if g2 < g_theta:
                best_theta, g_theta = theta, g2
                with open(self.savepath, 'a+', encoding='utf-8') as f:
                    f.write("{}\t{}\n".format(opt_count, np.linalg.norm((best_theta * g_theta).flatten())))
                # lbd_g2 = min_lbd
            # print(alpha)
            if alpha < 1e-6:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 1e-8):
                    break
            if opt_count>self.limit_count:
                break
        g_theta, _ = self.fine_grained_binary_search_local_targeted_original(x0, target, best_theta,
                                                                             initial_lbd=1.0, tol=beta / 500)
        dis = LA.norm((g_theta * best_theta).flatten(),ord=np.inf)
        target = self.predict_one_label(x0 + g_theta * best_theta)
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (
        dis, target, query_count + opt_count, timeend - timestart))
        return x0 + g_theta * best_theta

    def fine_grained_binary_search_local_targeted(self, x0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if self.predict_one_label(x0 + lbd * theta) != t:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while self.predict_one_label(x0 + lbd_hi * theta) != t:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 100:
                    return float('inf'), nquery, 1.0
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while self.predict_one_label(x0 + lbd_lo * theta) == t:
                lbd_lo = lbd_lo * 0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self.predict_one_label(x0 + lbd_mid * theta) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        temp_theta = np.abs(lbd_hi * theta)
        temp_theta = np.clip(temp_theta - 0.15, 0.0, None)
        loss = np.sum(np.square(temp_theta))
        # print(lbd_hi)
        return loss, nquery, lbd_hi

    def fine_grained_binary_search_local_targeted_original(self,x0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if self.predict_one_label(x0 + lbd * theta) != t:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while self.predict_one_label(x0 + lbd_hi * theta) != t:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 100:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while self.predict_one_label(x0 + lbd_lo * theta) == t:
                lbd_lo = lbd_lo * 0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self.predict_one_label(x0 + lbd_mid * theta) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, x0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            if self.predict_one_label(x0 + current_best * theta) != t:
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
            if self.predict_one_label(x0 + lbd_mid * theta) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def __call__(self, input_xi, target, initial_xi):
        #attack_targeted(self, initial_xi, x0, target, alpha=0.2, beta=0.001, iterations=5000):
        adv = self.attack_targeted(initial_xi, input_xi, target)
        return adv 