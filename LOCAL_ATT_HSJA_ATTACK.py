# -*- coding: UTF-8 -*-
import utils.Sincnet as Sincnet
import utils.mylog as mylog
import torch
import numpy as np
import librosa
import soundfile as sf
import scipy.fftpack as ffp
import os
import copy


class LAATTACK(object):
    def __init__(self, o_audio_file, t_audio_file, save_p_fname,save_info_fname,MODE="TIMIE",dct_field=0.65,):
        self.o_audio_file = o_audio_file
        self.t_audio_file = t_audio_file
        self.model = Sincnet.get_speaker_model(MODE)
        self.model = self.model.eval()
        self.speaker_label, self.label_speaker = Sincnet.get_speaker_label(MODE)
        self.save_p_fname=save_p_fname
        self.save_info_fname=save_info_fname

        self.dct = lambda x: ffp.dct(x, norm='ortho')
        self.idct = lambda ix: ffp.idct(ix, norm='ortho')
        self.o_audio, self.sr = sf.read(o_audio_file)
        self.t_audio, self.sr = sf.read(t_audio_file)
        # 目标label以及原始label以及name
        self.o_label = self.predict_one_label(self.o_audio)
        self.t_label = self.predict_one_label(self.t_audio)
        self.o_name = self.get_name_by_label(self.o_label)
        self.t_name = self.get_name_by_label(self.t_label)
        # 初始化音频
        self.audio_len = min(len(self.o_audio), len(self.t_audio))
        self.o_audio = self.o_audio[:self.audio_len]
        self.t_audio = self.t_audio[:self.audio_len]
        self.o_audio /= np.linalg.norm(self.o_audio, np.inf)
        self.t_audio /= np.linalg.norm(self.t_audio, np.inf)
        # 初始化音频微缩
        self.o_audio *= 0.95
        self.t_audio *= 0.95
        self.o2_audio = None
        # 定义扰动部分变量
        self.best_pretub_scale = float('inf')
        self.best_pretub = None
        self.best_clip_scale = float('inf')
        self.best_clip_perturb = None
        self.interval = [None, None]
        self.clip_perturb_len = None
        # 超参
        self.query_num = 0
        self.theta = 1e-4  # 二分结束条件
        self.dct_field = dct_field
    # 记录初始化参数函数
    def save_init_para(self):
        with open(self.save_info_fname, 'a+', encoding='utf-8') as f:
            f.write("dct:{},o_file:{},t_file:{}".format(
                self.dct_field, self.o_audio_file, self.t_audio_file
            ))

    def predict_one_label(self, data):
        pred_real, pred_pro = Sincnet.sentence_test(self.model,
                                                    torch.from_numpy(data).float().cuda())
        return pred_real

    def save_pickle_file(self, query_num=None, best_pretub_scale=None, is_use=False):
        # 当is_use为True的时候启用
        if is_use == True:
            with open(self.save_p_fname, 'a+', encoding='utf-8') as f:
                f.write("{}\t{}\n".format(query_num, best_pretub_scale))
        else:
            with open(self.save_p_fname, 'a+', encoding='utf-8') as f:
                f.write("{}\t{}\n".format(self.query_num, self.best_pretub_scale))

    def get_label_by_name(self, name):
        return self.speaker_label[name.lower()]

    def get_name_by_label(self, label):
        return self.label_speaker[label]

    def clip_audio(self, audio):
        return np.clip(audio, -1, 1)

    # 普通域下的二分搜索  返回扰动的上界
    def binary_search_None(self, audio_a, t_label, pretub, now_pretub_scale, best_pretub_scale):
        query = 0
        # 如果当前扰动的压缩值比最小的扰动压缩值大，就尝试下可以通过不，如果可以的话就暂时性当作二分上界
        if now_pretub_scale > best_pretub_scale:
            if self.predict_one_label(audio_a + pretub * now_pretub_scale) != t_label:
                query += 1
                return float('inf')
            lbd = now_pretub_scale
        else:
            lbd = now_pretub_scale
        lbd_hi = lbd
        lbd_low = 0
        while (lbd_hi - lbd_low) > 0.001:
            lbd_mid = (lbd_hi + lbd_low) / 2
            if self.predict_one_label(audio_a + lbd_mid * pretub) != t_label:
                query += 1
                lbd_low = lbd_mid
            else:
                lbd_hi = lbd_mid
        return lbd_hi, query

    # 把部分t放到o上然后查询差异
    def move_match_clip(self):
        win_len=int(self.audio_len/10)
        all_query = 0
        orgin_audio = self.o_audio
        self.o2_audio = orgin_audio
        min_pretub_scale = float('inf')
        min_pretub = None
        min_interval = [0, 0]
        for item in np.linspace(0.9,0.35, 15):
            step_start = 0
            step_len = int(self.audio_len * item)
            is_change = False
            # 对前面部分计算
            while step_start + step_len <= self.audio_len:
                temp_o_audio = copy.deepcopy(orgin_audio)
                temp_t_audio = copy.deepcopy(self.t_audio)
                temp_o_audio[step_start:step_start + step_len] = temp_t_audio[step_start:step_start + step_len]
                if self.predict_one_label(temp_o_audio) == self.t_label:
                    all_query += 1
                    temp_pretub = temp_o_audio - orgin_audio
                    temp_pretub_scale = np.linalg.norm(temp_pretub)
                    temp_pretub /= temp_pretub_scale
                    good_pretub_scale, query = self.binary_search_None(orgin_audio, t_label=self.t_label,
                                                                       pretub=temp_pretub,
                                                                       now_pretub_scale=temp_pretub_scale,
                                                                       best_pretub_scale=min_pretub_scale)
                    all_query += query
                    # 更新扰动
                    if good_pretub_scale < min_pretub_scale:
                        is_change = True
                        min_pretub_scale = good_pretub_scale
                        min_pretub = temp_pretub
                        min_interval = [step_start, step_start + step_len]
                        ttt_label = self.predict_one_label(orgin_audio + good_pretub_scale * min_pretub)
                        #self.LOG.info("找到更小扰动{} 区间为{},label为{},查询次数为{}".format(min_pretub_scale, min_interval, ttt_label,all_query))
                        self.save_pickle_file(all_query, min_pretub_scale, is_use=True)
                step_start += win_len
            # 对最后一部分进行计算
            temp_o_audio = copy.deepcopy(orgin_audio)
            temp_t_audio = copy.deepcopy(self.t_audio)
            step_start = self.audio_len - step_len
            temp_o_audio[step_start:] = temp_t_audio[step_start:]
            if self.predict_one_label(temp_o_audio) == self.t_label:
                temp_pretub = temp_o_audio - orgin_audio
                temp_pretub_scale = np.linalg.norm(temp_pretub)
                temp_pretub /= temp_pretub_scale
                good_pretub_scale, query = self.binary_search_None(orgin_audio, t_label=self.t_label,
                                                                   pretub=temp_pretub,
                                                                   now_pretub_scale=temp_pretub_scale,
                                                                   best_pretub_scale=min_pretub_scale)
                all_query += query
                # 更新扰动
                if good_pretub_scale < min_pretub_scale:
                    is_change = True
                    min_pretub_scale = good_pretub_scale
                    min_pretub = temp_pretub
                    min_interval = [step_start, self.audio_len]
                    ttt_label = self.predict_one_label(orgin_audio + good_pretub_scale * min_pretub)
                    #self.LOG.info("找到更小扰动{} 区间为{},label为{},查询次数为{}".format(min_pretub_scale, min_interval, ttt_label, all_query))
                    self.save_pickle_file(all_query, min_pretub_scale, is_use=True)
            # 说明没更新了
            if is_change == False:
                #self.LOG.info("提前终止")
                break
        return min_pretub, min_pretub_scale, orgin_audio, min_interval, all_query

    def move_dct_reduce(self, o_audio, t_label, best_pretub, best_pretub_scale, w_len=150,
                        w_shift=150, init_step=1.0, limit_query=1500):
        start, end = 0, int(self.clip_perturb_len * self.dct_field)
        table = [0 for item in list(range(start, end, w_shift))]
        now_label = self.predict_one_label(o_audio + best_pretub_scale * best_pretub)
        #self.LOG.info("当前的label为{},t_label为{}".format(now_label, t_label))
        if now_label != t_label:
            return None, None
        # 记录最小减少 和最小值
        is_reduce = True
        min_pretub = self.dct(self.best_clip_perturb)  # 最小的扰动
        min_scale = best_pretub_scale
        all_query = 0

        step = w_len
        shift = w_shift
        compress_step = init_step
        while is_reduce:
            is_reduce = False
            start_index = start
            compress_step -= 0.05  # 每次都扩大步长
            if compress_step <= 0.51:
                break
            print("压缩量", compress_step)
            while start_index + step <= end:
                if table[(start_index - start) // w_shift] >= 4:
                    start_index += step
                    continue
                temp_pretub = copy.deepcopy(min_pretub)
                temp_pretub[start_index:min(end, start_index + step)] *= compress_step
                idct_temp_pretub = self.idct(temp_pretub)
                # 计算改变后的audio
                temp_audio2 = best_pretub_scale * best_pretub
                temp_audio2[self.interval[0]:self.interval[1]] = idct_temp_pretub  # 改变后的扰动
                temp_audio = o_audio + temp_audio2
                # 数量增加
                all_query += 1
                self.query_num += 1
                if self.predict_one_label(temp_audio) == t_label:
                    now_scale = np.linalg.norm(temp_audio2)
                    if now_scale < min_scale:
                        min_pretub = temp_pretub
                        min_scale = now_scale
                        #self.LOG.info("发现更小扰动{},查询次数为{}".format(min_scale, all_query))
                        self.save_pickle_file(self.query_num, min_scale, is_use=True)
                        is_reduce = True
                else:
                    table[(start_index - start) // w_shift] += 1
                start_index += shift
                # 超出迭代次数
                if all_query >= limit_query:
                    break

        min_clip_pretub = self.idct(min_pretub)  # 切分下降扰动结果返回时域
        r_b = best_pretub_scale * best_pretub
        r_b[self.interval[0]:self.interval[1]] = min_pretub
        min_scale = np.linalg.norm(r_b)
        r_b /= min_scale

        # 返回带有下标的table
        table = [[index * w_len, item] for index, item in enumerate(table)]
        return min_clip_pretub, r_b, min_scale, all_query, table

    def move_dct_reduce2(self, o_audio, t_label, table, best_pretub, best_pretub_scale, w_len=150,
                         w_shift=30, init_stap=0.95, reduce_min=0.01, taboo_num=3, limit_query=1000):
        taboo_table = []
        for index, [start_index, num] in enumerate(table):
            if num >= taboo_num and num != -1:
                start = start_index
                taboo_table.extend([[item, 0] for item in list(range(start, start + w_len, w_shift))])
        is_reduce = True
        min_dct_pretub = self.dct(self.best_clip_perturb)
        min_dct_pretub_scale = best_pretub_scale
        all_query = 0

        compress_step = init_stap
        while is_reduce:
            is_reduce = False
            compress_step -= 0.05
            #self.LOG.info("压缩量为{}".format(compress_step))
            if compress_step <= 0.51:
                break
            for index, [item1, item2] in enumerate(taboo_table):
                if item2 >= taboo_num or item2 == -1:
                    continue
                temp_pretub = copy.deepcopy(min_dct_pretub)
                temp_pretub[item1:item1 + w_shift] *= compress_step
                idct_temp_pretub = self.idct(temp_pretub)
                # 带入算原始音频
                temp_audio2 = best_pretub_scale * best_pretub
                temp_audio2[self.interval[0]:self.interval[1]] = idct_temp_pretub
                temp_audio = o_audio + temp_audio2
                # 数量增加
                all_query += 1
                self.query_num += 1
                if self.predict_one_label(temp_audio) == t_label:
                    now_scale = np.linalg.norm(temp_audio2)
                    if now_scale < min_dct_pretub_scale:

                        # 如果衰减的太少了，直接不要了
                        if abs(min_dct_pretub_scale - now_scale) < reduce_min:
                            taboo_table[index][1] = -1

                        min_dct_pretub = temp_pretub
                        min_dct_pretub_scale = now_scale
                        #self.LOG.info("发现更小扰动{},查询次数为{}".format(min_dct_pretub_scale, all_query))
                        self.save_pickle_file(self.query_num, min_dct_pretub_scale, is_use=True)
                        is_reduce = True
                else:
                    taboo_table[index][1] = -1
                # 超出迭代次数
                if all_query >= limit_query:
                    break
        min_clip_dct_pretub = self.idct(min_dct_pretub)
        min_dct_pretub = best_pretub * best_pretub_scale
        min_dct_pretub[self.interval[0]:self.interval[1]] = min_clip_dct_pretub
        min_dct_pretub_scale = np.linalg.norm(min_dct_pretub)
        min_dct_pretub /= min_dct_pretub_scale
        return min_clip_dct_pretub, min_dct_pretub, min_dct_pretub_scale, all_query, taboo_table

    # 二分查找局部最小值
    # 输入的扰动是归一化的
    def binary_search_local_None(self, o_audio, pretub, pretub_sacle):
        # 扰动误差量
        lbd = pretub_sacle
        # 不断的扩大扰动看看是否攻击成功
        if self.predict_one_label(o_audio + lbd * pretub) != self.t_label:
            lbd_lo = lbd
            lbd_hi = lbd * 1.1
            self.query_num += 1
            while self.predict_one_label(o_audio + lbd_hi * pretub) != self.t_label:
                lbd_hi = lbd_hi * 1.1
                self.query_num += 1
                if lbd_hi > 50.0:
                    return 99999
        # 攻击成功就尝试向缩小扰动
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.9
            self.query_num += 1
            while self.predict_one_label(o_audio + lbd_lo * pretub) == self.t_label:
                lbd_lo = lbd_lo * 0.9
                self.query_num += 1
        while (lbd_hi - lbd_lo) > self.theta:
            lbd_mid = (lbd_hi + lbd_lo) / 2.0
            self.query_num += 1
            if self.predict_one_label(o_audio + lbd_mid * pretub) == self.t_label:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi

    def targeted_attack(self):
        self.save_init_para()
        if self.t_label == self.o_label:
            #self.LOG.info("Audio already target.No need to attack")
            return self.o_audio, None
        # 记录最初扰动差距
        self.save_pickle_file(query_num=1, best_pretub_scale=np.linalg.norm(self.t_audio - self.o_audio), is_use=True)
        # 时域切分操作
        clip_pretub, clip_pretub_scale, orgin_audio, clip_interval, clip_query = self.move_match_clip()

        self.interval = clip_interval
        self.best_pretub = clip_pretub
        self.best_pretub_scale = clip_pretub_scale
        self.o2_audio = orgin_audio
        self.best_clip_perturb = (clip_pretub * clip_pretub_scale)[self.interval[0]:self.interval[1]]  # 注意这个值是没有归一化的
        self.best_clip_scale = np.linalg.norm(self.best_clip_perturb)
        self.clip_perturb_len = len(self.best_clip_perturb)
        self.query_num += clip_query
        with open(self.save_info_fname, 'a+', encoding='utf-8') as f:
            f.write("selected interval:{} to {}".format(self.interval[0],self.interval[1]))
        return self.o2_audio, self.best_pretub * self.best_pretub_scale, self.query_num, self.interval
"""
        # self.save_audio(self.best_pretub_scale*self.best_pretub,"time_clip_pretub.wav")
        # dct大域衰减 reduce_pretub是返回的时域扰动
        min_clip_preturb, reduce_pretub, reduce_pretub_scale, all_query, table = self.move_dct_reduce(self.o2_audio,
                                                                                                      self.t_label,
                                                                                                      self.best_pretub,
                                                                                                      self.best_pretub_scale,
                                                                                                      w_len=200,
                                                                                                      w_shift=200,
                                                                                                      init_step=0.95,
                                                                                                      limit_query=2000)
        self.best_clip_perturb = min_clip_preturb
        self.best_clip_scale = np.linalg.norm(min_clip_preturb)
        self.best_pretub = reduce_pretub
        #self.LOG.info("dct衰减第一阶段查询次数为{},扰动量从{}下降到{},下降量为{}".format(all_query, self.best_pretub_scale,reduce_pretub_scale,self.best_pretub_scale - reduce_pretub_scale))
        self.best_pretub_scale = reduce_pretub_scale
        # dct小域衰减第一次
        #self.LOG.info("进入dct局部衰减第二阶段")
        min_clip_preturb, reduce_pretub, reduce_pretub_scale, all_query, table = self.move_dct_reduce2(self.o2_audio,
                                                                                                       self.t_label,
                                                                                                       table,
                                                                                                       self.best_pretub,
                                                                                                       self.best_pretub_scale,
                                                                                                       w_len=200,
                                                                                                       w_shift=40,
                                                                                                       init_stap=0.95,
                                                                                                       taboo_num=3,
                                                                                                       reduce_min=0.005,
                                                                                                       limit_query=1500)
        self.best_clip_perturb = min_clip_preturb
        self.best_clip_scale = np.linalg.norm(min_clip_preturb)
        self.best_pretub = reduce_pretub
        #self.LOG.info("dct衰减第二阶段查询次数为{},扰动量从{}下降到{},下降量为{}".format(all_query, self.best_pretub_scale,reduce_pretub_scale,self.best_pretub_scale - reduce_pretub_scale))
        self.best_pretub_scale = reduce_pretub_scale
        return self.o2_audio, self.best_pretub * self.best_pretub_scale,self.query_num,self.interval
"""
class HSJA(object):
    def __init__(self,MODE,SAVE_DIR_PATH,query_num,interval, num_iterations=2000, gamma=1.0,
                 stepsize_search='geometric_progression',
                 max_num_evals=1e4, init_num_evals=100,query_limit=25000, verbose=True):
        self.model = Sincnet.get_speaker_model(MODE)
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.stepsize_search = stepsize_search
        self.max_num_evals = max_num_evals
        self.init_num_evals = init_num_evals
        self.verbose = verbose
        self.SAVE_DIR_PATH=SAVE_DIR_PATH
        self.query_num=query_num
        self.query_limit=query_limit
        self.interval=interval

    def hsja(self, input_xi, label_or_target, initial_xi):
        # Set parameters
        # original_label = np.argmax(self.model.predict_label(input_xi))
        d = int(np.prod(input_xi.shape))
        # Set binary search threshold.
        theta = self.gamma / (np.sqrt(d) * d)

        # Initialize.
        perturbed = initial_xi

        # Project the initialization to the boundary.
        perturbed, dist_post_update = self.binary_search_batch(input_xi, perturbed, label_or_target, theta)
        dist = self.compute_distance(perturbed, input_xi)

        for j in np.arange(self.num_iterations):
            # params['cur_iter'] = j + 1

            # Choose delta.
            if j == 1:
                delta = 0.2
            else:
                delta = np.sqrt(d) * theta * dist_post_update
            num_evals = int(self.init_num_evals * np.sqrt(j + 1))
            num_evals = int(min([num_evals, self.max_num_evals]))

            # approximate gradient.
            gradf = self.approximate_gradient(perturbed, label_or_target, num_evals,
                                              delta)
            update = gradf

            # search step size.
            if self.stepsize_search == 'geometric_progression':
                # find step size.
                epsilon = self.geometric_progression_for_stepsize(perturbed, label_or_target,
                                                                  update, dist, j + 1)

                # Update the sample.
                perturbed = self.clip_image(perturbed + epsilon * update)

                # Binary search to return to the boundary.
                perturbed, dist_post_update = self.binary_search_batch(input_xi,
                                                                       perturbed[None], label_or_target, theta)

            elif self.stepsize_search == 'grid_search':
                # Grid search for stepsize.
                epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
                epsilons_shape = [20] + len(input_xi.shape) * [1]
                perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                perturbeds = self.clip_image(perturbeds)
                idx_perturbed = self.decision_function(perturbeds, label_or_target)

                if np.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    perturbed, dist_post_update = self.binary_search_batch(input_xi,
                                                                           perturbeds[idx_perturbed], label_or_target,
                                                                           theta)

            # compute new distance.
            dist = self.compute_distance(perturbed, input_xi)
            if self.verbose:
                with open(self.SAVE_DIR_PATH, 'a+', encoding='utf-8') as f:
                    f.write("{}\t{}\n".format(self.query_num,dist ))
                print('iteration: {:d}, distance {:.4E}'.format(j + 1, dist))
            if self.query_num>self.query_limit:
                break

        return perturbed

    def predict_one_label(self,data):
        self.query_num+=1
        data=data.squeeze()
        pred_real, pred_pro = Sincnet.sentence_test(self.model,
                                                    torch.from_numpy(data).float().cuda())
        return pred_real
    def predict_more_label(self,datas):
        preds=[]
        for item in datas:
            preds.append(self.predict_one_label(item))
        return np.array(preds)
    def decision_function(self, images, label):
        """
        Decision function output 1 on the desired side of the boundary,
        0 otherwise.
        """
        la=self.predict_more_label(images)
        la=np.array(la)
        return (la == label)


    def clip_image(self, image, clip_min=-1, clip_max=1):
        # Clip an image, or an image batch, with upper and lower threshold.
        return np.minimum(np.maximum(clip_min, image), clip_max)

    def compute_distance(self, x_ori, x_pert):
        # Compute the distance between two images.
        return np.linalg.norm(x_ori - x_pert)


    def approximate_gradient(self, sample, label_or_target, num_evals, delta):

        # Generate random vectors.
        noise_shape = [num_evals] + list(sample.shape)
        rv = np.random.randn(*noise_shape)
        rv[:,:,:self.interval[0]]=0
        rv[:, :, self.interval[1]:] = 0

        rv = rv / np.sqrt(np.sum(rv ** 2, axis=(1, 2), keepdims=True))
        perturbed = sample + delta * rv
        perturbed = self.clip_image(perturbed)
        rv = (perturbed - sample) / delta

        # query the model.
        decisions = self.decision_function(perturbed, label_or_target)
        decision_shape = [len(decisions)] + [1] * len(sample.shape)
        fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        if np.mean(fval) == 1.0:  # label changes.
            gradf = np.mean(rv, axis=0)
        elif np.mean(fval) == -1.0:  # label not change.
            gradf = - np.mean(rv, axis=0)
        else:
            fval -= np.mean(fval)
            gradf = np.mean(fval * rv, axis=0)

            # Get the gradient direction.
        gradf = gradf / np.linalg.norm(gradf)

        return gradf

    def project(self, original_image, perturbed_images, alphas):
        alphas_shape = [1] * len(original_image.shape)
        alphas = alphas.reshape(alphas_shape)
        return (1 - alphas) * original_image + alphas * perturbed_images


    def binary_search_batch(self, original_image, perturbed_images, label_or_target, theta):
        """ Binary search to approach the boundar. """

        # Compute distance between each of perturbed image and original image.
        dists_post_update = np.array([
            self.compute_distance(
                original_image,
                perturbed_image
            )
            for perturbed_image in perturbed_images])
        # print(dists_post_update)
        # Choose upper thresholds in binary searchs based on constraint.
        highs = np.ones(len(perturbed_images))
        thresholds = theta

        lows = np.zeros(len(perturbed_images))

        # Call recursive function.
        while np.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_images = self.project(original_image, perturbed_images, mids)
            #         print(mid_images.shape)
            # Update highs and lows based on model decisions.
            decisions = self.decision_function(mid_images, label_or_target)
            lows = np.where(decisions == 0, mids, lows)
            highs = np.where(decisions == 1, mids, highs)

        out_images = self.project(original_image, perturbed_images, highs)

        # Compute distance of the output image to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = np.array([
            self.compute_distance(
                original_image,
                out_image
            )
            for out_image in out_images])
        idx = np.argmin(dists)

        dist = dists_post_update[idx]
        out_image = out_images[idx]
        return out_image, dist


    def geometric_progression_for_stepsize(self, x, label_or_target, update, dist, j):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        epsilon = dist / np.sqrt(j)

        def phi(epsilon):
            new = x + epsilon * update
            success = self.decision_function(new, label_or_target)
            return success

        while not phi(epsilon):
            epsilon /= 2.0

        return epsilon

    def __call__(self, input_xi, label_or_target, initial_xi):
        label_or_target = np.array([label_or_target])
        adv = self.hsja(input_xi, label_or_target, initial_xi)
        return adv.squeeze()




