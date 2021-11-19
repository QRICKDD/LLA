# -- coding: utf-8 --
import utils.Sincnet as Sincnet
import numpy as np
import torch
import os
import soundfile as sf

class HSJA(object):
    def __init__(self,MODE,SAVE_DIR_PATH, num_iterations=2000, gamma=1.0, stepsize_search='geometric_progression',
                 max_num_evals=1e4, init_num_evals=100,query_limit=25000, verbose=True):
        self.model = Sincnet.get_speaker_model(MODE)
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.stepsize_search = stepsize_search
        self.max_num_evals = max_num_evals
        self.init_num_evals = init_num_evals
        self.verbose = verbose
        self.SAVE_DIR_PATH=SAVE_DIR_PATH
        self.query_num=0
        self.query_limit=query_limit

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
            # write_f_name=r"F:\SR-ATK\expH\fake_{}_{}.wav".format(self.query_num, np.linalg.norm(perturbed-input_xi))
            # sf.write(write_f_name, perturbed.squeeze(),16000)
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

    def __call__(self, input_xi, label_or_target, initial_xi=None, target=None):
        label_or_target = np.array([label_or_target])
        adv = self.hsja(input_xi, label_or_target, initial_xi)
        return adv.squeeze()