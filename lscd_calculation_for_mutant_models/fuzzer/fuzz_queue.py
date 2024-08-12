import datetime
import os
import random
import time
from random import randint

import numpy as np
import torch


class Seed(object):
    """Class representing a single element of a corpus."""

    def __init__(
        self,
        transformation_class,
        coverage,
        root_seed,
        parent,
        predictions,
        ground_truths,
        gt_label_dict=None,
        l0_ref=0,
        l2_ref=0,
        linf_ref=0,
        ssim_ref=1,
        mse_ref=0,
        euc_dist=0,
        num_classes=43,
    ):
        """Inits the object.

        Args:
          transformation_class: a transformation state to represent how the seed was transformed
                                0 - 6 are affine transformation, 7 - 8 are pixel value transformations
          coverage: a list to show the coverage(info for each neuron of each layer)
          root_seed: maintain the initial seed from which the current seed is sequentially mutated
          parent: seed on which mutation was done to get current seed
          prediction: the prediction result
          ground_truth: the ground truth of the current seed
          l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
          between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1})  in Equation 2
        Returns:
          Initialized object.
        """

        self.transformation_class = transformation_class
        self.predictions = predictions
        self.parent = parent
        self.root_seed = root_seed
        self.coverage = coverage
        self.gt_label_dict = gt_label_dict
        self.queue_time = None
        # time something was added to queue: first init seed is 0th
        self.id = None
        # The initial probability to select the current seed.
        self.probability = 0.8
        self.fuzzed_time = 0

        self.ground_truths = ground_truths

        self.l0_ref = l0_ref
        self.l2_ref = l2_ref
        self.linf_ref = linf_ref
        self.ssim_ref = ssim_ref
        self.mse_ref = mse_ref

        # For LSCD Calculations
        self.euc_dist = euc_dist

        if self.parent is None:
            # For equal seed selection criteria
            comb_dict = {}
            for i in range(num_classes):
                comb_dict[i] = 0

            self.samples_mutated_class = comb_dict


class FuzzQueue(object):
    """Class that holds inputs and associated coverage."""

    def __init__(self, outdir, random, sampling, cov_num, criteria):
        """

        :param outdir: test ouput dir
        :param random: whether it is random testing
        :param sampling: seed selection strategy
        :param cov_num: total size from coverage class
        :param criteria: fuzzing criteria
        """
        self.plot_file = open(os.path.join(outdir, "plot.log"), "a+")
        self.out_dir = outdir
        self.mutations_processed = 0
        self.queue = []
        self.sampling = sampling
        self.start_time = time.time()
        self.random = random
        self.criteria = criteria

        self.log_time = time.time()
        # Like AFL, it records the coverage of the seeds in the queue
        self.virgin_bits = torch.empty(cov_num, dtype=torch.uint8).fill_(255)

        self.uniq_crashes = 0
        self.total_queue = 0
        self.total_cov = cov_num

        # Some log information
        self.last_crash_time = self.start_time
        self.last_reg_time = self.start_time
        self.current_id = 0
        self.seed_attacked = set()
        self.seed_attacked_first_time = dict()

        self.dry_run_cov = None

        # REG_MIN and REG_GAMMA are the p_min and gamma in Equation 3
        self.REG_GAMMA = 5
        self.REG_MIN = 0.3
        self.REG_INIT_PROB = 0.8

    def has_new_bits(self, seed):
        """
        implementation for Line-9 in Algorithm1
        get coverage of virgin bits, find if mutated bits have coverage, if increased update the coverage
        :param seed:
        :return: bool
        """
        if self.criteria != "lscd":
            temp = np.invert(seed.coverage, dtype=np.uint8)
            cur = torch.bitwise_and(self.virgin_bits, temp)
            has_new = not torch.equal(cur, self.virgin_bits)
            if has_new:
                self.virgin_bits = cur
        else:
            prev_distance = seed.euc_dist
            curr_distance = seed.coverage
            increase_percentage = ((curr_distance - prev_distance) / prev_distance) * 100
            has_new = increase_percentage >= 10
        return has_new or self.random

    def plot_log(self, id, coverage):
        """
        Plot the data during fuzzing, include: the current time, current iteration, length of queue, initial coverage,
        total coverage, number of crashes, number of seeds that are attacked, number of mutations, mutation speed
        :param id:
        :return:
        """
        queue_len = len(self.queue)
        if self.criteria != "lscd":
            coverage = self.compute_cov()
        else:
            coverage = str(
                round(float(coverage.mean()), 3)
            )  # TO DO: Write current value for plot as well. Not the mean value.

        current_time = time.time()
        self.plot_file.write(
            "%d,%d,%d,%s,%s,%d,%d,%s,%s\n"
            % (
                time.time(),
                id,
                queue_len,
                self.dry_run_cov,
                coverage,
                self.uniq_crashes,
                len(self.seed_attacked),
                self.mutations_processed,
                round(float(self.mutations_processed) / (current_time - self.start_time), 2),
            )
        )
        self.plot_file.flush()

    def write_logs(self):
        log_file = open(os.path.join(self.out_dir, "fuzz.log"), "w+")
        for k in self.seed_attacked_first_time:
            log_file.write("%s:%s\n" % (k, self.seed_attacked_first_time[k]))
        log_file.close()
        self.plot_file.close()

    def log(self, seed):
        queue_len = len(self.queue)
        if self.criteria != "lscd":
            coverage = self.compute_cov()
        else:
            coverage = str(round(float(seed.coverage), 3))  # self.dry_run_cov
        current_time = time.time()
        print(
            "Metrics %s | corpus_size %s | crashes_size %s | mutations_per_second: %s | total_exces %s "
            "| last new reg: %s | last new adv %s | coverage: %s -> %s%%"
            % (
                self.criteria,
                queue_len,
                self.uniq_crashes,
                round(float(self.mutations_processed) / (current_time - self.start_time), 2),
                self.mutations_processed,
                datetime.timedelta(seconds=(time.time() - self.last_reg_time)),
                datetime.timedelta(seconds=(time.time() - self.last_crash_time)),
                self.dry_run_cov,
                coverage,
            )
        )

    def compute_cov(self):
        """Compute the current coverage in the queue"""
        coverage = round(float(self.total_cov - np.count_nonzero(self.virgin_bits == 0xFF)) * 100 / self.total_cov, 2)
        return str(coverage)

    def fuzzer_handler(self, iteration, cur_seed, bug_found, coverage_inc):
        """
        The handler after each iteration
        :param iteration: current iteration
        :param cur_seed: current seed
        :param bug_found: bool
        :param coverage_inc: bool
        :return:
        """

        if self.sampling == "deeptest" and not coverage_inc:
            # If deeptest cannot increase the coverage, it will pop the last seed from the queue
            self.queue.pop()

        elif self.sampling == "prob":
            # Update the probability based on the Equation 3 in the paper
            if cur_seed.probability > self.REG_MIN and cur_seed.fuzzed_time < self.REG_GAMMA * (1 - self.REG_MIN):
                cur_seed.probability = self.REG_INIT_PROB - float(cur_seed.fuzzed_time) / self.REG_GAMMA
            #  modification to get uniform dist. after n iterations.
            # TO DO:
            # if iteration>150:
            # randomly selecting a seed after certain number of iterations or based on coverage gain.
            # cur_seed.probability = .5
        if bug_found:
            # Log the initial seed from which we found the adversarial. It is for the statics of Table 6
            self.seed_attacked.add(cur_seed.root_seed)
            if not (cur_seed.parent in self.seed_attacked_first_time):
                # Log the information about when (which iteration) the initial seed is attacked successfully.
                self.seed_attacked_first_time[cur_seed.root_seed] = iteration

    def select_next(self):
        """Different seed selection strategies (See details in Section 4)"""
        if self.random == 1 or self.sampling == "uniform":
            return self.random_select()
        elif self.sampling == "tensorfuzz":
            return self.tensorfuzz()
        elif self.sampling == "deeptest":
            return self.deeptest_next()
        elif self.sampling == "prob":
            return self.prob_next()
        elif self.sampling == "equal_samples":
            return self.eq_next()

    def random_select(self):
        return random.choice(self.queue)

    def tensorfuzz(self):
        """Grabs new input from corpus according to sample_function."""
        corpus = self.queue
        reservoir = corpus[-5:] + [random.choice(corpus)]
        choice = random.choice(reservoir)
        return choice

    def deeptest_next(self):
        return self.queue[-1]

    def prob_next(self):
        """Grabs new input from corpus according to sample_function."""
        while True:
            if self.current_id == len(self.queue):
                self.current_id = 0

            cur_seed = self.queue[self.current_id]
            if randint(0, 100) < cur_seed.probability * 100:
                # Based on the probability, we decide whether to select the current seed.
                cur_seed.fuzzed_time += 1
                self.current_id += 1
                return cur_seed
            else:
                self.current_id += 1

    def eq_next(self, max_times=100):
        """Grabs new input based on how many samples from that class is mutated."""

        while True:
            if self.current_id == len(self.queue):
                self.current_id = 0

            cur_seed = self.queue[self.current_id]
            gt_class = int(cur_seed.gt_label_dict)

            if cur_seed.parent is not None:
                seeds_mutated_so_far = cur_seed.parent.samples_mutated_class[gt_class]
            else:
                seeds_mutated_so_far = cur_seed.samples_mutated_class[gt_class]

            if seeds_mutated_so_far <= max_times:
                if randint(0, 100) < cur_seed.probability * 100:
                    # Based on the probability, we decide whether to select the current seed.
                    cur_seed.fuzzed_time += 1
                    self.current_id += 1
                    if cur_seed.parent is not None:
                        cur_num_mutations_all = cur_seed.parent.samples_mutated_class
                        cur_num_mutations_all.update(
                            {cur_num_mutations_all[gt_class]: cur_num_mutations_all[gt_class] + 1}
                        )
                        cur_seed.samples_mutated_class = cur_num_mutations_all
                    else:
                        cur_seed.samples_mutated_class[gt_class] += 1
                    return cur_seed
            else:
                # If seed is mutated enough no. of times. Then we delete all seeds from that class in queue.
                self.queue = [seed for seed in self.queue if int(seed.gt_label_dict) != gt_class]
                self.current_id += 1
