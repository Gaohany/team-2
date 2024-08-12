import json
import os
import time
from datetime import datetime

import numpy as np
import pyflann
import torch
from torchvision.transforms import transforms

from fuzzer.fuzz_queue import FuzzQueue
from fuzzer.save_predictions import *


class TensorInputCorpus(FuzzQueue):
    """Class that holds inputs and associated coverage."""

    def __init__(self, outdir, random, sampling, threshold, algorithm):
        """
        returns initialized object of class
        :param outdir: test output directory
        :param random: whether random testing
        :param sampling: looks at the whole current corpus and samples the next element to mutate in loop.
        :param threshold: for nearest distance
        :param algorithm: algorithm to use when searching for nearest neighbours
        """
        FuzzQueue.__init__(self, outdir, random, sampling, 1, "Near")

        self.flann = pyflann.FLANN()
        self.threshold = threshold
        self.algorithm = algorithm
        self.corpus_buffer = []
        self.lookup_array = []
        self._BUFFER_SIZE = 50

    def is_interesting(self, seed):
        """
        Does seed improve coverage?
        :param seed:
        :return:
        """
        _, approx_distances = self.flann.nn_index((seed.coverage).numpy(), 1, algorithm=self.algorithm)
        exact_distances = [torch.sum(torch.square(seed.coverage - buffer_elt)) for buffer_elt in self.corpus_buffer]
        nearest_distance = min(exact_distances + approx_distances.tolist())
        return nearest_distance > self.threshold or self.random

    def build_index_and_flush_buffer(self):
        """Builds the nearest neighbor index and flushes buffer of examples.
        This method first empties the buffer of examples that have not yet been added to the nearest neighbor index.
        Then it rebuilds that index using the contents of the whole corpus.
        """
        print("Total %s Flushing buffer and building index." % len(self.corpus_buffer))
        self.corpus_buffer[:] = []
        self.lookup_array = np.array([np.array(element.coverage) for element in self.queue])
        self.flann.build_index(self.lookup_array, algorithm=self.algorithm)

    def save_if_interesting(self, seed, data, crash, dry_run=False, suffix=None, config=None):
        """
        Save the seed if it is a bug or increases the coverage.
        :param seed: seed object being considered
        :param data: stacked image seed (for reference and mutation)
        :param crash: function being called for a seed that failed
        :param dry_run: function being called for a dry run
        :param suffix: for filename
        :return: true
        """
        if len(self.corpus_buffer) >= self._BUFFER_SIZE or len(self.queue) == 1:
            self.build_index_and_flush_buffer()

        self.mutations_processed += 1
        current_time = time.time()
        if dry_run:
            self.dry_run_cov = 0
        if current_time - self.log_time > 5:
            self.log_time = current_time
            self.log()

        if seed.parent is None:
            describe_op = "src_%s" % suffix
        else:
            describe_op = "src_%06d_%s" % (seed.parent.id, "" if suffix is None else suffix)

        if crash:
            trans = transforms.ToPILImage()
            file_name = "%s/crashes/id_%06d_%s.png" % (self.out_dir, self.uniq_crashes, describe_op)
            trans(data).save(file_name, "PNG")
            self.uniq_crashes += 1
            self.last_crash_time = current_time
        else:
            file_name = "%s/queue/id_%06d_%s.pt" % (self.out_dir, self.total_queue, describe_op)
            if dry_run or self.is_interesting(seed):
                self.last_reg_time = current_time
                seed.queue_time = current_time
                seed.id = self.total_queue
                seed.file_name = file_name
                seed.probability = self.REG_INIT_PROB
                self.queue.append(seed)
                self.corpus_buffer.append(seed.coverage)
                self.total_queue += 1
                torch.save(data, file_name)
            else:
                del seed
                return False

        return True


class ImageInputCorpus(FuzzQueue):
    """Class that holds inputs and associated coverage."""

    def __init__(self, outdir, random, sampling, cov_num, criteria):
        """
        returns initialized object of the class
        :param outdir: test output directory
        :param random: whether random testing
        :param sampling: looks at the whole current corpus and samples the next element to mutate in loop.
        :param cov_num: the total number of items to keep the coverage for. (last paragraph in Section 3.3)
        :param criteria: fuzzing criteria
        """

        FuzzQueue.__init__(self, outdir, random, sampling, cov_num, criteria)

    def save_if_interesting(
        self,
        seed,
        data,
        crash,
        gt_label=None,
        dry_run=False,
        suffix=None,
        config=None,
        only_coverage=False,
        type_crash=None,
        print_logs: bool = True,
    ):
        """
        Save the seed if it is a bug or increases the coverage.
        :param seed: seed object being considered
        :param data: stacked image seed (for reference and mutation)
        :param crash: function being called for a seed that failed
        :param dry_run: function being called for a dry run
        :param suffix: for filename
        :param print_logs: whether to print any log messages
        :return:
        """
        self.mutations_processed += 1
        current_time = time.time()

        # compute the the initial coverage
        if dry_run and (self.criteria != "lscd"):
            self.dry_run_cov = self.compute_cov()
        elif dry_run and (self.criteria == "lscd"):
            self.dry_run_cov = seed.coverage
        else:
            pass

        if current_time - self.log_time > 5:
            self.log_time = current_time
            self.log(seed)

        if seed.parent is None:
            describe_op = "src_%s" % suffix
        else:
            describe_op = seed.root_seed

        if crash:
            trans = transforms.ToPILImage()

            # crashed image have a unique id followed by the name of image of root seed.
            if type_crash == "rec":
                file_name = "%s/crashes_rec/id_%06d_%s" % (self.out_dir, self.uniq_crashes, describe_op)
            elif type_crash == "f1":
                file_name = "%s/crashes_f1/id_%06d_%s" % (self.out_dir, self.uniq_crashes, describe_op)
            elif type_crash == "hyb":
                file_name = "%s/crashes_hyb/id_%06d_%s" % (self.out_dir, self.uniq_crashes, describe_op)
            else:
                file_name = "%s/crashes_rec/id_%06d_%s" % (self.out_dir, self.uniq_crashes, describe_op)

            file_name_png = file_name + ".png"
            file_name_json = file_name + ".json"

            average_precision_dict = {
                "op_class": seed.predictions["op_class"],
                "op_class_prob": seed.predictions["op_class_prob"],
                "op_probs": seed.predictions["op_probs"],
                "org_image_name": describe_op,
                "transformation_class": seed.transformation_class,
                "ssim_ref": seed.ssim_ref,
                "mse_ref": seed.mse_ref,
                "l0_ref": seed.l0_ref,
                "l2_ref": seed.l2_ref,
                "linf_ref": seed.linf_ref,
                "gt_classes": seed.gt_label_dict,
                "org_pred_class": seed.ground_truths["op_class"],
                "org_class_pred_prob": seed.ground_truths["op_class_prob"],
                "org_pred_probs": seed.ground_truths["op_probs"],
            }

            if self.criteria == "lscd":
                average_precision_dict.update(
                    {"org_dist": seed.euc_dist, "curr_dist": str(round(float(seed.coverage), 3))}
                )

            save_predictions(
                average_precision=average_precision_dict,
                file_name_json=file_name_json,
                height=32,
                width=32,
                dataset=config.data,
                criteria=self.criteria,
                print_logs=print_logs,
            )  # To see difficulty type and can be merged with other function.

            # To save proper image, we apply denormalization, as data is normalized image with channel-first format.
            mean = torch.tensor(getattr(config, "norm_mean_" + config.detection_model.image_set)).view(-1, 1, 1)
            std = torch.tensor(getattr(config, "norm_std_" + config.detection_model.image_set)).view(-1, 1, 1)
            data = (data * std) + mean

            trans(data).save(file_name_png, "PNG")
            self.uniq_crashes += 1
            self.last_crash_time = current_time

        else:
            file_name = "%s/queue/id_%06d_%s.pt" % (self.out_dir, self.total_queue, describe_op)
            # During dry_run process, we will keep all initial seeds.
            if self.has_new_bits(seed) or dry_run:
                self.last_reg_time = current_time
                seed.queue_time = current_time
                seed.id = self.total_queue
                seed.file_name = file_name
                seed.probability = self.REG_INIT_PROB
                self.queue.append(seed)
                del seed.coverage
                self.total_queue += 1
                if not only_coverage:
                    torch.save({"images": data, "seed_obj": seed}, file_name)
            else:
                del seed
                return False
        return True
