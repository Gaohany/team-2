import gc

import torch
import tqdm

from fuzzer.fuzz_queue import Seed
from mutation.augmix.mutators import mutator
from mutation.geometric.mutators import image_random_mutate


class Fuzzer(object):
    """Class representing the fuzzer itself."""

    def __init__(self, corpus, network, model, predict, num_mutants, plot=True):
        """
        For each seed, compute the coverage and check whether it is a "bug", i.e., adversarial example

        Args:
          corpus: An InputCorpus object.
          model: object of model to be tested
          predict: prediction function
          num_mutants: number of mutated images to try and generate
        Returns:
          Initialized object.
        """
        self.plot = plot
        self.queue = corpus
        self.predict = predict
        self.model = model
        self.num_mutants = num_mutants
        self.network = network

    def inspect_mutants(
        self,
        queue,
        root_seed,
        parent,
        mutation_coverage,
        mutated_data,
        mutated_predictions_dict,
        mutated_detections,
        config,
    ):
        """
        Check for each mutant and decide whether it will be queued
        :param queue: object of corpus containing seeds to be mutated and tested
        :param root_seed: initial seed from which this was generated
        :param parent: immediate parent seed from which this was mutated
        :param mutation_coverage: coverage provided by the mutated seeds
        :param mutated_data: info related to mutations of seed
        :param mutated_predictions_list: predictions of the mutated seeds
        :return:
        """
        coverage_increased = False
        bug_found = False

        for idx in range(len(mutation_coverage)):
            if config.mutation_criteria == "geometric":
                seeds, mutated_seeds, transformation_classes, l0_batches, linf_batches = mutated_data
                input = Seed(
                    transformation_class=transformation_classes[idx],
                    coverage=mutation_coverage[idx],
                    root_seed=root_seed,
                    parent=parent,
                    predictions=mutated_predictions_dict[idx],
                    ground_truths=parent.ground_truths,
                    gt_label_dict=parent.gt_label_dict,
                    l0_ref=l0_batches[idx],
                    linf_ref=linf_batches[idx],
                )
            else:
                (
                    seeds,
                    mutated_seeds,
                    transformation_classes,
                    l0_batches,
                    l2_batches,
                    linf_batches,
                    ssim_batches,
                    mse_batches,
                ) = mutated_data
                input = Seed(
                    transformation_class=transformation_classes[idx],
                    coverage=mutation_coverage[idx],
                    root_seed=root_seed,
                    parent=parent,
                    predictions=mutated_predictions_dict[idx],
                    ground_truths=parent.ground_truths,
                    gt_label_dict=parent.gt_label_dict,
                    l0_ref=l0_batches[idx],
                    l2_ref=l2_batches[idx],
                    linf_ref=linf_batches[idx],
                    ssim_ref=ssim_batches[idx],
                    mse_ref=mse_batches[idx],
                    euc_dist=parent.euc_dist,
                )

            crash_rec, crash_f1, crash_hyb = self.network.objective_function(input)

            if len(crash_rec) > 0:
                self.model.eval()
                for i in crash_rec:
                    queue.save_if_interesting(
                        seed=input,
                        data=mutated_seeds[idx],
                        crash=True,
                        gt_label=(parent.ground_truths, mutated_detections[idx]),
                        suffix=i,
                        config=config,
                        type_crash="rec",
                        print_logs=False,
                    )  # ground_truths = {'op_class', 'op_class_prob', 'op_probs'}, gt_label_dict is only gt_class.
                bug_found = True

            else:
                new_img = torch.cat((seeds[idx : idx + 1], mutated_seeds[idx : idx + 1]))
                saved = queue.save_if_interesting(
                    seed=input,
                    data=new_img,
                    crash=False,
                    gt_label=(parent.gt_label_dict),
                    config=config,
                    print_logs=False,
                )
                coverage_increased = coverage_increased or saved

        return bug_found, coverage_increased

    def loop(self, config: dict, centroids=None):
        """
        Fuzzes a machine learning model in a loop
        :param config: config dict
        :return: None
        """
        max_iterations = config.max_iterations
        mutation_criteria = config.mutation_criteria
        iteration = 0

        print("Starting fuzzing process ...")
        for iteration in tqdm.tqdm(range(max_iterations)):
            if len(self.queue.queue) < 1 or iteration >= max_iterations:
                break

            if iteration % 1 == 0:
                gc.collect()

            seed = self.queue.select_next()
            # Get a mutated batch for each input
            if mutation_criteria == "geometric":
                mutated_data_batch = image_random_mutate(seed, self.num_mutants, config.classA, config.classB, config)
            elif mutation_criteria == "augmix" or "transformations":
                mutated_data_batch = mutator(seed, self.num_mutants, config)
            else:
                raise NotImplementedError

            # Grab the coverage and predictions for mutated seeds
            if (
                config.data == "mnist"
                or config.data == "gtsrb"
                or config.data == "gtsrb-gray"
                or config.data == "svhn"
            ):
                mutated_images = mutated_data_batch[1].to(config.device)
                gt_classes = seed.gt_label_dict
                coverage_list, detections_list, predictions_dict_list = [], [], []
                for sample in range(self.num_mutants):
                    mutant_image = mutated_images[sample]
                    mutant_data = (mutant_image, gt_classes)
                    if config.fuzz_criteria == "lscd":
                        # Coverage list is torch tensor of euclidean distance. To make consistancy.
                        coverage, detections, output_dict, feature_vector = self.predict(
                            mutant_data, centroids[int(gt_classes)]
                        )  # test_set_fuzz.dataset[idx]
                        coverage = torch.tensor(coverage).reshape([1, 1])
                    else:
                        coverage, detections, output_dict, feature_vector = self.predict(
                            mutant_data
                        )  # (num_mutants,3,h,w)
                    # coverage, detections, output_dict = self.predict(mutant_data)
                    coverage_list.append(coverage)
                    detections_list.append(detections)
                    predictions_dict_list.append(output_dict)
                coverage = torch.stack(coverage_list, dim=0)
                coverage = torch.reshape(coverage, (coverage.shape[0], coverage.shape[2]))

            else:
                raise NotImplementedError("Extend predict method for this datatype & network.")

            # Plot the data
            if self.plot:
                self.queue.plot_log(iteration, coverage)

            if coverage is not None and len(coverage) > 0:
                bug_found, coverage_increased = self.inspect_mutants(
                    self.queue,
                    seed.root_seed,
                    seed,
                    coverage,
                    mutated_data_batch,
                    predictions_dict_list,
                    detections_list,
                    config,
                )
            else:
                bug_found = False
                coverage_increased = False

            self.queue.fuzzer_handler(iteration, seed, bug_found, coverage_increased)

            del mutated_data_batch
            del coverage
            del predictions_dict_list
            del detections_list
