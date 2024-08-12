import os

import torch
import tqdm
from PIL import Image
from torchvision.transforms import transforms

from fuzzer.construct_initial_seeds import get_test_dataloader
from fuzzer.fuzz_queue import Seed


def dry_run(init_seed_dir, predict, queue, config, centroids=None):
    """
    This function works with the initial set of seeds (which never crash), predicting their result
    and computing the coverage provided by them. If the seed increases coverage then they are queued.
    Each queued seed will contain two images, i.e., the reference image and mutant.
    :param init_seed_dir: path to initial seeds directory
    :param predict: will give prediction for the seed (output for each layers + final class prediction)
    :param queue: queue of the seeds
    :return:
    """
    print("Initiating dry run for Classification Model ...")
    test_dataset = get_test_dataloader(config)
    # test_dataset = test_loader.dataset  # As the get_test_dataloader returns Dataloader not dataset.

    if not config.fuzz_entire_test_set:
        seed_idx = []
        seed_images = os.listdir(init_seed_dir)
        seed_images_names = [seed_images[i].split(".png")[0] for i in range(len(seed_images))]
        for i in range(len(test_dataset)):
            if test_dataset.image_ids[i] in seed_images_names:
                seed_idx.append(i)

        test_set_fuzz = torch.utils.data.Subset(test_dataset, seed_idx)

    else:
        print("Using entire test dataset for testing.")
        test_set_fuzz = test_dataset  # test_loader.dataset

    for idx, data in tqdm.tqdm(enumerate(test_set_fuzz)):  # len(test_set_fuzz)
        images, label = data
        index_from_dataset = test_set_fuzz.indices[idx]
        seed_images_name = test_set_fuzz.dataset.image_ids[index_from_dataset].split(".png")[0]
        # print("Attempting dry run with '%s'..." % seed_images_name)

        if config.fuzz_criteria == "lscd":
            # Coverage list is torch tensor of euclidean distance. To make consistancy.
            coverage_list, detections, output_dict, feature_vector = predict(
                data, centroids[int(label)]
            )  # test_set_fuzz.dataset[idx]
            coverage = round(float(coverage_list), 3)
        else:
            coverage_list, detections, output_dict, feature_vector = predict(data)
            coverage = coverage_list[0]
        # gt_label = test_set_fuzz.dataset.classes[idx]
        gt_label = label

        # if gt_label != detections:
        #     print("Error in dry run.")

        # seed = Image.open(os.path.join(test_set_fuzz[idx][3]), mode='r').convert('RGB')
        # seed = preprocess(seed, config)
        seed = test_set_fuzz[idx][0].unsqueeze(dim=0).to(device=config.device)
        seed_obj = Seed(
            transformation_class=config.mutation_criteria,
            coverage=coverage,
            root_seed=seed_images_name,
            parent=None,
            predictions=output_dict,
            ground_truths=output_dict,
            gt_label_dict=gt_label,
            l0_ref=seed.nelement(),
            euc_dist=coverage,
        )

        seed = torch.cat((seed, seed), dim=0)
        queue.save_if_interesting(
            seed=seed_obj,
            data=seed,
            crash=False,
            gt_label=gt_label,
            dry_run=True,
            suffix=seed_images_name,
            print_logs=False,
        )

    return None
