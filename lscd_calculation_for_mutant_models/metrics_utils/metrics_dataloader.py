from dataset.corner_case_dataset import CornerCaseDataset
from dataset.GTSRB.gtsrb_dataset import GTSRBDataset
from dataset.GTSRB.gtsrb_dataset_gray import GTSRBDataset_gray
from dataset.MNIST.mnist_dataset import MNISTDataset
from dataset.SVHN.svhn_dataset import SVHNDataset
from dataset.mt_dataset import MTDataset


def metrics_dataloader(config, type, mode):
    if type == "org":
        if config.data == "mnist":
            dataset_test = MNISTDataset(
                config=config, image_set=config.detection_model.image_set, mode=mode, augmentation=False
            )
        elif config.data == "gtsrb":
            dataset_test = GTSRBDataset(
                config=config, image_set=config.detection_model.image_set, mode=mode, augmentation=False
            )

        elif config.data == "gtsrb-gray":
            dataset_test = GTSRBDataset_gray(
                config=config, image_set=config.detection_model.image_set, mode=mode, augmentation=False
            )

        elif config.data == "svhn":
            dataset_test = SVHNDataset(
                config=config, image_set=config.detection_model.image_set, mode=mode, augmentation=False
            )
        else:
            raise NotImplementedError("Please extend the method to defined dataset.")

    elif type == "crashes":
        if config.data == "mnist" or config.data == "svhn" or config.data == "gtsrb" or config.data == "gtsrb-gray":
            dataset_test = CornerCaseDataset(
                config=config, image_set=config.detection_model.image_set, mode=mode, augmentation=False
            )
        else:
            raise NotImplementedError("Please extend the method to defined dataset.")
    elif type == "mt":
        if config.data == "mnist" or config.data == "svhn" or config.data == "gtsrb" or config.data == "gtsrb-gray":
            dataset_test = MTDataset(
                config=config, image_set=config.detection_model.image_set, mode=mode, augmentation=False
            )
        else:
            raise NotImplementedError("Please extend the method to defined dataset.")
    else:
        raise NotImplementedError

    print("Length of the dataset is: ", len(dataset_test))

    return dataset_test
