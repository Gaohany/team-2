from models.gtsrb.gtsrb_lenet.model import LeNet
from models.gtsrb.gtsrb_new.model import GTSRB_new

from models.mnist.lenet5.model import MNIST_Lenet5
from models.svhn.svhn_mixed.model import SVHN_mixed

model_structure = {
    "mnist-lenet5": MNIST_Lenet5,
    "gtsrb-new": GTSRB_new,
    "gtsrb-lenet": LeNet,
    "svhn-mixed": SVHN_mixed,
}

metrics_param = {
    "kmnc": 1500,
    "bknc": 10,
    "tknc": 10,
    "nbc": 10,
    "newnc": 10,
    "nc": 0.75,
    "flann": 1.0,
    "snac": 10,
    "nlc": None,
    "lscd": 1,
}
