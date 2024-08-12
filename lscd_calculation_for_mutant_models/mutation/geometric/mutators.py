import random
import time

import torch

from mutation.geometric.image_mutations import (
    blur,
    brightness,
    contrast,
    image_noise,
    pixel_change,
    rotation,
    scale,
    shear,
    translation,
)

# params order: translation, scale, shear, rotation, contrast, brightness, blur, pixel change, noise
params = [
    list(range(-3, 3)),
    list(map(lambda x: x * 0.1, list(range(7, 12)))),
    list(map(lambda x: x * 0.1, list(range(-6, 6)))),
    list(range(-50, 50)),
    list(map(lambda x: x * 0.6, list(range(5, 13)))),
    list(range(-20, 20)),
    list(range(1, 10)),
    list(range(1, 250)),
    list(range(1, 4)),
]

transformations = [translation, scale, shear, rotation, contrast, brightness, blur, pixel_change, image_noise]


def mutate_one(reference_img, seed, transformation_class, l0_ref, linf_ref, classA, classB, config, try_num=50):
    """
    l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
    between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1}) in Equation 2 of the paper
    :param reference_img: img from which to mutate
    :param seed: img to be mutated
    :param transformation_class: 0 : both Affine and Pixel allowed, 1 : only pixel because Affine already used
    :param l0_ref:
    :param linf_ref:
    :param try_num: maximum number of trials in Algorithm 2
    :return: reference image, mutated image, transformation type applied, l0 reference and l infinity reference
    """

    # a, b is the alpha and beta in Equation 1 in the paper
    a = 0.02
    b = 0.20
    z, x, y = seed.shape

    # l0: alpha * size(s), l_infinity: beta * 255 in Equation 1
    l0 = int(a * x * y * z)
    l_infinity = int(b * 255)

    for ii in range(try_num):
        random.seed(time.time())
        if transformation_class == 0:  # 0: can choose class A and B
            tid = random.sample(classA + classB, 1)[0]
            # Randomly select one transformation   Line-7 in Algorithm2
            transformation = transformations[tid]
            selected_params = params[tid]
            # Randomly select one parameter Line 10 in Algo2
            param = random.sample(selected_params, 1)[0]
            seed = seed.to("cpu")
            reference_img = reference_img.to("cpu")
            # Perform the transformation  Line 11 in Algo2
            # img_mutated = transformation(copy.deepcopy(seed), param)
            img_mutated = transformation(seed, param)
            if tid in classA:
                sub = torch.sub(reference_img, img_mutated)
                # check whether it is a valid mutation. i.e., Equation 1 and Line 12 in Algo2
                l0_ref = (sub != 0).sum()
                linf_ref = torch.max(torch.abs(sub))
                if l0_ref < l0 or linf_ref < l_infinity:
                    return reference_img, img_mutated, 0, 1, l0_ref, linf_ref
            else:  # B
                # If the current transform is Affine, update the reference image and transform state of the seed.
                # reference_img = transformation(copy.deepcopy(reference_img), param)
                reference_img = transformation(reference_img, param)
                return reference_img, img_mutated, 1, 1, l0_ref, linf_ref

        if transformation_class == 1:  # 1: can only choose class A
            tid = random.sample(classA, 1)[0]
            transformation = transformations[tid]
            selected_params = params[tid]
            param = random.sample(selected_params, 1)[0]
            seed = seed.to("cpu")
            reference_img = reference_img.to("cpu")
            # img_mutated = transformation(copy.deepcopy(seed), param)
            img_mutated = transformation(seed, param)
            sub = torch.sub(reference_img, img_mutated)

            # To compute the value in Equation 2 in the paper.
            l0_new = l0_ref + (sub != 0).sum()
            linf_new = max(linf_ref, torch.max(torch.abs(sub)))
            if l0_new < l0 or linf_new < l_infinity:
                return reference_img, img_mutated, 1, 1, l0_ref, linf_ref
    # Otherwise the mutation is failed. Line 20 in Algo 2
    return reference_img, seed, transformation_class, 0, l0_ref, linf_ref


def mutate_without_limitation(ref_img, classA, classB):
    tid = random.sample(classA + classB, 1)[0]
    transformation = transformations[tid]
    ori_shape = ref_img.shape
    selected_params = params[tid]
    param = random.sample(selected_params, 1)[0]
    img_new = transformation(ref_img, param)
    img_new = img_new.reshape(ori_shape)
    return img_new


# Algorithm 2
def image_random_mutate(seed, num_mutants, classA, classB, config):
    """
    Randomly mutate image and return the  successful and acceptable mutations and corresponding info
    :param seed: information about the image being read in, object of class seed
    :param num_mutants: number of mutated images to try and generate
    :return: reference images, mutated images, transformation type applied, l0 reference and l infinity reference
    """
    imgs = torch.load(seed.file_name)
    reference_img, seed_img = imgs[0], imgs[1]
    reference_images, mutated_seeds, transformation_classes, l0_references, linf_references = [], [], [], [], []
    for i in range(num_mutants):
        reference_img_out, mutated_seed, transformation, successful_mutation, l0_ref, linf_ref = mutate_one(
            reference_img, seed_img, seed.transformation_class, seed.l0_ref, seed.linf_ref, classA, classB, config
        )
        if successful_mutation:
            reference_images.append(reference_img_out)
            mutated_seeds.append(mutated_seed)
            transformation_classes.append(transformation)
            l0_references.append(l0_ref)
            linf_references.append(linf_ref)
        else:
            print("Mutation failed continuing with original seed.")
            reference_images.append(reference_img)
            mutated_seeds.append(seed_img)
            transformation_classes.append(transformation)
            l0_references.append(l0_ref)
            linf_references.append(linf_ref)
    if mutated_seeds:
        mutated_images = torch.stack(mutated_seeds)
        reference_images = torch.stack(reference_images)
        return reference_images, mutated_images, transformation_classes, l0_references, linf_references
    return reference_images, mutated_seeds, transformation_classes, l0_references, linf_references


if __name__ == "__main__":
    # for testing purposes send in a random image and view mutated results, some mutations are verifybly bad.
    # More limitations needed.
    print("main Test.")
