import collections
from collections import OrderedDict

import torch
import numpy as np

from fuzzer.util import predict
from utils.util import z_score_normalization


class Coverage:
    def __init__(self, model, profiling_dict, config, k):
        """
        Args:
            model: model object (access to model.intermediate_layers, model.num_neurons)
            profiling_dict: models profile
            config: configuration information
            k: metric parameter corresponding to criteria
        """
        self.model = model
        self.fuzz_criteria = config.fuzz_criteria
        self.profiling_dict = profiling_dict
        self.network_type = config.network_type
        self.config = config
        self.normalize_data = config.normalize_data
        self.outputs = []
        self.layer_start_index = []
        self.start = 0
        self.eucl_dist = 0

        if self.fuzz_criteria == "nbc":
            self.k = k + 1
            self.bytearray_len = self.k * 2
        elif self.fuzz_criteria == "snac":
            self.k = k + 1
            self.bytearray_len = self.k
        elif self.fuzz_criteria == "nc":
            self.k = k
            self.bytearray_len = 1
        else:
            self.k = k
            self.bytearray_len = self.k

        num = 0
        for i, layer in enumerate(self.model.intermediate_layers):
            self.layer_start_index.append(num)
            num += int(self.model.num_neurons[i] * self.bytearray_len)

        self.total_size = num
        self.cov_dict = collections.OrderedDict()

    def predict(self, input, centroid=None):
        """
        preprocess the seed and forward pass of network
        :param input: seed under consideration
        :return: output of layers  of network and the final predicted class
        """
        if len(input) == 0:
            return None, None
        layers_outputs, detections, output_dict, feature_vector = predict(
            self.model, input, self.network_type, self.config
        )  # self.normalize_data
        coverage_list = self.update_coverage(layers_outputs, centroid, feature_vector)
        return coverage_list, detections, output_dict, feature_vector

    def kmnc_coverage(self, layers_outputs, ptr):
        """
        For each neuron, the range of its values (obtained from training data) are partitioned into k sections.
        An input covers a section of a neuron if the output value falls into the corresponding value section range.
        KMNC measures the ratio of all covered sections of all neurons of a DNN.
        :param layers_outputs: outputs of layers of network
        :param ptr: array to be filled in
        :return:
        """
        for idx, layer_name in enumerate(self.model.intermediate_layers):
            layer_outputs = layers_outputs[layer_name]
            for seed_id, layer_output in enumerate(layer_outputs):
                for neuron_idx in range(layer_output.shape[0]):  # layer_output.shape[0]), self.model.num_neurons[0]
                    profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]
                    _, _, _, lower_bound, upper_bound = profiling_data_list

                    unit_range = (upper_bound - lower_bound) / self.k
                    output = torch.mean(layer_output[neuron_idx, ...])

                    if unit_range == 0.0 or output > upper_bound or output < lower_bound:
                        continue

                    subrange_index = int((output - lower_bound) / unit_range)
                    if subrange_index == self.k:
                        subrange_index -= 1

                    id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + subrange_index
                    num = ptr[seed_id][id]
                    assert num == 0
                    if num < 255:
                        num += 1
                        ptr[seed_id][id] = num

    def bknc_coverage(self, layers_outputs, ptr, rev):
        """
        layer level testing criterion, which measures the ratio of neurons that have once been the most active k
        neurons of each layer on a given test set.
        :param layers_outputs: outputs of layers of network
        :param ptr: array to be filled in
        :param rev: bknc : false or tknc : true
        :return:
        """
        for idx, layer_name in enumerate(self.model.intermediate_layers):
            layer_outputs = layers_outputs[layer_name]
            for seed_id, layer_output in enumerate(layer_outputs):
                layer_output_dict = {}
                for neuron_idx in range(self.model.num_neurons[0]):
                    output = torch.mean(layer_output[neuron_idx, ...])
                    layer_output_dict[neuron_idx] = output

                sorted_index_output_dict = OrderedDict(
                    sorted(layer_output_dict.items(), key=lambda x: x[1], reverse=rev)
                )
                # for list if the top_k > current layer neuron number, the whole list would be used, not out of bound
                top_k_node_index_list = list(sorted_index_output_dict.keys())[: self.k]

                for top_sec, top_idx in enumerate(top_k_node_index_list):
                    id = self.start + self.layer_start_index[idx] + top_idx * self.bytearray_len + top_sec
                    num = ptr[seed_id][id]
                    if num < 255:
                        num += 1
                        ptr[seed_id][id] = num

    def nbc_coverage(self, layers_outputs, ptr):
        """
        NBC analyzes the value range of a neuron covered by training data, and measures to what extent the corner-case
        regions outside major functional range of a neuron are covered.
        :param layers_outputs: outputs of layers of network
        :param ptr: array to fill in
        :return:
        """
        for idx, layer_name in enumerate(self.model.intermediate_layers):
            layer_outputs = layers_outputs[layer_name]
            for seed_id, layer_output in enumerate(layer_outputs):
                for neuron_idx in range(layer_output.shape[0]): # self.model.num_neurons[0]

                    output = torch.mean(layer_output[neuron_idx, ...])

                    profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]

                    _, _, _, lower_bound, upper_bound = profiling_data_list

                    k_multisection = 1500
                    unit_range = (upper_bound - lower_bound) / k_multisection
                    if unit_range == 0:
                        unit_range = 0.05

                    # the hypo active case, the store targets from low to -infi
                    if output < lower_bound:
                        # float here
                        target_idx = (lower_bound - output) / unit_range
                        if target_idx > (self.k - 1):
                            id = (
                                self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + self.k - 1
                            )
                        else:
                            id = (
                                self.start
                                + self.layer_start_index[idx]
                                + neuron_idx * self.bytearray_len
                                + int(target_idx)
                            )
                        num = ptr[seed_id][id]
                        if num < 255:
                            num += 1
                            ptr[seed_id][id] = num
                        continue

                    # the hyperactive case
                    if output > upper_bound:
                        target_idx = (output - upper_bound) / unit_range
                        if target_idx > (self.k - 1):
                            id = (
                                self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + self.k - 1
                            )
                        else:
                            id = (
                                self.start
                                + self.layer_start_index[idx]
                                + neuron_idx * self.bytearray_len
                                + int(target_idx)
                            )
                        num = ptr[seed_id][id]
                        if num < 255:
                            num += 1
                            ptr[seed_id][id] = num
                        continue

    def snac_coverage(self, layers_outputs, ptr):
        """
        for each neuron, SNAC considers the value range that is above the maximum value seen during training.
        SNAC measures how the upper corner-case regions of neurons are covered.
        :param layers_outputs: outputs of layers of network
        :param ptr: array to fill in
        :return:
        """
        for idx, layer_name in enumerate(self.model.intermediate_layers):
            layer_outputs = layers_outputs[layer_name]
            for seed_id, layer_output in enumerate(layer_outputs):
                for neuron_idx in range(self.model.num_neurons[0]):
                    output = torch.mean(layer_output[neuron_idx, ...])

                    profiling_data_list = self.profiling_dict[(layer_name, neuron_idx)]
                    _, _, _, lower_bound, upper_bound = profiling_data_list

                    k_multisection = 1000
                    unit_range = (upper_bound - lower_bound) / k_multisection
                    if unit_range == 0:
                        unit_range = 0.05

                    # the hyperactive case
                    if output > upper_bound:
                        target_idx = (output - upper_bound) / unit_range
                        if target_idx > (self.k - 1):
                            id = (
                                self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + self.k - 1
                            )
                        else:
                            id = (
                                self.start
                                + self.layer_start_index[idx]
                                + neuron_idx * self.bytearray_len
                                + int(target_idx)
                            )
                        num = ptr[seed_id][id]
                        if num < 255:
                            num += 1
                            ptr[seed_id][id] = num
                        continue

    def nc_coverage(self, layers_outputs, ptr):
        """
        Given an input, a neuron is activated and marked as covered if its output value is above a predefined threshold.
        NC measures the ratio of activated neurons of a DNN.
        :param layers_outputs: outputs of layers of network
        :param ptr: array to be filled in
        :return: the neurons that can be covered by the input
        """
        for idx, layer_name in enumerate(self.model.intermediate_layers):
            layer_outputs = layers_outputs[layer_name]
            for seed_id, layer_output in enumerate(layer_outputs):
                scaled = z_score_normalization(layer_output)
                for neuron_idx in range(layer_output.shape[0]): # self.model.num_neurons[0]
                    if torch.mean(scaled[neuron_idx, ...]) > self.k:
                        id = self.start + self.layer_start_index[idx] + neuron_idx * self.bytearray_len + 0
                        ptr[seed_id][id] = 1

    def lscd_coverage(self, layers_outputs, ptr, centroid, feature_vector):
        ab = centroid - feature_vector
        dist = np.linalg.norm(ab, ord=2)
        # print('Euclidean Distance from centroid: {}'.format(dist))
        ptr = np.array([dist])
        return ptr

    @torch.no_grad()
    def update_coverage(self, layers_outputs, centroid=None, feature_vector=None):
        """
        We implement the following metrics:
        NC from DeepXplore and DeepTest. KMNC, BKNC, TKNC, NBC, SNAC from DeepGauge2.0.
        :param layers_outputs: The outputs of internal layers for a batch of mutants
        :return: ptr: array that records the coverage information
        """
        v = list(layers_outputs.values())
        batch_num = v[0].shape[0]
        ptr = torch.zeros(self.total_size, dtype=torch.uint8).unsqueeze(0).repeat(batch_num, 1)
        if len(v) > 0 and batch_num > 0:
            if self.fuzz_criteria == "kmnc":
                self.kmnc_coverage(layers_outputs, ptr)
            elif self.fuzz_criteria == "bknc":
                self.bknc_coverage(layers_outputs, ptr, False)
            elif self.fuzz_criteria == "tknc":
                self.bknc_coverage(layers_outputs, ptr, True)
            elif self.fuzz_criteria == "nbc":
                self.nbc_coverage(layers_outputs, ptr)
            elif self.fuzz_criteria == "snac":
                self.snac_coverage(layers_outputs, ptr)
            elif self.fuzz_criteria == "nc":
                self.nc_coverage(layers_outputs, ptr)
            elif self.fuzz_criteria == "flann":
                return layers_outputs[-1]
            elif self.fuzz_criteria == "lscd":
                # ptr = torch.zeros(1, dtype=torch.float32).unsqueeze(0).repeat(batch_num, 1)
                return self.lscd_coverage(layers_outputs, ptr, centroid, feature_vector)
            else:
                print("Please select a valid coverage criteria as feedback:")
                print("['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc', 'flann', 'lscd']")
                raise NotImplementedError
        return ptr
