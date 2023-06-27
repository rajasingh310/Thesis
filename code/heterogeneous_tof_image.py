import random
import numpy as np
import math

class HeterogeneousImage:

    def __init__(self, dataset, n_samples=10, h=100, w=100, rotation=True, mats=None):

        def is_prime(n):
            # check if n is less than or equal to 1
            if n <= 1:
                return False
            # check for factors from 2 to the square root of n
            for i in range(2, int(math.sqrt(n)) + 1):
                # check if n is divisible by i
                if n % i == 0:
                    return False
            # no factor found, so n is prime
            return True

        # define a function to find and output the pair of factors whose difference is the least
        def min_diff_pair(n):
            # create an empty list to store the factors
            factors = []
            # loop from 1 to n
            for i in range(1, n + 1):
                # check if i is a factor of n
                if n % i == 0:
                    # append i to the list of factors
                    factors.append(i)
            # initialize the minimum difference to a large number
            min_diff = float('inf')
            # initialize the pair with the minimum difference to None
            min_pair = None
            # loop through half of the list of factors
            for j in range(int(np.ceil(len(factors) / 2))):
                # pair up the j-th factor with the (n-j-1)-th factor
                pair = (factors[j], factors[-j - 1])
                # calculate the difference between the pair
                diff = pair[1] - pair[0]
                # check if the difference is less than the minimum difference
                if diff < min_diff:
                    # update the minimum difference and the pair with the minimum difference
                    min_diff = diff
                    min_pair = pair
            # return the pair with the minimum difference
            return min_pair

        n_mats = dataset.shape[0]
        self.new_dataset = np.empty((n_samples, 3), dtype=object)
        for i in range(n_samples):

            if mats is None:
                n = random.randrange(1, n_mats + 1)

            else:
                n = mats

            mat_list = np.random.permutation(n_mats)
            a = h * w

            while a > dataset[mat_list[0], ].shape[2]:
                n = random.randrange(1, n_mats + 1)

            if n == 1:
                self.new_dataset[i, 0] = dataset[mat_list[0], ][:, :, :a].reshape(dataset[mat_list[0], ].shape[0], h, w)
                self.new_dataset[i, 1] = np.full(shape=(h, w), fill_value=mat_list[0])
                self.new_dataset[i, 2] = n

            else:
                if n <= 3:
                    b_w = int(w / n)
                    e_w = w - int(b_w*n)

                    sub_img = np.empty((n, 2), dtype=object)
                    for j in range(n):

                        if j == n-1:
                            b_w = b_w + e_w
                        b_a = h * b_w
                        sub_img[j, 0] = dataset[mat_list[j]][:, :, :b_a].reshape(dataset[mat_list[j]].shape[0], h, b_w)
                        sub_img[j, 1] = np.full(shape=(h, b_w), fill_value=mat_list[j])

                    self.new_dataset[i, 0] = np.concatenate(sub_img[:, 0], axis=2)
                    self.new_dataset[i, 1] = np.concatenate(sub_img[:, 1], axis=1)
                    self.new_dataset[i, 2] = n

                    if rotation:
                        rand_rot = random.randrange(1, 3)
                        self.new_dataset[i, 0] = np.rot90(self.new_dataset[i, 0], rand_rot, axes=(1, 2))
                        self.new_dataset[i, 1] = np.rot90(self.new_dataset[i, 1], rand_rot, axes=(0, 1))

                else:
                    if not is_prime(n):
                        pairs = min_diff_pair(n)
                        n_row = pairs[0]
                        n_col = pairs[1]

                        b_h = int(h / n_row)
                        b_w = int(w / n_col)

                        e_w = w - b_w*n_col
                        e_h = h - b_h*n_row

                        sub_img = np.empty((n_col, 2), dtype=object)
                        img = np.empty((n_row, 2), dtype=object)
                        l = 0
                        for j in range(n_row):
                            for k in range(n_col):

                                if k == n_col-1:
                                    new_b_w = b_w + e_w

                                else:
                                    new_b_w = b_w

                                if j == n_row-1:
                                    new_b_h = b_h + e_h

                                else:
                                    new_b_h = b_h

                                b_a = new_b_h * new_b_w
                                sub_img[k, 0] = dataset[mat_list[l]][:, :, :b_a].reshape(dataset[mat_list[l]].shape[0], new_b_h, new_b_w)
                                sub_img[k, 1] = np.full(shape=(new_b_h, new_b_w), fill_value=mat_list[l])

                                l += 1

                            img[j, 0] = np.concatenate(sub_img[:, 0], axis=2)
                            img[j, 1] = np.concatenate(sub_img[:, 1], axis=1)

                        self.new_dataset[i, 0] = np.concatenate(img[:, 0], axis=1)
                        self.new_dataset[i, 1] = np.concatenate(img[:, 1], axis=0)
                        self.new_dataset[i, 2] = n

                        if rotation:
                            rand_rot = random.randrange(1, 3)
                            self.new_dataset[i, 0] = np.rot90(self.new_dataset[i, 0], rand_rot, axes=(1, 2))
                            self.new_dataset[i, 1] = np.rot90(self.new_dataset[i, 1], rand_rot, axes=(0, 1))

                    else:
                        new_n1 = n-1

                        pairs1 = min_diff_pair(new_n1)
                        n_row1 = pairs1[0]
                        n_col1 = pairs1[1]
                        error1 = n_col1 - n_row1

                        new_n2 = n - 2

                        pairs2 = min_diff_pair(new_n2)
                        n_row2 = pairs2[0]
                        n_col2 = pairs2[1]
                        error2 = n_col2 - n_row2

                        if error1 <= error2:

                            b_h = int(h / n_row1)
                            b_w = int(w / n_col1)
                            e_w = w - b_w * n_col1

                            if e_w < 3:
                                b_w = int(b_w * 0.9)

                            e_w = w - b_w * n_col1
                            e_h = h - b_h * n_row1

                            sub_img = np.empty((n_col1, 2), dtype=object)
                            img = np.empty((n_row1, 2), dtype=object)
                            l = 0
                            for j in range(n_row1):
                                for k in range(n_col1):

                                    if j == n_row1 - 1:
                                        new_b_h = b_h + e_h

                                    else:
                                        new_b_h = b_h

                                    b_a = new_b_h * b_w
                                    sub_img[k, 0] = dataset[mat_list[l]][:, :, :b_a].reshape(dataset[mat_list[l]].shape[0], new_b_h, b_w)
                                    sub_img[k, 1] = np.full(shape=(new_b_h, b_w), fill_value=mat_list[l])

                                    l += 1

                                img[j, 0] = np.concatenate(sub_img[:, 0], axis=2)
                                img[j, 1] = np.concatenate(sub_img[:, 1], axis=1)

                            last_image = dataset[mat_list[l]][:, :, :h*e_w].reshape(dataset[mat_list[l]].shape[0], h, e_w)
                            last_image_label = np.full(shape=(h, e_w), fill_value=mat_list[l])

                            self.new_dataset[i, 0] = np.concatenate((np.concatenate(img[:, 0], axis=1), last_image), axis=2)
                            self.new_dataset[i, 1] = np.concatenate((np.concatenate(img[:, 1], axis=0), last_image_label), axis=1)
                            self.new_dataset[i, 2] = n

                            if rotation:
                                rand_rot = random.randrange(1, 3)
                                self.new_dataset[i, 0] = np.rot90(self.new_dataset[i, 0], rand_rot, axes=(1, 2))
                                self.new_dataset[i, 1] = np.rot90(self.new_dataset[i, 1], rand_rot, axes=(0, 1))

                        else:
                            b_h = int(h / n_row2)
                            b_w = int(w / n_col2)
                            e_w = w - b_w * n_col1

                            if e_w < 3:
                                b_w = int(b_w * 0.9)

                            e_w = w - b_w * n_col2
                            e_h = h - b_h * n_row2

                            sub_img = np.empty((n_col2, 2), dtype=object)
                            img = np.empty((n_row2, 2), dtype=object)
                            l = 0
                            for j in range(n_row2):
                                for k in range(n_col2):

                                    if j == n_row2 - 1:
                                        new_b_h = b_h + e_h

                                    else:
                                        new_b_h = b_h

                                    b_a = new_b_h * b_w
                                    sub_img[k, 0] = dataset[mat_list[l]][:, :, :b_a].reshape(dataset[mat_list[l]].shape[0], new_b_h, b_w)
                                    sub_img[k, 1] = np.full(shape=(new_b_h, b_w), fill_value=mat_list[l])

                                    l += 1

                                img[j, 0] = np.concatenate(sub_img[:, 0], axis=2)
                                img[j, 1] = np.concatenate(sub_img[:, 1], axis=1)

                            last_image1 = dataset[mat_list[l]][:, :, :int(h/2) * e_w].reshape(dataset[mat_list[l]].shape[0], int(h/2), e_w)
                            last_image_label1 = np.full(shape=(int(h/2), e_w), fill_value=mat_list[l])

                            last_image2 = dataset[mat_list[l+1]][:, :, :int(h/2) * e_w].reshape(dataset[mat_list[l+1]].shape[0], int(h/2), e_w)
                            last_image_label2 = np.full(shape=(int(h/2), e_w), fill_value=mat_list[l+1])

                            last_image = np.concatenate((last_image1, last_image2), axis=1)
                            last_image_label = np.concatenate((last_image_label1, last_image_label2), axis=0)

                            self.new_dataset[i, 0] = np.concatenate((np.concatenate(img[:, 0], axis=1), last_image), axis=2)
                            self.new_dataset[i, 1] = np.concatenate((np.concatenate(img[:, 1], axis=0), last_image_label), axis=1)
                            self.new_dataset[i, 2] = n

                            if rotation:
                                rand_rot = random.randrange(1, 3)
                                self.new_dataset[i, 0] = np.rot90(self.new_dataset[i, 0], rand_rot, axes=(1, 2))
                                self.new_dataset[i, 1] = np.rot90(self.new_dataset[i, 1], rand_rot, axes=(0, 1))
