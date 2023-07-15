import numpy as np
import random


class HetToFImage:

    def __init__(self, abs_dataset, z_, r_i_dataset, n_samples=5, h=10, w=10, mats=None, filter=False):

        r = range(h)
        r = np.array(r)
        r = np.reshape(r, (h, 1))
        r = np.repeat(r, w, axis=1)

        c = range(w)
        c = np.array(c)
        c = np.reshape(c, (1, w))
        c = np.repeat(c, h, axis=0)

        n_mats = abs_dataset.shape[0]
        n_inst = abs_dataset.shape[1]

        self.new_abs_dataset = np.empty((n_samples, 1), dtype=object)
        self.new_r_i_dataset = np.empty((n_samples, 1), dtype=object)
        self.labels = np.empty((n_samples, 1), dtype=object)
        self.num_mats = np.empty((n_samples, 1), dtype=object)
        for m in range(n_samples):
            
            if mats is None:
                n = random.randrange(1, n_mats + 1)
            else:
                n = mats

            self.num_mats[m, 0] = n

            p = [0] * n
            i = 0
            while i < n:
                pos = (random.randrange(h), random.randrange(w))
                if pos in p:
                    pass
                else:
                    p[i] = pos
                    i += 1
    
            r_i = np.empty((n, 1), dtype=object)
            c_i = np.empty((n, 1), dtype=object)
            i_i = np.empty((n, 1), dtype=object)
            for i in range(n):
                r_i[i, 0] = (r - p[i][0]) ** 2
                c_i[i, 0] = (c - p[i][1]) ** 2
                i_i[i, 0] = np.sqrt(r_i[i, 0] + c_i[i, 0])
                i_i[i, 0] = i_i[i, 0][np.newaxis, ...]
    
            l = np.concatenate(i_i[:, 0], axis=0)
            l = np.argmin(l, axis=0)
    
            l_i = np.empty((n, 1), dtype=object)

            for i in range(n):
                new_l = l.copy()
                new_l[new_l == 1] = 1000
                a = np.unique(new_l)
    
                new_l[new_l == a[i]] = 1
                new_l[new_l != 1] = 0
    
                l_i[i, 0] = new_l[np.newaxis, ...]

            d_i_abs = np.empty((n, 1), dtype=object)
            d_i_r_i = np.empty((n, 1), dtype=object)
            label_i = np.empty((n, 1), dtype=object)
            mat_list = np.random.permutation(n_mats)
            inst_list = np.random.permutation(n_inst)
            for i in range(n):
                d_i_abs[i, 0] = abs_dataset[mat_list[i], inst_list[0]] * l_i[i, 0]
                d_i_abs[i, 0][d_i_abs[i, 0] == 0] = 1

                d_i_r_i[i, 0] = r_i_dataset[mat_list[i], inst_list[0]] * l_i[i, 0]
                d_i_r_i[i, 0][d_i_r_i[i, 0] == 0] = 1

                if mat_list[i] == 0:
                    mat_list[i] = 1000

                label_i[i, 0] = mat_list[i] * l_i[i, 0]
                label_i[i, 0][label_i[i, 0] == 0] = 1

            d_abs = np.concatenate(d_i_abs[:, 0], axis=0)
            d_abs = np.prod(d_abs, axis=0)

            #   if filter:
                #   d_abs = d_abs * z_[0, inst_list[0]]
            self.new_abs_dataset[m, 0] = d_abs

            d_r_i = np.prod(d_i_r_i[:, 0])
            self.new_r_i_dataset[m, 0] = d_r_i

            lab = np.concatenate(label_i[:, 0], axis=0)
            lab = np.prod(lab, axis=0)
            lab[lab == 1000] = 0
            self.labels[m, 0] = lab
