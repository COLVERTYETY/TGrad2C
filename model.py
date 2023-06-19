from tinygrad.tensor import Tensor
from tinygrad.nn import optim
import numpy as np



class Model():
    def __init__(self) -> None:
        self.l1 = Tensor.glorot_uniform(1, 2)
        self.mid = Tensor.glorot_uniform(2, 2)
        self.l2 = Tensor.glorot_uniform(2, 1)

    def __call__(self, x):
        x = x/3.1415
        x = (x.dot(self.l1)).tanh()
        x = (x.dot(self.mid)).tanh()
        x = (x.dot(self.l2))
        return x

    def save(self, filename):
        weights = dict()
        for i, param in enumerate(optim.get_parameters(self)):
            for attr in dir(self):
                if getattr(self, attr) is param:
                    print("saving", attr)
                    weights[attr] = param.cpu().numpy()
        #  save with npz
        with open(filename+'.npz', 'wb') as f:
            np.savez(f, **weights)

    def load(self, filename):
        arrays = np.load(filename)
        for attr in dir(self):
            if attr in arrays:
                print("loading", attr)
                setattr(self, attr, Tensor(arrays[attr]))
        print(f"successfully loaded {filename}")

    def __repr__(self) -> str:
        rep = ""
        for param in optim.get_parameters(self):
            rep += f"{param.shape} {param.flatten().detach().numpy()}\n"
        return rep
