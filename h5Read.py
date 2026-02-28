import h5py

with h5py.File("mean_std_0.h5", "r") as f:
    def print_structure(name, obj):
        print(name, "—", type(obj).__name__,
              getattr(obj, "shape", ""), getattr(obj, "dtype", ""))
    f.visititems(print_structure)