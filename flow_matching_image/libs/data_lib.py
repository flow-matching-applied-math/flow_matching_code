import os, gzip, struct, math, random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Optional, Callable, Tuple
from PIL import Image
from torchvision import transforms,datasets
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
import numpy as np
import pdb




###############################
#####   DATASET SETUP      ####
###############################



def setup_data_loader(dataset_name="mnist",shuffle_data=1,image_size=-1,n_channels=-1,batch_size = 16):

	# database setup
	# this can be either a pre-defined distribution specified by a string (mnist, gaussian, cifar10),
	# or a database specified by a directory name
	if(dataset_name == "mnist"):
		image_size = (28,28)
		n_channels = 1
		mnist_data_root = "./data/mnist"
		if(not os.path.isdir(mnist_data_root)):
			os.makedirs(mnist_data_root, exist_ok=True)
		data_loader = mnist_data_loader(data_root=mnist_data_root, image_size=image_size,batch_size=batch_size,shuffle=shuffle_data)
		
	elif(dataset_name == "cifar10"):
		image_size = (32,32)
		n_channels = 3
		cifar10_data_root = "./data/cifar10"
		if(not os.path.isdir(cifar10_data_root)):
			os.makedirs(cifar10_data_root, exist_ok=True)
			
		data_loader = cifar10_data_loader(data_root=cifar10_data_root, image_size=image_size,batch_size=batch_size,shuffle=shuffle_data)
	elif(dataset_name == "gaussian"):
		n_channels = n_channels
		if(image_size == -1):
			raise ValueError("If you choose a Gaussian as the target data, you must specify the image size")
		if(np.isscalar(image_size)):
			image_size = (image_size,image_size)
		# if we use a gaussian, set an artificial "length" of the database, otherwise infinite loop
		data_loader = RandomLoader("gaussian", (batch_size,n_channels,image_size[0],image_size[0]),length=1000)
	else: 
		# custom database, in one folder. 
		if( not os.path.isdir(dataset_name)):
			print("Dataset : ", dataset_name)

		# determine image size automatically if necessary
		# by default, this size is chosen to be the size of the first image in the database
		if(image_size == -1):
			img_0 = get_first_image_dir(dataset_name)
			image_size = (img_0.shape[1],img_0.shape[2])
		elif(np.isscalar(image_size)):
			# convert image to square size, if image size specified and is a scalar
			image_size = (image_size,image_size)
		# determine number of channels automatically if necessary
		if(n_channels == -1):
			img_0 = get_first_image_dir(dataset_name)
			n_channels = img_0.shape[0]

		data_loader,data_base = custom_dataloader(root_dir=dataset_name,\
			image_size=image_size, n_channels=n_channels,\
			batch_size = batch_size, num_workers = 4, shuffle = shuffle_data)
	
	return(data_loader,image_size,n_channels)


# def data_setup(target_dataset="mnist",source_dataset="gaussian",shuffle_target=1,shuffle_source=1,image_size=-1,n_channels=-1,batch_size = 16):


# 	# first, setup target dataset
# 	if(target_dataset == "mnist"):
# 		image_size = (28,28)
# 		in_channels = 1
# 		out_channels = 1
# 		n_channels = 1
# 		mnist_data_root = "./data/mnist"
# 		if(not os.path.isdir(mnist_data_root)):
# 			os.makedirs(mnist_data_root, exist_ok=True)
# 		target_loader = mnist_data_loader(data_root=mnist_data_root, image_size=image_size,batch_size=batch_size,shuffle=shuffle_target)
		
# 	elif(target_dataset == "cifar10"):
# 		image_size = (32,32)
# 		in_channels = 3
# 		out_channels = 3
# 		n_channels = 3
# 		cifar10_data_root = "./data/cifar10"
# 		if(not os.path.isdir(cifar10_data_root)):
# 			os.makedirs(cifar10_data_root, exist_ok=True)
			
# 		target_loader = cifar10_data_loader(data_root=cifar10_data_root, image_size=image_size,batch_size=batch_size,shuffle=shuffle_target)
# 	elif(target_dataset == "gaussian"):
# 		in_channels = 1
# 		out_channels = 1
# 		n_channels = 1
# 		if(image_size == -1):
# 			raise ValueError("If you choose a Gaussian as the target data, you must specify the image size")
# 		image_size = (image_size,image_size)
# 		# if we use a gaussian, set an artificial "length" of the database, otherwise infinite loop
# 		target_loader = RandomLoader("gaussian", (batch_size,n_channels,image_size[0],image_size[0]),length=1000)
# 	else: 
# 		# custom target database, in one folder. 
# 		if( not os.path.isdir(target_dataset)):
# 			print("Target dataset : ", target_dataset)

# 		# determine image size automatically if necessary
# 		# by default, this size is chosen to be the size of the first image in the database
# 		if(image_size == -1):
# 			img_0 = get_first_image_dir(target_dataset)
# 			image_size = (img_0.shape[1],img_0.shape[2])
# 		else: # convert image to square size, if image size specified
# 			image_size = (image_size,image_size)
# 		# determine number of channels automatically if necessary
# 		if(n_channels == -1):
# 			img_0 = get_first_image_dir(target_dataset)
# 			n_channels = img_0.shape[0]
# 		# input and output channels are necessarily equal
# 		in_channels = n_channels
# 		out_channels = n_channels

# 		target_loader,target_data_base = custom_dataloader(root_dir=target_dataset,\
# 			image_size=image_size, n_channels=n_channels,\
# 			batch_size = batch_size, num_workers = 4, shuffle = shuffle_target)



# 	# second, source distribution setup
# 	# this can be either a pre-defined distribution specified by a string,
# 	# or a database specified by a directory name
# 	if(source_dataset == "gaussian"):
# 		source_loader = RandomLoader("gaussian", (batch_size,n_channels,image_size[0],image_size[0]))
# 	else:
# 		# custom database, in one folder
# 		# check if source_dataset is a directory name
# 		if( not os.path.isdir(source_dataset)):
# 			raise ValueError("Error, source_dataset must be either a known distribution or a directory")
# 			print("Source dataset : ", source_dataset)

# 		# the size and number of channels is now imposed by the target distribution
# 		source_loader,source_data_base = custom_dataloader(root_dir=source_dataset,\
# 			image_size=image_size, n_channels=n_channels,\
# 			batch_size = batch_size, num_workers = 4, shuffle = shuffle_source)
	
# 	data_loader = FMloader(target_loader,source_loader)
	
# 	return(data_loader,target_loader,source_loader,image_size,in_channels,out_channels)

class FMloader:
    """
    Iterates over all batches of target_loader.
    source_loader is cycled automatically if it is shorter.

    Usage:
        loader = FMloader(target_loader, source_loader)
        for x_src, x_tgt in loader:
            ...
    """
    def __init__(self, target_loader, source_loader):
        self.target_loader = target_loader
        self.source_loader = source_loader

    def __iter__(self):
        src_iter = iter(self.source_loader)

        for batch_tgt in self.target_loader:
            try:
                batch_src = next(src_iter)
            except StopIteration:
                # source_loader exhausted: restart it (with a new shuffle)
                src_iter = iter(self.source_loader)
                batch_src = next(src_iter)

            # if dataset returns (input, label), extract only inputs
            x_src = batch_src[0] if isinstance(batch_src, (tuple, list)) else batch_src
            x_tgt = batch_tgt[0] if isinstance(batch_tgt, (tuple, list)) else batch_tgt

            yield x_src, x_tgt

    def __len__(self):
        return len(self.target_loader)

# class FMloader:
#     """
#     Iterates over all batches of target_loader.
#     source_loader is cycled automatically if it is shorter.

#     Usage:
#         loader = FMloader(target_loader,source_loader)
#         for x_src, x_tgt in loader:
#             ...
#     """
#     def __init__(self, target_loader,source_loader):
#         self.target_loader = target_loader
#         self.source_loader = source_loader

#     def __iter__(self):
#         import itertools

#         src_iter = itertools.cycle(self.source_loader)   # infinite cycling
#         tgt_iter = iter(self.target_loader)             # finite

#         for batch_tgt in tgt_iter:  # iterate over all target batches
#             batch_src = next(src_iter)


#             # if dataset returns (input, label), extract only inputs
#             x_src = batch_src[0] if isinstance(batch_src, (tuple, list)) else batch_src
#             x_tgt = batch_tgt[0] if isinstance(batch_tgt, (tuple, list)) else batch_tgt

#             yield x_src, x_tgt

#     def __len__(self):
#         # Length equals target_loader length
#         return len(self.target_loader)

class RandomLoader:
    """
    Loader that generates random batches from a specified distribution.

    shape must include batch size, e.g. (64, 3, 32, 32)
    distribution options: "gaussian"
    """

    def __init__(self, distribution: str, shape: tuple, device="cpu", length=-1):
        self.distribution = distribution.lower()
        self.shape = shape
        self.device = device
        self.length = length

    def __iter__(self):
        if self.length == -1:
            # Infinite loader
            while True:
                yield self._sample()
        else:
            # Finite loader
            for _ in range(self.length):
                yield self._sample()

    def __len__(self):
        return self.length

    def _sample(self):
        if self.distribution == "gaussian":
            return torch.randn(self.shape, device=self.device)
        else:
            raise ValueError(f"Unknown distribution '{self.distribution}'")

###############################
##    CUSTOM IMAGE DATASET   ##
###############################

class custom_dataset(Dataset):
	"""
	Loads all images from a single directory
	Returns only the transformed image.
	"""
	def __init__(
		self,
		root_dir: str,
		n_channels: int=1,
		transform: Optional[Callable] = None,
		extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
	):
		self.root_dir = root_dir
		self.transform = transform
		self.n_channels = n_channels
		self.extensions = extensions

		self.files = [
			os.path.join(root_dir, f)
			for f in sorted(os.listdir(root_dir))
			if f.lower().endswith(self.extensions)
		]

		if len(self.files) == 0:
			raise RuntimeError(f"No image files found in: {root_dir}")

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		img_path = self.files[idx]
		img = Image.open(img_path)

		# Enforce the desired number of channels
		if self.n_channels == 1:
			# Single-channel grayscale, no channel repetition
			if img.mode != "L":
				img = img.convert("L")  # 1-channel
		elif self.n_channels == 3:
			# Standard RGB
			if img.mode != "RGB":
				img = img.convert("RGB")  # 3 channels
		else:
			raise ValueError(f"Unsupported n_channels={self.n_channels}, use 1 or 3.")

		img = self.transform(img)  # typically ToTensor -> (C, H, W)

		# Check: if transform returned (H, W), add channel dim
		if isinstance(img, torch.Tensor) and img.ndim == 2:
			img = img.unsqueeze(0)

		return img


def custom_dataloader(
	root_dir: str,
	image_size: Tuple[int, int] = (128, 128),
	n_channels: int=1,
	batch_size: int = 32,
	num_workers: int = 4,
	shuffle: bool = True,
) -> DataLoader:
	"""
	Returns a PyTorch DataLoader for a single-folder image dataset.
	"""
	transform = transforms.Compose([
		transforms.Resize(image_size),
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x * 2.0 - 1.0)
	])

	dataset = custom_dataset(root_dir, n_channels=n_channels,transform=transform)

	dl = DataLoader( dataset, batch_size=batch_size, shuffle=shuffle,\
		num_workers=num_workers, pin_memory=True)


	return dl,dataset


###############################
########      MNIST     #######
###############################

def mnist_data_loader(data_root, image_size=28,batch_size=64,max_dataset_size=-1,shuffle=True):
	tfm = transforms.Compose([
		transforms.Resize(image_size),
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x * 2.0 - 1.0) # normalise to (-1,1)
	])
	ds = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
	if(max_dataset_size>0):
		ds = torch.utils.data.random_split(ds, [max_dataset_size, len(ds)-max_dataset_size])[0]
	dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
	return dl

###############################
########      CIFAR     #######
###############################

def cifar10_data_loader(data_root, image_size=32, batch_size=64, max_dataset_size=-1,shuffle=True):
	tfm = transforms.Compose([
		transforms.Resize(image_size),       # CIFAR10 is already 32x32, but allows resizing
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x * 2.0 - 1.0)  # scale to [-1, 1]
	])
	ds = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
	if max_dataset_size > 0:
		ds, _ = torch.utils.data.random_split(
			ds, [max_dataset_size, len(ds) - max_dataset_size]
		)
	
	dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
	return dl


###############################
####   DIRECTORY TOOLS     ####
###############################

def get_last_directory(path_str):
	path = Path(path_str)
	return path.name or path.parent.name  # Handles trailing slash

def get_first_image_dir(directory, extensions={".jpg", ".jpeg", ".png", ".bmp", ".tiff"}):
	# List only image files
	files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in extensions]

	if not files:
		raise FileNotFoundError("No image files found in the directory.")

	first_image_path = os.path.join(directory, files[0])
	
	# Open and convert to tensor
	first_image = transforms.ToTensor()(Image.open(first_image_path))
	
	return first_image


###############################
####     MODEL STATS       ####
###############################

def print_model_stats(model, name="model"):
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

	# bytes occupied by tensors in current dtype(s)
	param_bytes  = sum(p.nelement() * p.element_size() for p in model.parameters())
	buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
	total_bytes = param_bytes + buffer_bytes

	def fmt(n):  # pretty numbers
		return f"{n:,}"
	def fmt_mb(b):
		return f"{b/1024**2:.2f} MiB"

	print(f"[{name}]")
	print(f"  Parameters: total={fmt(total_params)}, trainable={fmt(trainable_params)}")
	print(f"  Memory (params+buffers): {fmt_mb(total_bytes)} "
		  f"(params={fmt_mb(param_bytes)}, buffers={fmt_mb(buffer_bytes)})")


###############################
###  NORMALISATION TOOLS     ##
###############################

def normalize_to_0_1(imgs: torch.Tensor) -> torch.Tensor:
	"""
	Normalize a batch of images to [0,1] range per image.
	imgs: (B, C, H, W)
	"""
	min_val = imgs.amin(dim=(1,2,3), keepdim=True)
	max_val = imgs.amax(dim=(1,2,3), keepdim=True)
	return (imgs - min_val) / (max_val - min_val + 1e-8)