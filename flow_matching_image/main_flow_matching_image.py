import argparse, os, random, math
import numpy as np
from pathlib import Path
from libs.data_lib import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pdb

from libs.utils import *
from models.fm_models import *

print(torch.cuda.is_available())

# ----------------------------
# Trajectory sampling tools
# ----------------------------

def imshow_tensor(x):
	img = x[0].detach().cpu().numpy()      # Take first image, move to CPU, convert to NumPy
	img = img.transpose(1, 2, 0)            # (C,H,W) -> (H,W,C)
	img = (img * 0.5 + 0.5).clip(0,1)

	plt.imshow(img)
	plt.axis('off')
	plt.show()

@torch.no_grad()
def save_grid(x, path, nrow=8):
	grid = vutils.make_grid(x.clamp(0, 1), nrow=nrow)
	vutils.save_image(grid, path)

@torch.no_grad()
def save_trajectory_grid(traj, path):
	steps, B, C, H, W = traj.shape
	imgs = traj.permute(1, 0, 2, 3, 4).reshape(B * steps, C, H, W)
	grid = vutils.make_grid(normalize_to_0_1(imgs), nrow=steps) #.clamp(0, 1)
	vutils.save_image(grid, path)

# ----------------------------
# FM pieces (arch-agnostic)
# ----------------------------


def get_velocity(model, xt, t, arch: str):
	if arch == "potential":
		xt = xt.detach().requires_grad_(True)
		psi = model(xt, t)
		# v ~ (x_1-x_0)
		# v = \nabla ( \sum_i^B \psi(xt_i))
		#\nabla_{x_1}( psi(x_1) + psi(x_2)+ psi(x_3)) = \nabla_{x_1}( psi(x_1))
		v = torch.autograd.grad(psi.sum(), xt, create_graph=True)[0]
		return v, xt
	else:
		v = model(xt, t)
		return v, xt

# ---- schedules ----
def coeffs_and_derivs(t: torch.Tensor, schedule: str):
	if schedule == "linear":
		a0 = 1.0 - t
		a1 = t
		a0_dot = -torch.ones_like(t)
		a1_dot = torch.ones_like(t)
		return a0, a1, a0_dot, a1_dot
	elif schedule == "cosine":
		half_pi_t = (math.pi * 0.5) * t
		a0 = torch.cos(half_pi_t)
		a1 = torch.sin(half_pi_t)
		a0_dot = -(math.pi * 0.5) * torch.sin(half_pi_t)
		a1_dot = (math.pi * 0.5) * torch.cos(half_pi_t)
		return a0, a1, a0_dot, a1_dot
	elif schedule == "ddpm":
		epsilon=0.0000001
		a0 = torch.sqrt(1-t)
		a1 = torch.sqrt(t)
		a0_dot = -1/(2*torch.sqrt(1-t)+epsilon)
		a1_dot = 1/(2*torch.sqrt(t)+epsilon)
		return a0, a1, a0_dot, a1_dot
	else:
		raise ValueError(f"Unknown schedule: {schedule}")

def fm_training_step(model, x0, x1, device, arch="potential", schedule="linear", weight_decay=0.0):
	B = x1.size(0)
	x0 = x0.to(device)
	x1 = x1.to(device)
	t = torch.rand(B, device=device)
	a0, a1, a0_dot, a1_dot = coeffs_and_derivs(t, schedule)
	xt = a0.view(B, 1, 1, 1) * x0 + a1.view(B, 1, 1, 1) * x1
	v, _ = get_velocity(model, xt, t, arch)
	target = a0_dot.view(B, 1, 1, 1) * x0 + a1_dot.view(B, 1, 1, 1) * x1
	loss = F.mse_loss(v, target)
	if weight_decay > 0:
		loss = loss + weight_decay * v.pow(2).mean()
	return loss

def sample_euler(model, source_loader, steps=50, device="cuda", arch="unet_tiny"):
	model.eval()
	h = 1.0 / steps
	torch.manual_seed(123)
	# get batch of source data
	x = next(iter(source_loader))[:n_samples, :, :, :].to(device)
	
	for k in range(steps):
		t = torch.full((x.shape[0],), (k + 0.5) / steps, device=device)
		v, _ = get_velocity(model, x, t, arch)
		x = x + h * v
	return x

def sample_euler_trajectory(model, source_loader, n_samples, steps=50, device="cuda", arch="unet_tiny"):
	model.eval()
	h = 1.0 / steps

	torch.manual_seed(123)

	# get batch of source data
	x = next(iter(source_loader))[:n_samples, :, :, :].to(device)

	# preallocate trajectory tensor: (steps, n_samples, C, H, W)
	traj = torch.empty(
		(steps, n_samples, *x.shape[1:]),
		device=device,
		dtype=x.dtype,
	)

	with torch.no_grad():
		for k in range(steps):
			t = torch.full((n_samples,), (k + 0.5) / steps, device=device)
			v, _ = get_velocity(model, x, t, arch)
			x = x + h * v
			traj[k].copy_(x)  # write in-place, no growing list

	return traj

# ----------------------------
# Main
# ----------------------------
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--target_dataset", type=str, default="mnist",\
		help="Target distribution (must be a string or a directory")
	parser.add_argument("--source_dataset", type=str, default="gaussian",\
		help="Source distribution (must be a string or a directory)")
	parser.add_argument("--shuffle_target", type=int, default="1",\
		help="Whether to shuffle the target distribution or not (0 or 1)")
	parser.add_argument("--shuffle_source", type=int, default="1",\
		help="Whether to shuffle the source distribution or not (0 or 1)")
	parser.add_argument("--train", type=int, default=1)
	parser.add_argument("--model_path",type=str, default="")
	parser.add_argument("--save_dir", type=str, default="./results")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--wd", type=float, default=0.0)
	parser.add_argument("--image_size", type=int, default=-1)
	parser.add_argument("--n_channels", type=int, default=1)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--ckpt_every", type=int, default=20)
	parser.add_argument("--sample_every", type=int, default=20)
	parser.add_argument("--samples_n", type=int, default=8)
	parser.add_argument("--steps_sampler", type=int, default=20)
	parser.add_argument("--max_samples", type=int, default=1000)
	parser.add_argument("--subset_seed", type=int, default=123)
	parser.add_argument("--classes", type=int, nargs="*", default=[1])
	parser.add_argument("--arch", type=str, choices=["potential", "unet", "unet_tiny"], default="unet_tiny")
	parser.add_argument("--schedule", type=str, choices=["linear", "cosine", "ddpm"], default="linear")
	

	args = parser.parse_args()

	set_seed(args.seed)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print("device:", device, "| arch:", args.arch, "| schedule:", args.schedule)
	print("target_dataset: ",args.target_dataset)
	print("source_dataset: ",args.source_dataset)
	print("train: ",args.train)
	print("Save dir: ", args.save_dir)
	print("args.sample_every: ",args.sample_every)
	print("args.batch_size: ",args.batch_size)
	print("args.epochs: ",args.epochs)
	print("args.image_size: ",args.image_size)
	print("args.n_channels: ",args.n_channels)


	####################################
	#########    DATASET SETUP   #######
	####################################

	if(args.train==1):
		target_loader,img_size_target,n_channels_target =\
			setup_data_loader(dataset_name=args.target_dataset,\
				shuffle_data=args.shuffle_target,\
				image_size=args.image_size,\
				n_channels=args.n_channels,\
				batch_size=args.batch_size)
		if(args.image_size == -1):
			args.image_size = img_size_target
		if(args.n_channels == -1):
			args.n_channels = n_channels_target

		source_loader,img_size_source,n_channels_source =\
			setup_data_loader(dataset_name=args.source_dataset,\
				shuffle_data=args.shuffle_source,\
				image_size=args.image_size,\
				n_channels=args.n_channels,\
				batch_size=args.batch_size)
			

		if not (img_size_target == img_size_source and n_channels_target == n_channels_source):
			raise ValueError("Error in the creation of the data loaders: the image sizes and number of channels are inconsistent between source and target.")
		image_size = img_size_target
		n_channels = n_channels_target

		# create a name for the dataset, if it is a directory
		if(os.path.isdir(args.target_dataset)):
			target_dataset_name = get_last_directory(args.target_dataset)
		else:
			target_dataset_name = args.target_dataset
		if(os.path.isdir(args.source_dataset)):
			source_dataset_name = get_last_directory(args.source_dataset)
		else:
			source_dataset_name = args.source_dataset

		# create a data loader which loads batches of source and target data
		train_loader = FMloader(target_loader,source_loader)
	else:
		# if we are testing a pre-trained model, the data shapes must be manually specified
		target_dataset_name = args.target_dataset
		source_loader,_,_ =\
			setup_data_loader(dataset_name=args.source_dataset,\
				shuffle_data=args.shuffle_source,\
				image_size=args.image_size,\
				n_channels=args.n_channels,\
				batch_size=args.batch_size)
		n_channels = args.n_channels


	# add the target database to the save_dir
	args.save_dir =args.save_dir +"/"+target_dataset_name
	# if necessary, create the directory
	os.makedirs(args.save_dir, exist_ok=True)

	####################################
	#########    ARCHITECTURE   ########
	####################################

	if args.arch == "unet_tiny":
		model = UNetTiny(in_channels=n_channels, base=32, out_channels=n_channels).to(device)
		print_model_stats(model, name="UNetTiny")
	elif args.arch == "unet":
		model = UNet(in_channels=n_channels, out_channels=n_channels,c=64).to(device)
		print_model_stats(model, name="UNet")
	else:
		print("Error, unknown architecture")

	####################################
	#########   TRAIN OR TEST   ########
	####################################

	if(args.train==1):

		# optimiser
		opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

		#training 
		for epoch in range(1, args.epochs + 1):
			model.train()
			running = 0.0
			for x0,x1 in train_loader:
				# clip size of x0 if necessary (if x1 is too small)
				x0 = x0[:x1.shape[0],:,:,:]

				opt.zero_grad(set_to_none=True)
				loss = fm_training_step(model,x0=x0,x1=x1,device=device,\
					arch=args.arch,schedule=args.schedule,weight_decay=args.wd
				)
				loss.backward()
				opt.step()
				running += loss.item()
			print(f"[Epoch {epoch:03d}] loss={running/len(train_loader):.6f}")

			if epoch % args.ckpt_every == 0:
				torch.save(
					{"model": model.state_dict(), "args": vars(args)},
					Path(args.save_dir) / f"{target_dataset_name}_model_epoch_{epoch:03d}_{args.arch}_{args.schedule}.pt"
				)

			if epoch % args.sample_every == 0:
				traj = sample_euler_trajectory(
					model,
					source_loader,
					n_samples=args.samples_n,
					steps=args.steps_sampler,
					device=device,
					arch=args.arch
				)
				save_trajectory_grid(
					traj,
					Path(args.save_dir) / f"{target_dataset_name}_{args.arch}_{args.schedule}_samples_epoch_{epoch:03d}.png"
				)
	else:
		ckpt = torch.load(args.model_path, map_location=device)
		state_dict = ckpt["model"]
		model.load_state_dict(state_dict)
		traj = sample_euler_trajectory(
			model,
			source_loader,
			n_samples=args.samples_n,
			steps=args.steps_sampler,
			device=device,
			arch=args.arch
		)
		save_trajectory_grid(
			traj,
			Path(args.save_dir) / f"{target_dataset_name}_{args.arch}_{args.schedule}_samples.png"
		)
		print("Saved results to : ", Path(args.save_dir) / f"{target_dataset_name}_{args.arch}_{args.schedule}_samples.png")

if __name__ == "__main__":
	main()