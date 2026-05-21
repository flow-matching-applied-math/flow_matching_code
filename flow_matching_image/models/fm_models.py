
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.utils import *

def init_kaiming(m):
	if isinstance(m, (nn.Conv2d, nn.Linear)):
		nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
		if m.bias is not None:
			nn.init.zeros_(m.bias)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embedding_l = nn.Linear(3, dim)

    def forward(self, time):
        device = time.device
        time = torch.cat([time, torch.cos(time), torch.sin(time)], dim=-1)
        return self.embedding_l(time)


# ----------------------------
# Architectures
# ----------------------------


# ----------------------------
# Tiny UNet (t as extra channel)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True), #nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True), #nn.SiLU(inplace=True),
        )
        self.block.apply(init_kaiming)

    def forward(self, x):
        return self.block(x)


class UNetTiny(nn.Module):
    """
    Tiny UNet for MNIST (28x28).
    Time conditioning: t is concatenated as an extra image channel at the input.
    Skip connections are additive (u + e), not concatenation.
    """
    def __init__(self, in_channels=1, base=32, out_channels=1):
        super().__init__()
        c0 = in_channels + 1  # +1 for t-map
        c1, c2 = base, base * 2

        # Encoder
        self.enc1 = DoubleConv(c0, c1)           # 28x28
        self.down1 = nn.MaxPool2d(2)             # -> 14x14
        self.enc2 = DoubleConv(c1, c2)           # 14x14
        self.down2 = nn.MaxPool2d(2)             # -> 7x7

        # Bottleneck
        self.bott = DoubleConv(c2, c2)           # 7x7

        # Decoder (note: up2 outputs c2 so we can add with e2)
        self.up2  = nn.ConvTranspose2d(c2, c2, 2, stride=2)  # 7->14
        self.dec2 = DoubleConv(c2, c1)                       # after add, reduce to c1
        self.up1  = nn.ConvTranspose2d(c1, c1, 2, stride=2)  # 14->28
        self.dec1 = DoubleConv(c1, c1)                       # after add

        self.out = nn.Conv2d(c1, out_channels, 1)
        self.out.apply(init_kaiming)

    def forward(self, x, t):
        b, _, h, w = x.shape
        if t.dim() == 1:
            t = t.view(b, 1, 1, 1)
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t.view(b, 1, 1, 1)
        tmap = t.expand(b, 1, h, w)
        x0 = torch.cat([x, tmap], dim=1)

        # Encoder
        e1 = self.enc1(x0)       # (B, c1, 28, 28)
        p1 = self.down1(e1)      # (B, c1, 14, 14)

        e2 = self.enc2(p1)       # (B, c2, 14, 14)
        p2 = self.down2(e2)      # (B, c2, 7, 7)

        # Bottleneck
        btt = self.bott(p2)      # (B, c2, 7, 7)

        # Decoder with additive skips
        u2 = self.up2(btt)       # (B, c2, 14, 14)
        s2 = u2 - e2             # add skip
        # Why minus instead of plus in previous line ?
        # Because if we consider that the lower layers calculate roughly x1, and the input is x0 (close to x0,
        # where the problem is more difficult), and the output should be x1-x0,
        # then the most logical skip is -e2
        d2 = self.dec2(s2)       # (B, c1, 14, 14)

        u1 = self.up1(d2)        # (B, c1, 28, 28)
        s1 = u1 - e1             # add skip
        d1 = self.dec1(s1)       # (B, c1, 28, 28)

        return self.out(d1)      # (B, out_channels, 28, 28)





class ResBlock(nn.Module):
	def __init__(self, c_in, c_out, act=nn.SiLU):
		super().__init__()
		self.act = act()
		self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=True)
		self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=True)
		self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)
		self.apply(init_kaiming)

	def forward(self, x):
		y = self.act(self.conv1(x))
		y = self.act(self.conv2(y))
		s = self.skip(x)
		return self.act(y + s)


class UNetSmall(nn.Module):
	"""
	UNet with long skip *additions* and residual blocks.
	Maps (x, t) -> v(x, t) with same shape as x.
	Time conditioning is concatenated as an extra channel at input.

	Depth: 28 -> 14 -> 7 -> 4  (encoder)
	       4  -> 7  -> 14 -> 28 (decoder)
	"""
	def __init__(self, in_channels: int = 1, out_channels: int = 1):
		super().__init__()
		C_in = in_channels + 1

		# -------- Encoder --------
		self.enc1 = ResBlock(C_in, 32, act=nn.SiLU)   # 28 x 28
		self.pool1 = nn.MaxPool2d(2)                  # 28 -> 14

		self.enc2 = ResBlock(32, 64, act=nn.SiLU)     # 14 x 14
		self.pool2 = nn.MaxPool2d(2)                  # 14 -> 7

		self.enc3 = ResBlock(64, 128, act=nn.SiLU)    # 7 x 7
		self.down3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True)  # 7 -> 4

		# Bottleneck (4 x 4)
		self.bott = ResBlock(128, 128, act=nn.SiLU)

		# -------- Decoder (additive long skips) --------
		self.up3    = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=0)  # 4 -> 7
		self.iconv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
		self.dec3   = ResBlock(128, 128, act=nn.SiLU)   # was ResBlock(256,128)

		self.up2    = nn.Upsample(scale_factor=2, mode="bilinear")   # 7 -> 14
		self.iconv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True)
		self.dec2   = ResBlock(64, 64, act=nn.SiLU)                  # was ResBlock(128,64)

		self.up1    = nn.Upsample(scale_factor=2, mode="bilinear")   # 14 -> 28
		self.iconv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)
		self.dec1   = ResBlock(32, 32, act=nn.SiLU)                  # was ResBlock(64,32)

		self.out = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=True)

		self.apply(init_kaiming)

	def forward(self, x: torch.Tensor, t: torch.Tensor):
		B, _, H, W = x.shape
		if t.dim() == 1:
			t = t.view(B, 1, 1, 1)
		elif t.dim() == 2 and t.shape[1] == 1:
			t = t.view(B, 1, 1, 1)

		x_in = torch.cat([x, t.expand(B, 1, H, W)], dim=1)

		# ----- Encode -----
		e1 = self.enc1(x_in)               # 32 x 28 x 28
		p1 = self.pool1(e1)                # 32 x 14 x 14

		e2 = self.enc2(p1)                 # 64 x 14 x 14
		p2 = self.pool2(e2)                # 64 x 7 x 7

		e3 = self.enc3(p2)                 # 128 x 7 x 7
		p3 = self.down3(e3)                # 128 x 4 x 4

		# Bottleneck
		b = self.bott(p3)                  # 128 x 4 x 4

		# ----- Decode (add skips) -----
		u3 = self.up3(b)                   # 128 x 7 x 7
		u3 = self.iconv3(u3)               # 128 x 7 x 7
		u3 = u3 - e3                       # skip
		d3 = self.dec3(u3)                 # 128 x 7 x 7

		u2 = self.up2(d3)                  # 128 x 14 x 14
		u2 = self.iconv2(u2)               # 64 x 14 x 14
		u2 = u2 - e2                       # ADD instead of concat
		d2 = self.dec2(u2)                 # 64 x 14 x 14

		u1 = self.up1(d2)                  # 64 x 28 x 28
		u1 = self.iconv1(u1)               # 32 x 28 x 28
		u1 = u1 - e1                       # skip
		d1 = self.dec1(u1)                 # 32 x 28 x 28

		return self.out(d1)                # out_channels x 28 x 28



class DoubleConvUnetBig(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, c=64):
        super().__init__()
        
        # Time embeddings for each skip connection
        self.time_mlp1 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, c)
        )
        
        self.time_mlp2 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 2*c),
            nn.SiLU(),
            nn.Linear(2*c, 2*c)
        )
        
        self.time_mlp3 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 4*c),
            nn.SiLU(),
            nn.Linear(4*c, 4*c)
        )
        
        # Encoder
        self.conv1 = DoubleConvUnetBig(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConvUnetBig(c, 2*c)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConvUnetBig(2*c, 4*c)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConvUnetBig(4*c, 8*c)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(8*c, 4*c, kernel_size=2, stride=2)
        self.conv5 = DoubleConvUnetBig(12*c, 4*c)  # 4c + 4c + 4c (skip + up + time)
        self.upconv2 = nn.ConvTranspose2d(4*c, 2*c, kernel_size=2, stride=2)
        self.conv6 = DoubleConvUnetBig(6*c, 2*c)   # 2c + 2c + 2c
        self.upconv1 = nn.ConvTranspose2d(2*c, c, kernel_size=2, stride=2)
        self.conv7 = DoubleConvUnetBig(3*c, c)     # c + c + c
        
        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    
    def forward(self, x, t):
        # Time embeddings
        t = t.view(-1, 1)
        if t.numel() == 1:
            t = t.expand(x.shape[0], -1)
        t1 = self.time_mlp1(t)        # Shape: [batch, c]
        t2 = self.time_mlp2(t)        # Shape: [batch, 2c]
        t3 = self.time_mlp3(t)        # Shape: [batch, 4c]
        
        # Encoder
        conv1 = self.conv1(x)                 # Shape: [batch, c, H, W]
        pool1 = self.pool1(conv1)             # Shape: [batch, c, H/2, W/2]
        
        conv2 = self.conv2(pool1)             # Shape: [batch, 2c, H/2, W/2]
        pool2 = self.pool2(conv2)             # Shape: [batch, 2c, H/4, W/4]
        
        conv3 = self.conv3(pool2)             # Shape: [batch, 4c, H/4, W/4]
        pool3 = self.pool3(conv3)             # Shape: [batch, 4c, H/8, W/8]
        
        conv4 = self.conv4(pool3)             # Shape: [batch, 8c, H/8, W/8]
        
        # Helper function for padding
        def pad_if_needed(upsampled, skip):
            if (upsampled.shape[-1] + 1) == skip.shape[-1]:
                return F.pad(upsampled, (0, 1, 0, 1), mode='replicate')
            return upsampled
        # Decoder Step 1
        up3 = self.upconv3(conv4)              # Shape: [batch, 4c, H/4, W/4]
        up3 = pad_if_needed(up3, conv3)        # Ensure up3 matches conv3 dimensions
        # Reshape t3 to [batch, 4c, H/4, W/4]
        t_emb3 = t3.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, up3.size(2), up3.size(3))  
        # Alternatively, use expand to save memory:
        # t_emb3 = t3.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, up3.size(2), up3.size(3))
        concat3 = torch.cat([up3, conv3, t_emb3], dim=1)   # Shape: [batch, 12c, H/4, W/4]
        conv5 = self.conv5(concat3)            # Shape: [batch, 4c, H/4, W/4]

        # Decoder Step 2
        up2 = self.upconv2(conv5)              # Shape: [batch, 2c, H/2, W/2]
        up2 = pad_if_needed(up2, conv2)        # Ensure up2 matches conv2 dimensions
        # Reshape t2 to [batch, 2c, H/2, W/2]
        t_emb2 = t2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, up2.size(2), up2.size(3))
        concat2 = torch.cat([up2, conv2, t_emb2], dim=1)   # Shape: [batch, 6c, H/2, W/2]
        conv6 = self.conv6(concat2)            # Shape: [batch, 2c, H/2, W/2]

        # Decoder Step 3
        up1 = self.upconv1(conv6)              # Shape: [batch, c, H, W]
        up1 = pad_if_needed(up1, conv1)        # Ensure up1 matches conv1 dimensions
        # Reshape t1 to [batch, c, H, W]
        t_emb1 = t1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, up1.size(2), up1.size(3))
        concat1 = torch.cat([up1, conv1, t_emb1], dim=1)   # Shape: [batch, 3c, H, W]
        conv7 = self.conv7(concat1)            # Shape: [batch, c, H, W]

        # Final Convolution
        return self.final_conv(conv7)          # Shape: [batch, out_channels, H, W]