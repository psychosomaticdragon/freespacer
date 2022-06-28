import torch
import torchaudio
import torch.nn.functional as F
import torch.fft
import scipy.io.wavfile as wavfile
import math
import argparse
import numpy as np
        
parser = argparse.ArgumentParser(description='spectral shaper. Usage - supermono <spectrum> <target> <output file>')
parser.add_argument('spectrum', type=str,
                    help='spectrum filename')
parser.add_argument('target', type=str,
                    help='target filename')
parser.add_argument('output', type=str,
                    help='output filename')
                    

# don't mess with these unless you know what you're doing.
window_size = 2**12
stride_size = 2**9
n = 4

# these should probably be parameters to pass, but can't be fucked rn lmao

smoothing = 256
rolloff = 256
slope_factor = 0.5
antialias = 64
threshold = -60.0
ratio = 16.0


slope = torch.linspace(1, window_size//2+1, window_size//2+1).pow(slope_factor)



antialias_window = torch.linspace(0, math.pi, antialias).cos()
antialias_window.add_(-antialias_window.min())
antialias_window.div_(antialias_window.max())

rolloff_window = torch.linspace(0, math.pi, rolloff).cos()
rolloff_window.add_(-rolloff_window.min())
rolloff_window.div_(rolloff_window.max())

correction = 0
for i in range(n):
    y = torch.ones(1)*math.pi*2
    correction = correction + ((y.cos()*i/n+1)/2).pow(n-1)

args = parser.parse_args()
coswin = torch.linspace(-math.pi, math.pi, window_size).cos()
coswin.add_(-coswin.min())
coswin.div_(coswin.max()).pow(n-1).div(correction)

sample_rate_spectrum, spectrum = wavfile.read(args.spectrum)
sample_rate_target, target = wavfile.read(args.target)

spectrum = torch.from_numpy(spectrum).float().t().unsqueeze(0)
spectrum.div_(spectrum.abs().max())
target = torch.from_numpy(target).float().t().unsqueeze(0)


if spectrum.shape[-1] > target.shape[-1]:
    zeros = torch.zeros(spectrum.shape)
    zeros[:,:,:target.shape[-1]] = target
    target = zeros

if target.shape[-1] > spectrum.shape[-1]:
    zeros = torch.zeros(target.shape)
    zeros[:,:,:spectrum.shape[-1]] = spectrum
    spectrum = zeros

spectrum_shape = spectrum.shape
spectrum = F.unfold(spectrum.transpose(0,1).unsqueeze(-1), [(window_size),1], stride=[stride_size,1]).squeeze(-1).transpose(1,2)
spectrum = torch.fft.rfft(spectrum, dim=-1, norm='forward')
spectrum_mag = spectrum.real.pow(2).add(spectrum.imag.pow(2)).pow(0.5)
spectrum_mag = torch.fft.rfft(spectrum_mag, dim=-1, norm='ortho')
spectrum_mag[:,:,smoothing:].mul_(0)
spectrum_mag[:,:,smoothing-rolloff:smoothing].mul_(rolloff_window)
spectrum_mag = torch.fft.irfft(spectrum_mag, n=window_size//2+1, norm='ortho')
spectrum_mag.add_(-spectrum_mag.min())
spectrum_mag[:,:,-antialias:].mul_(antialias_window)
spectrum_mag.mul_(slope)

spectrum_mag = spectrum_mag.add(1e-7).log10().mul(20).clamp(threshold)
spectrum_mag.add_(-spectrum_mag.min())

target_shape = target.shape
target = F.unfold(target.transpose(0,1).unsqueeze(-1), [(window_size),1], stride=[stride_size,1]).squeeze(-1).transpose(1,2)
target = torch.fft.rfft(target, dim=-1, norm='forward')
target_mag = target.real.pow(2).add(target.imag.pow(2)).pow(0.5)#.mul(spectrum_mag)
target_mag.mul_(slope)
target_mag = target_mag.add(1e-7).log10().mul(20).add(-spectrum_mag*(1-1/ratio))
target_mag = 10**(target_mag/20).add(-1e-7)
target_mag.div_(slope)
target_mag[:,:,-antialias:].mul_(antialias_window)

target_phase = torch.atan2(target.real, target.imag)
target.real = target_mag*torch.sin(target_phase)
target.imag = target_mag*torch.cos(target_phase)

target = torch.fft.irfft(target, n=window_size, norm='forward').mul(coswin)
target = F.fold(target.transpose(1,2), [target_shape[-1], 1], [window_size,1], stride=[stride_size,1]).squeeze()
target.div_(target.abs().max())
wavfile.write(args.output, sample_rate_target, target.numpy().transpose().astype('float32'))
exit()
