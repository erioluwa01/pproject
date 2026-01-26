import torch
from denoiser import pretrained

# Load Facebook denoiser model
model = pretrained.dns64()  # or dns48 depending on your model

def denoise(audio):
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        denoised = model(audio_tensor)
    return denoised.squeeze().cpu().numpy()
