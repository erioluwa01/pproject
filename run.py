from src.preprocessing import preprocess_audio, save_audio
from src.denoise import denoise
from src.postprocessing import save_output
import os

os.makedirs(r"data\raw", exist_ok=True)

input_file = r"data\raw\Recording (7).m4a"
processed_file = r"data\raw\my_audio_resampled.wav"
output_file = r"data\raw\my_audio_denoised.wav"

audio, sr = preprocess_audio(input_file)
save_audio(audio, processed_file, sr)

denoised_audio = denoise(audio)
save_output(denoised_audio, output_file, sr)

print(f"Denoised audio saved to {output_file}")
