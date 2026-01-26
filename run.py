from src.preprocessing import preprocess_audio, save_audio
from src.denoise import denoise
from src.postprocessing import save_output

input_file = "data/raw/my_audio.wav"
processed_file = "data/processed/my_audio_resampled.wav"
output_file = "outputs/my_audio_denoised.wav"

# Preprocess
audio, sr = preprocess_audio(input_file)
save_audio(audio, processed_file, sr)

# Denoise
denoised_audio = denoise(audio)

# Save final output
save_output(denoised_audio, output_file, sr)

print(f"Denoised audio saved to {output_file}")
