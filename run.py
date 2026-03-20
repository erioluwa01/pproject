import os
from src.preprocessing import preprocess_audio, save_audio
from src.denoise import denoise
from src.postprocessing import save_output

DATA_DIR = os.path.join("data", "raw")
INPUT_FILE = os.path.join(DATA_DIR, "Recording (7).m4a")
PROCESSED_FILE = os.path.join(DATA_DIR, "my_audio_resampled.wav")
OUTPUT_FILE = os.path.join(DATA_DIR, "my_audio_denoised.wav")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    audio, sr = preprocess_audio(INPUT_FILE)
    save_audio(audio, PROCESSED_FILE, sr)

    denoised_audio = denoise(audio, sample_rate=sr)
    save_output(denoised_audio, OUTPUT_FILE, sr)

    print(f"Denoised audio saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()