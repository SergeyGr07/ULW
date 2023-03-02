import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np



# Load the .wav file
y, sr = librosa.load('Whistlers.wav')

# Compute power spectrogram
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

# Display the spectrogram
plt.figure(figsize=(10, 5))
#Choosing a color of spectrogram
librosa.display.specshow(D, y_axis='linear', x_axis='time', cmap='inferno')
plt.colorbar(format='%+5.0f dB')
plt.title('Power spectrogram')
plt.show()

