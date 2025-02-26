import numpy as np
import pydub
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import interp1d
import librosa
import librosa.display

def load_and_resample(file, target_sample_rate):
    """오디오 파일을 로드하고 리샘플링하여 numpy 배열로 반환."""
    audio = pydub.AudioSegment.from_file(file)
    audio = audio.set_channels(1)  # Mono로 변환
    original_sample_rate = audio.frame_rate
    if original_sample_rate != target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)
        sample_rate = target_sample_rate
    else:
        sample_rate = original_sample_rate
    audio = audio.set_sample_width(2)  # 16-bit PCM
    audio_array = np.array(audio.get_array_of_samples())
    max_value = 32768  # 16-bit 최대값
    audio_float = audio_array / max_value
    return audio_float, sample_rate

def calculate_snr(signal, noise):
    """기본 SNR 계산."""
    power_signal = np.mean(signal**2)
    power_noise = np.mean(noise**2)
    if power_noise == 0:
        return float('inf')
    return 10 * np.log10(power_signal / power_noise)

def calculate_gz(audio):
    """WADA-SNR의 G_z 파라미터 계산."""
    audio = np.where(audio == 0, 1e-10, audio)  # 0 나누기 방지
    abs_audio = np.abs(audio)
    mean_abs = np.mean(abs_audio)
    mean_log_abs = np.mean(np.log(abs_audio))
    G_z = np.log(mean_abs) - mean_log_abs
    return G_z

def gz_to_snr(gz):
    """G_z 값을 SNR로 매핑 (테이블 기반)."""
    snr_values = np.arange(-20, 101)
    gz_values = np.array([0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 
                          0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272, 
                          0.41526426, 0.4178192, 0.42077252, 0.42452799, 0.42918886, 0.43510373, 
                          0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236, 
                          0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496, 
                          0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985, 
                          0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349, 
                          1.01047155, 1.0362095, 1.06136425, 1.08579312, 1.1094819, 1.13277995, 
                          1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313, 
                          1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935, 
                          1.3605727, 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619, 
                          1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938, 
                          1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915, 
                          1.5229097, 1.528578, 1.53389835, 1.5391211, 1.5439065, 1.54858517, 
                          1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767, 
                          1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477, 
                          1.5941969, 1.59693155, 1.599446, 1.60185011, 1.60408668, 1.60627134, 
                          1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074, 1.61688905, 
                          1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143, 
                          1.62632463, 1.6274027, 1.62842767, 1.62945532, 1.6303307, 1.63128026, 
                          1.63204102])
    snr_interp = interp1d(gz_values, snr_values, kind='quadratic', bounds_error=False, 
                          fill_value=(snr_values[0], snr_values[-1]))
    return float(snr_interp(gz))

def calculate_wada_snr(mixed_audio):
    """WADA-SNR 알고리즘으로 SNR 추정."""
    gz = calculate_gz(mixed_audio)
    snr = gz_to_snr(gz)
    return snr

def mix_audio(signal_array, noise_array, desired_snr, fixed_mode, snr_method='basic'):
    """오디오를 믹싱하고 SNR을 조정."""
    power_signal = np.mean(signal_array**2)
    power_noise = np.mean(noise_array**2)

    if fixed_mode == "Noise Fixed":
        desired_power_signal = power_noise * 10**(desired_snr / 10)
        scaling_factor = np.sqrt(desired_power_signal / power_signal)
        scaled_signal = signal_array * scaling_factor
        mixed_audio = scaled_signal + noise_array
        # actual_snr = desired_snr
    elif fixed_mode == "Signal Fixed":
        desired_power_noise = power_signal / 10**(desired_snr / 10)
        scaling_factor = np.sqrt(desired_power_noise / power_noise)
        scaled_noise = noise_array * scaling_factor
        mixed_audio = signal_array + scaled_noise
        # actual_snr = desired_snr
    else:  # None 모드
        mixed_audio = signal_array + noise_array

    if snr_method == 'wada':
        actual_snr = calculate_wada_snr(mixed_audio)
    else:
        actual_snr = calculate_snr(signal_array, noise_array)

    mixed_audio = np.clip(mixed_audio, -1, 1)  # 클리핑 처리
    return mixed_audio, actual_snr

def get_frequency_spectrum_figure(audio, sample_rate):
    """주파수 파워 스펙트럼 그래프 생성."""
    fft_result = np.fft.rfft(audio)
    power_spectrum = np.abs(fft_result)**2
    freq = np.fft.rfftfreq(len(audio), d=1/sample_rate)
    power_dB = 10 * np.log10(power_spectrum + 1e-10)
    fig, ax = plt.subplots()
    ax.set_title("Frequency Power Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.plot(freq, power_dB)
    plt.close(fig)
    return fig

def get_spectrogram_figure(audio, sample_rate):
    """스펙트로그램 그래프 생성."""
    stft = librosa.stft(audio)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log', ax=ax)
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.close(fig)
    return fig

def save_audio_to_wav(audio_array, sample_rate, output_file):
    """오디오 배열을 WAV 파일로 저장."""
    audio_int = (audio_array * 32767).astype(np.int16)
    wavfile.write(output_file, sample_rate, audio_int)