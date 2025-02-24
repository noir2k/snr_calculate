import streamlit as st
from audio_utils import (
    load_and_resample,
    mix_audio,
    get_frequency_spectrum_figure,
    get_spectrogram_figure,
    save_audio_to_wav
)

# 상수 정의
TARGET_SAMPLE_RATE = 44100

# Streamlit 인터페이스 설정
st.title("Audio Mixing and SNR Calculation")
st.write("Upload noise and signal audio files to mix them and calculate SNR.")

# 파일 업로드
noise_file = st.file_uploader("Upload noise audio file", type=["wav", "mp3"])
signal_file = st.file_uploader("Upload signal audio file", type=["wav", "mp3"])

# 사용자 입력
desired_snr = st.selectbox("Desired SNR (dB)", options=[0, 5, 10, 15, 20, 25], index=0)
fixed_mode = st.selectbox("Fixed Mode", ["None", "Noise Fixed", "Signal Fixed"])
snr_method = st.selectbox("SNR Estimation Method", ["Basic SNR", "WADA-SNR"])

# 처리 버튼
if st.button("Process"):
    if noise_file is None or signal_file is None:
        st.error("Please upload both noise and signal audio files.")
    else:
        try:
            # 오디오 로드 및 리샘플링
            noise_array, noise_sr = load_and_resample(noise_file, TARGET_SAMPLE_RATE)
            signal_array, signal_sr = load_and_resample(signal_file, TARGET_SAMPLE_RATE)

            if noise_sr != signal_sr:
                st.error("Sample rates do not match after resampling. Please check the files.")
            else:
                # 길이 맞춤
                min_length = min(len(noise_array), len(signal_array))
                noise_array = noise_array[:min_length]
                signal_array = signal_array[:min_length]

                # SNR 추정 방식 설정
                snr_method_choice = 'wada' if snr_method == "WADA-SNR" else 'basic'

                # 오디오 믹싱
                mixed_audio, actual_snr = mix_audio(
                    signal_array, noise_array, desired_snr, fixed_mode, snr_method=snr_method_choice
                )

                # SNR 표시
                if fixed_mode == "None":
                    st.write(f"Actual SNR ({snr_method}): {actual_snr:.2f} dB")
                else:
                    st.write(f"Target SNR: {desired_snr} dB")

                # 주파수 스펙트럼 그래프 생성 및 표시
                st.subheader("Frequency Power Spectrum")
                fig_spec = get_frequency_spectrum_figure(mixed_audio, TARGET_SAMPLE_RATE)
                st.pyplot(fig_spec)

                # 스펙트로그램 생성 및 표시
                st.subheader("Spectrogram")
                fig_spectrogram = get_spectrogram_figure(mixed_audio, TARGET_SAMPLE_RATE)
                st.pyplot(fig_spectrogram)

                # 오디오 파일 저장 및 다운로드 제공
                output_file = "mixed_audio.wav"
                save_audio_to_wav(mixed_audio, TARGET_SAMPLE_RATE, output_file)
                with open(output_file, "rb") as f:
                    st.download_button("Download mixed audio", f, file_name=output_file)

                # 오디오 재생
                st.subheader("Play Mixed Audio")
                st.audio(output_file, format='audio/wav')

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")