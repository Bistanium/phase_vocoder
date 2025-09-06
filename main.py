from pathlib import Path
import sys
import tkinter
import tkinter.filedialog
import math
from fractions import Fraction

import numpy as np
import scipy
import resampy
import soundfile as sf
from numba import njit
from tqdm import tqdm


class Settings:
    # on:1 off:0
    resampling = 0
    # 0埋め倍数 2or4
    pad_multiple = 2


@njit(cache=True)
def WrapToPi(phases):
    PI = np.pi
    return (phases + PI) % (2 * PI) - PI


# スペクトログラム -> 振幅, 位相
def AnalyseComplexSpectrogram(spectrogram):
    magnitude_spectrogram = np.abs(spectrogram)
    phase_spectrogram = np.angle(spectrogram)

    return magnitude_spectrogram, phase_spectrogram


# 振幅, 位相 -> スペクトログラム
def ReconstructSpectrogram(magnitude_spectrogram, phase_spectrogram):
    spectrogram = magnitude_spectrogram * np.exp(1j * phase_spectrogram)

    return spectrogram


@njit(cache=True)
def DivideIntoRegions(data, n_min, n_max):
    frameSize = len(data)
    peakIndices = detect_peaks_linear_n(data, n_min, n_max)
    numPeaks = len(peakIndices)

    if numPeaks == 0:
        return np.array([[0, frameSize]], dtype=np.int64), np.array([0], dtype=np.int64)

    borders = np.empty(numPeaks + 1, dtype=np.int64)
    borders[0] = 0
    borders[-1] = frameSize

    if numPeaks > 1:
        for k in range(numPeaks - 1):
            i1 = peakIndices[k]
            i2 = peakIndices[k+1]
            
            # 区間内の最小値のインデックスを取得
            min_idx = i1
            min_val = data[i1]
            for i in range(i1+1, i2):
                if data[i] < min_val:
                    min_val = data[i]
                    min_idx = i
            borders[k+1] = min_idx

    regions = np.empty((numPeaks, 2), dtype=np.int64)
    for k in range(numPeaks):
        regions[k, 0] = borders[k]
        regions[k, 1] = borders[k+1]

    return regions, peakIndices


# 可変比較点数, parallel=True はnumbaのループ
@njit(cache=True)
def detect_peaks_linear_n(data, n_min=2, n_max=10):
    length = len(data)
    n_array = np.linspace(n_min, n_max, length).astype(np.int64)

    # 結果格納用（最悪ケースで全ビンがピークだったと仮定）
    peaks = np.empty(length, dtype=np.int64)
    count = 0

    for i in range(length):
        n = n_array[i]
        if i - n < 0 or i + n + 1 > length:
            continue

        center = data[i]
        left_max = 0
        for j in range(i - n, i):
            if data[j] > left_max:
                left_max = data[j]

        right_max = 0
        for j in range(i + 1, i + n + 1):
            if data[j] > right_max:
                right_max = data[j]
        
        if center > left_max and center > right_max:
            peaks[count] = i
            count += 1

    return peaks[:count]


def detect_transients(magnitude_prev, magnitude, threshold=0.2):
    n_bins = len(magnitude)  # 周波数ビン数
    # 線形重みを作成（低域: 0 → 高域: 1）
    weights = np.linspace(0, 1, n_bins)

    # 差分を計算（スペクトル差分）
    diff = magnitude - magnitude_prev
    positive_diff = np.maximum(diff, 0)

    # 重み適用
    weighted_diff = positive_diff * weights

    # RMS計算
    flux = np.sqrt(np.mean(np.square(weighted_diff)))

    # しきい値を使ってトランジェントを検出
    trunjent = flux > threshold

    return trunjent


def pair_by_closest(A, B, length, n_min=2, n_max=32):
    i = len(A) - 1
    length_B = len(B)
    result_A = np.empty(length_B, dtype=np.int64)  # B の長さで固定配列を作る
    result_B = np.empty(length_B, dtype=np.int64)
    idx = length_B - 1  # result に書き込むインデックス（逆順で埋める）
    t_array = np.linspace(n_min, n_max, length).astype(np.int64)

    for peak in reversed(B):
        while i - 1 >= 0 and abs(A[i - 1] - peak) <= abs(A[i] - peak):
            i -= 1

        threshold = t_array[peak]
        if i >= 0 and abs(A[i] - peak) <= threshold:
            result_A[idx] = A[i]
            result_B[idx] = peak
            i -= 1
        else:
            result_A[idx] = peak
            result_B[idx] = peak

        idx -= 1

    return result_A, result_B


@njit(cache=True)
def _pvm(regions, peak_indices, phase_spectrogram, new_phase_spectrogram, synthesis_phase, beta):
    len_peak_indices = len(peak_indices)
    for j in range(len_peak_indices):
        peak = peak_indices[j]
        region = regions[j]
        phase_current_peak = phase_spectrogram[peak]
        # synthesis_phaseはピークのときのみしか値がないため
        synthesis_phase_peak = synthesis_phase[j]

        # 領域内の全周波数ビンに対してピーク位相を使って位相補正
        start_idx = region[0]
        end_idx = region[1]
        for k in range(start_idx, end_idx):
            # 元のフレームの位相のピークとの差
            delta = WrapToPi(phase_spectrogram[k] - phase_current_peak)
            new_phase = synthesis_phase_peak + beta * delta
            # irfftで共役鏡像は作られる
            new_phase_spectrogram[k] = new_phase

    return new_phase_spectrogram


def phase_vocoder(prev_spectrogram, spectrogram, analysis_hop, synthesis_hop, prev_new_phase_spectrogram, before_peak_indices, alpha):
    prev_magnitude_spectrogram, prev_phase_spectrogram = AnalyseComplexSpectrogram(prev_spectrogram)
    magnitude_spectrogram, phase_spectrogram = AnalyseComplexSpectrogram(spectrogram)

    is_trunjent = detect_transients(prev_magnitude_spectrogram, magnitude_spectrogram)

    half_frame_size = len(phase_spectrogram)
    # half_n=n/2+1よりnを求める
    frame_size = (half_frame_size - 1) * 2

    # スペクトラムごとに領域とピークを検出
    if is_trunjent:
        n_min, n_max = 2, 32
    else:
        n_min, n_max = 2, 8
    regions, peak_indices = DivideIntoRegions(magnitude_spectrogram, n_min, n_max)

    # 前のピークが移動したとき、周波数的に近ければ同じピークとみなす。
    t_min, t_max = 2, 32
    before_peak_bin, peak_bin = pair_by_closest(before_peak_indices, peak_indices, half_frame_size, t_min, t_max)

    # 各周波数ビンに対応する角周波数
    omega = 2 * np.pi * peak_bin / frame_size
    # フレーム間の周波数の位相の変化と対応する周波数のビンから計算される位相の変化との差
    delta_phi = WrapToPi(phase_spectrogram[peak_bin] - prev_phase_spectrogram[before_peak_bin] - omega * analysis_hop)
    # omegaを補正して位相の変化がより正確になるようにする
    instantaneous_frequency = omega + delta_phi / analysis_hop
    # フレーム間がsynthesis_hopになったときにどのくらい位相が変化したか
    synthesis_phase = prev_new_phase_spectrogram[before_peak_bin] + instantaneous_frequency * synthesis_hop

    # すべてのビンを書き換えるのでemptyでも良い
    new_phase_spectrogram = np.empty(half_frame_size, dtype=np.float64)

    # betaは(alpha+2)/3 が良いとか
    beta = (alpha + 2) / 3

    # phase vocoder main
    new_phase_spectrogram = _pvm(regions, peak_indices, phase_spectrogram, new_phase_spectrogram, synthesis_phase, beta)

    new_spectrogram = ReconstructSpectrogram(magnitude_spectrogram, new_phase_spectrogram)

    return new_spectrogram, new_phase_spectrogram, peak_indices


def choose_file():
    # ファイル選択
    while True:
        fTyp = [("Audio File", ".wav"), ("wav", ".wav")]
        input_name = tkinter.filedialog.askopenfilename(filetypes = fTyp)
        input_path_obj = Path(input_name)

        if input_name:
            extension = input_path_obj.suffix
            if extension == ".wav":
                break
            else:
                sys.exit("ファイルが正しくありません")
        else:
            sys.exit()

    return input_path_obj


# Wave読み込み
def read_wav(file_path, normalize=False):
    # サウンドファイルを読み込む
    data, samplerate = sf.read(file_path, dtype=np.float64)

    if normalize:
        data = data / np.max(np.abs(data))

    # ステレオの場合、チャンネルを分離
    if data.ndim == 2:
        data_l = data[:, 0]  # 左
        data_r = data[:, 1]  # 右
    else:
        data_l = data
        data_r = None

    return data_l, data_r, samplerate


# wavファイル書き込み
def write_wav(file_path, data_l, data_r, samplerate):
    if data_r is not None:
        data = np.column_stack((data_l, data_r))
    else:
        data = data_l

    # 32bit浮動小数型に変換
    data = data.astype(np.float32)

    # ファイルを書き出す
    sf.write(file_path, data, samplerate, subtype='FLOAT')


def audio_to_segment(data_l, data_r, win_size, a_step, s_step, hop_a, hop_s, i):
    len_data = len(data_l)
    win_func = scipy.signal.windows.hann(win_size, sym=False)

    hop_a_start = a_step * i
    hop_s_start = s_step * i
    pad_size = win_size * (Settings.pad_multiple - 1)

    center = win_size // 2
    a_start = int(hop_a_start + 0.5)
    s_start = int(hop_s_start + 0.5)
    a_end = a_start + win_size
    if a_end < len_data:
        win_data = data_l[a_start:a_end] * win_func
        pad_win_data = np.pad(win_data, (0, pad_size), mode='constant', constant_values=0)
        ret_ls = np.roll(pad_win_data, -center)
    else:
        rest_size = win_size - len(data_l[a_start:])
        padded_l = np.pad(data_l[a_start:], (0, rest_size), mode='constant', constant_values=0)
        win_data = padded_l * win_func
        pad_win_data = np.pad(win_data, (0, pad_size), mode='constant', constant_values=0)
        ret_ls = np.roll(pad_win_data, -center)
    # フレーム間の大きさの記録
    hop_a[i] = int(hop_a_start + a_step + 0.5) - a_start
    hop_s[i] = int(hop_s_start + s_step + 0.5) - s_start

    if data_r is not None:
        if a_end < len_data:
            win_data = data_r[a_start:a_end] * win_func
            pad_win_data = np.pad(win_data, (0, pad_size), mode='constant', constant_values=0)
            ret_rs = np.roll(pad_win_data, -center)
        else:
            rest_size = win_size - len(data_r[a_start:])
            padded_r = np.pad(data_r[a_start:], (0, rest_size), mode='constant', constant_values=0)
            win_data = padded_r * win_func
            pad_win_data = np.pad(win_data, (0, pad_size), mode='constant', constant_values=0)
            ret_rs = np.roll(pad_win_data, -center)
    else:
        ret_rs = None

    return ret_ls, ret_rs, hop_a, hop_s


def segment_to_audio(data_ls, data_rs, win_size, hop_s, ret_l, ret_r, wsum, i):
    win_func = scipy.signal.windows.hann(win_size, sym=False)

    center = win_size // 2
    s_start = np.sum(hop_s[:i])
    s_end = s_start + win_size
    ret_l[s_start:s_end] += np.roll(data_ls, center)[:win_size] * win_func
    if data_rs is not None:
        ret_r[s_start:s_end] += np.roll(data_rs, center)[:win_size] * win_func
    else:
        ret_r = None
    wsum[s_start:s_end] += win_func ** 2

    return ret_l, ret_r, wsum


def normalize(ret_l, ret_r, wsum):
    epsilon = 1e-8
    pos = wsum > epsilon
    ret_l[pos] /= wsum[pos]
    if ret_r is not None:
        ret_r[pos] /= wsum[pos]

    return ret_l, ret_r


# リサンプリング
def resampling(left_audio, right_audio, fs, rate):
    original_fs = int(fs * 1e2 + 0.5)
    target_fs = int(fs * rate * 1e2 + 0.5)
    r = Fraction(original_fs, target_fs).limit_denominator()
    sr_orig = r.numerator
    sr_new = r.denominator

    resampled_left = resampy.resample(left_audio, sr_orig=sr_orig, sr_new=sr_new, filter='sinc_window', num_zeros=32)
    if right_audio is not None:
        resampled_right = resampy.resample(right_audio, sr_orig=sr_orig, sr_new=sr_new, filter='sinc_window', num_zeros=32)
    else:
        resampled_right = None

    return resampled_left, resampled_right


def pad(data, amount):
    if amount >= 0:
        new_data = np.pad(data, (amount, amount), mode='constant', constant_values=0)
    else:
        new_amout = -amount
        new_data = data[new_amout:-new_amout]

    return new_data

  
def calculate_rms(waveform):
    #二乗 → 平均 → 平方根
    return np.sqrt(np.mean(np.square(waveform)))


def main():
    # Wavファイル選択
    input_path_obj = choose_file()

    print()
    # キー指定
    is_log = False
    while True:
        try:
            pitchvalue = input('音程(-12 ~ 12) : ')
        except EOFError:
            sys.exit()
        pitchvalue = pitchvalue.strip() # 空白文字のみの入力は無視する
        if pitchvalue != '':
            if pitchvalue[0] == "l":
                is_log = True
                pitchvalue = pitchvalue[1:]
            try: # 数字かチェック
                pitch_float = float(pitchvalue)
                if pitch_float <= 48 and pitch_float >= -48: #ピッチの範囲指定
                    if is_log:
                        #log(2^(1/12))が底
                        pitch_float = math.log10(pitch_float) / 0.0250858329719984329344782412270
                    pitch_1 = round(1.059463094359295264561825294946 ** pitch_float, 12)
                    break
                else:
                    print("ピッチは-48以上48以下までです\n")
            except ValueError:
                print("数字を入力してください\n")

    # ピッチが+の時に名前に+を入れる
    if pitch_1 > 1:
        namepitch = "+" + str(pitch_float)
    else:
        namepitch = str(pitch_float)

    # ファイル名重複チェック
    base_name = input_path_obj.stem
    output_path_obj = input_path_obj.with_name(f"{base_name}_pv.key" + str(namepitch) + "_.wav")

    if output_path_obj.exists():
        base_name = output_path_obj.stem
        for i in range(1, 1000):
            output_path_obj = output_path_obj.with_name(f"{base_name} ({i}).wav")
            if not output_path_obj.exists():
                break
        else:
            sys.exit("ファイル名が重複しすぎているため作成できません")

    print("\ninput:", input_path_obj.name)
    print("output:", output_path_obj.name)
    print()

    # Wav読み込み
    data_l, data_r, fs = read_wav(input_path_obj, normalize=False)
    
    if data_r is not None:
        exist_r = True
    else:
        exist_r = False

    new_data_l = pad(data_l, 4096)
    if exist_r:
        new_data_r = pad(data_r, 4096)
    else:
        new_data_r = None

    # MID成分とSIDE成分分離
    if exist_r:
        if calculate_rms(new_data_l) >= calculate_rms(new_data_r):
            left_is_master = True
            data_master = new_data_l
            data_slave = new_data_r
        else:
            left_is_master = False
            data_master = new_data_r
            data_slave = new_data_l
    else:
        data_master = new_data_l
        data_slave = None

    del data_l, data_r, new_data_l, new_data_r
    
    if exist_r:
        print("left_is_master", left_is_master)

    # fsが44.1khz, 48kHz仮定
    win_size = 1024 * 2
    if pitch_1 >= 1.0:
        s_step = win_size // 4
        a_step = s_step / pitch_1
    else:
        a_step = win_size // 4
        s_step = a_step * pitch_1
    print("a_step:", a_step, "s_step:", s_step, "\n")
    
    # 配列の準備
    end_range = int((len(data_master) + a_step - 1) / a_step)
    hop_a = np.zeros(end_range, dtype=np.int64)
    hop_s = np.zeros(end_range, dtype=np.int64)

    audio_size = int(len(data_master) * pitch_1 + win_size)
    ret_l = np.zeros(audio_size, dtype=np.float64)
    if exist_r:
        ret_r = np.zeros(audio_size, dtype=np.float64)
    else:
        ret_r = None
    wsum = np.zeros(audio_size, dtype=np.float64)

    peak_indices = [0]
    # 最初のフレームの処理
    data_master_seg, data_slave_seg, hop_a, hop_s = audio_to_segment(data_master, data_slave, win_size, a_step, s_step, hop_a, hop_s, 0)
    ret_l, ret_r, wsum = segment_to_audio(data_master_seg, data_slave_seg, win_size, hop_s, ret_l, ret_r, wsum, 0)
    FFT_master_prev = scipy.fft.rfft(data_master_seg)
    new_phase_spectrogram = np.angle(FFT_master_prev)
    epsilon = 1e-8

    for i in tqdm(range(1, end_range), desc="フェーズボコーダ処理"):
        data_master_seg, data_slave_seg, hop_a, hop_s = audio_to_segment(data_master, data_slave, win_size, a_step, s_step, hop_a, hop_s, i)
        FFT_master = scipy.fft.rfft(data_master_seg)
        new_FFT_master, new_phase_spectrogram, peak_indices = phase_vocoder(FFT_master_prev, FFT_master, hop_a[i-1], hop_s[i-1], new_phase_spectrogram, peak_indices, pitch_1)
        IFFT_master = scipy.fft.irfft(new_FFT_master)
        FFT_master_prev = FFT_master
        if exist_r:
            FFT_slave = scipy.fft.rfft(data_slave_seg)
            mask = np.abs(FFT_master) > epsilon
            len_FFT_slave = len(FFT_slave)
            amp_ratio = np.zeros(len_FFT_slave, dtype=np.float64)
            phase_diff = np.zeros(len_FFT_slave, dtype=np.float64)
            amp_ratio[mask] = np.abs(FFT_slave[mask]) / np.abs(FFT_master[mask])
            phase_diff[mask] = np.angle(FFT_slave[mask]) - np.angle(FFT_master[mask])
            new_FFT_slave = amp_ratio * np.abs(new_FFT_master) * np.exp(1j * (new_phase_spectrogram + phase_diff))
            IFFT_slave = scipy.fft.irfft(new_FFT_slave)
        else:
            IFFT_slave = None
        ret_l, ret_r, wsum = segment_to_audio(IFFT_master, IFFT_slave, win_size, hop_s, ret_l, ret_r, wsum, i)

    ret_master, ret_slave = normalize(ret_l, ret_r, wsum)
    del ret_l, ret_r, wsum

    ret_master = pad(ret_master, -int(2048 * pitch_1))
    if exist_r:
        ret_slave = pad(ret_slave, -int(2048 * pitch_1))
    else:
        ret_slave = None

    # リサンプリング
    if Settings.resampling == 1:
        ret_master, ret_slave = resampling(ret_master, ret_slave, fs, 1/pitch_1)

    # ステレオ復元
    if exist_r:
        if left_is_master:
            ret_data_l = ret_master
            ret_data_r = ret_slave
        else:
            ret_data_l = ret_slave
            ret_data_r = ret_master
    else:
        ret_data_l = ret_master
        ret_data_r = None

    del ret_master, ret_slave

    #書き出し
    write_wav(output_path_obj, ret_data_l, ret_data_r, fs)


if __name__ == "__main__":
    main()