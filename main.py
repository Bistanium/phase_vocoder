from fractions import Fraction
from pathlib import Path
import sys
import tkinter
import tkinter.filedialog

from numba import njit
import numpy as np
import resampy
import scipy
import soundfile as sf
from tqdm import tqdm


class Settings:
    # enabled:1  disabled:0
    resampling = 0
    # 0埋め倍数  recommended 2 or 4
    pad_multiple = 2


# スペクトログラム -> 振幅, 位相
def analyse_complex_spectrogram(spectrogram):
    magnitude_spectrogram = np.abs(spectrogram)
    phase_spectrogram = np.angle(spectrogram)

    return magnitude_spectrogram, phase_spectrogram


# 振幅, 位相 -> スペクトログラム
def reconstruct_spectrogram(magnitude_spectrogram, phase_spectrogram):
    spectrogram = magnitude_spectrogram * np.exp(1j * phase_spectrogram)

    return spectrogram


@njit(cache=True)
def WrapToPi(phases):
    PI = np.pi
    return (phases + PI) % (2 * PI) - PI


@njit(cache=True)
def divide_into_regions(data, n_min, n_max):
    frameSize = len(data)
    peakIndices = detect_peaks(data, n_min, n_max)
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


@njit(cache=True)
def detect_peaks(data, n_min=2, n_max=10):
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


@njit(cache=True)
def pair_by_closest(A, B, length, n_min, n_max):
    mB = B.shape[0]

    t_array = np.linspace(n_min, n_max, length).astype(np.int64)

    t_vals = t_array[B]

    # searchsorted で left/right を一括取得
    left = np.searchsorted(A, B - t_vals, side='left')
    right = np.searchsorted(A, B + t_vals, side='right')

    counts_in_window = right - left
    counts_per_row = counts_in_window + 1  # フォールバック分
    max_cand = np.max(counts_per_row)

    # 候補行列
    cand = np.full((mB, max_cand), -1, dtype=A.dtype)

    # 候補生成
    for k in range(mB):
        l = left[k]
        r = right[k]
        L = r - l
        if L > 0:
            vals = A[l:r].copy()
            dists = np.abs(vals - B[k])
            # 安定ソート (選択ソート)
            for i in range(L):
                best = i
                for j in range(i + 1, L):
                    if dists[j] < dists[best]:
                        best = j
                    elif dists[j] == dists[best] and vals[j] < vals[best]:
                        best = j
                if best != i:
                    tmpv = vals[i]
                    vals[i] = vals[best]
                    vals[best] = tmpv
                    tmpd = dists[i]
                    dists[i] = dists[best]
                    dists[best] = tmpd
            for i in range(L):
                cand[k, i] = vals[i]
            cand[k, L] = B[k]
        else:
            cand[k, 0] = B[k]

    # ポインタと初期割り当て
    ptr = np.zeros(mB, dtype=np.int64)
    rA = np.empty(mB, dtype=A.dtype)
    for k in range(mB):
        rA[k] = cand[k, ptr[k]]

    # 重複解消
    while True:
        any_dup = False
        progressed = False
        processed = np.zeros(mB, dtype=np.uint8)

        for i in range(mB):
            if processed[i]:
                continue
            v = rA[i]
            idxs = np.empty(mB, dtype=np.int64)
            cnt = 0
            for j in range(mB):
                if rA[j] == v:
                    idxs[cnt] = j
                    cnt += 1
            for q in range(cnt):
                processed[idxs[q]] = 1

            if cnt > 1:
                any_dup = True
                keeper = idxs[0]
                best_dist = abs(B[keeper] - v)
                for q in range(1, cnt):
                    cur = idxs[q]
                    dist = abs(B[cur] - v)
                    if dist < best_dist:
                        keeper = cur
                        best_dist = dist
                    elif dist == best_dist and B[cur] > B[keeper]:
                        keeper = cur
                for q in range(cnt):
                    cur = idxs[q]
                    if cur == keeper:
                        continue
                    if ptr[cur] < counts_per_row[cur] - 1:
                        ptr[cur] += 1
                        rA[cur] = cand[cur, ptr[cur]]
                        progressed = True

        if not any_dup or not progressed:
            break

    return rA, B


@njit(cache=True)
def _epn(regions, peak_indices, phase_spectrogram, new_phase_spectrogram, synthesis_phase, beta):
    len_peak_indices = len(peak_indices)
    for j in range(len_peak_indices):
        peak = peak_indices[j]
        # ピークの占める領域
        region = regions[j]
        phase_spectrogram_peak = phase_spectrogram[peak]
        # synthesis_phaseはピークのときのみの値しかないため
        synthesis_phase_peak = synthesis_phase[j]

        # 領域内の全周波数ビンに対してピーク位相を使って位相補正
        start_idx = region[0]
        end_idx = region[1]
        for k in range(start_idx, end_idx):
            # 元のフレームの位相のピークとの差
            delta = WrapToPi(phase_spectrogram[k] - phase_spectrogram_peak)
            new_phase = synthesis_phase_peak + beta * delta
            # irfftに共役鏡像を作らせる
            new_phase_spectrogram[k] = new_phase

    return new_phase_spectrogram


def detect_transients(magnitude_prev, magnitude, threshold=0.2):
    n_bins = len(magnitude)  # 周波数ビン数
    # 線形重みを作成（低域: 0 → 高域: 1）
    weights = np.linspace(0, 1, n_bins)

    # 負の増加を除いた差分を計算
    diff = magnitude - magnitude_prev
    positive_diff = np.maximum(diff, 0)

    weighted_diff = positive_diff * weights

    # RMS計算
    flux = np.sqrt(np.mean(np.square(weighted_diff)))

    # しきい値を使ってトランジェントを検出
    trunjent = flux > threshold

    return trunjent


def edit_phase(prev_spectrogram, spectrogram, a_step, s_step, prev_new_phase_spectrogram, before_peak_indices, i):
    prev_magnitude_spectrogram, prev_phase_spectrogram = analyse_complex_spectrogram(prev_spectrogram)
    magnitude_spectrogram, phase_spectrogram = analyse_complex_spectrogram(spectrogram)

    is_trunjent = detect_transients(prev_magnitude_spectrogram, magnitude_spectrogram)

    half_frame_size = len(phase_spectrogram)
    # half_n=n/2+1よりnを求める
    frame_size = (half_frame_size - 1) * 2

    # スペクトラムごとに領域とピークを検出
    if is_trunjent:
        n_min, n_max = 2, 32
    else:
        n_min, n_max = 2, 8
    regions, peak_indices = divide_into_regions(magnitude_spectrogram, n_min, n_max)

    # 前のピークが移動したとき、周波数的に近ければ同じピークとみなす。
    t_min, t_max = 2, 32
    before_peak_bin, peak_bin = pair_by_closest(before_peak_indices, peak_indices, half_frame_size, t_min, t_max)

    # フレーム間の間隔の計算
    analysis_hop = int(a_step * i + 0.5) - int(a_step * (i - 1) + 0.5)
    synthesis_hop = int(s_step * i + 0.5) - int(s_step * (i - 1) + 0.5)

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
    alpha = s_step / a_step
    beta = (alpha + 2) / 3

    # ループ部分をnumbaで高速化
    new_phase_spectrogram = _epn(regions, peak_indices, phase_spectrogram, new_phase_spectrogram, synthesis_phase, beta)

    new_spectrogram = reconstruct_spectrogram(magnitude_spectrogram, new_phase_spectrogram)

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


def audio_to_segment(data_l, data_r, win_size, a_step, i):
    len_data = len(data_l)
    win_func = scipy.signal.windows.hann(win_size, sym=False)

    pad_size = win_size * (Settings.pad_multiple - 1)
    # center の値表
    # xxXyy000000 : 0
    # 000000xxXyy : win_size
    # 000xxXyy000 : -pad_size // 2
    # Xyy000000xx : win_size // 2
    center = win_size // 2
    start = int(a_step * i + 0.5)
    end = start + win_size
    if end < len_data:
        win_data = data_l[start:end] * win_func
        pad_win_data = np.pad(win_data, (0, pad_size), mode='constant', constant_values=0)
        ret_ls = np.roll(pad_win_data, -center)
    else:
        rest_size = win_size - len(data_l[start:])
        padded_l = np.pad(data_l[start:], (0, rest_size), mode='constant', constant_values=0)
        win_data = padded_l * win_func
        pad_win_data = np.pad(win_data, (0, pad_size), mode='constant', constant_values=0)
        ret_ls = np.roll(pad_win_data, -center)

    if data_r is not None:
        if end < len_data:
            win_data = data_r[start:end] * win_func
            pad_win_data = np.pad(win_data, (0, pad_size), mode='constant', constant_values=0)
            ret_rs = np.roll(pad_win_data, -center)
        else:
            rest_size = win_size - len(data_r[start:])
            padded_r = np.pad(data_r[start:], (0, rest_size), mode='constant', constant_values=0)
            win_data = padded_r * win_func
            pad_win_data = np.pad(win_data, (0, pad_size), mode='constant', constant_values=0)
            ret_rs = np.roll(pad_win_data, -center)
    else:
        ret_rs = None

    return ret_ls, ret_rs


def segment_to_audio(data_ls, data_rs, win_size, s_step, ret_l, ret_r, wsum, i):
    win_func = scipy.signal.windows.hann(win_size, sym=False)

    center = win_size // 2
    start = int(s_step * i + 0.5)
    end = start + win_size
    ret_l[start:end] += np.roll(data_ls, center)[:win_size] * win_func
    if data_rs is not None:
        ret_r[start:end] += np.roll(data_rs, center)[:win_size] * win_func
    else:
        ret_r = None
    wsum[start:end] += win_func ** 2

    return ret_l, ret_r, wsum


def normalize(ret_l, ret_r, wsum):
    epsilon = 1e-8
    pos = wsum > epsilon
    ret_l[pos] /= wsum[pos]
    if ret_r is not None:
        ret_r[pos] /= wsum[pos]

    return ret_l, ret_r


# リサンプリング
def resampling(left_audio, right_audio, sr, rate):
    original_sr = int(sr * 1e2 + 0.5)
    target_sr = int(sr * rate * 1e2 + 0.5)
    # 最大公約数で割っておいたほうが速いらしい
    r = Fraction(original_sr, target_sr).limit_denominator()
    sr_orig = r.numerator
    sr_new = r.denominator

    resampled_left_audio = resampy.resample(left_audio, sr_orig=sr_orig, sr_new=sr_new, filter='sinc_window', num_zeros=32)
    if right_audio is not None:
        resampled_right_audio = resampy.resample(right_audio, sr_orig=sr_orig, sr_new=sr_new, filter='sinc_window', num_zeros=32)
    else:
        resampled_right_audio = None

    return resampled_left_audio, resampled_right_audio


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


def phase_vocoder(left_audio, right_audio, pitch):
    if right_audio is not None:
        exist_r = True
    else:
        exist_r = False

    # 端のデータが復元できるように左右の端に0を付け足す
    padded_left_audio = pad(left_audio, 4096)
    if exist_r:
        padded_right_audio = pad(right_audio, 4096)
    else:
        padded_right_audio = None

    # データ大きさが大きいほうをメインとする
    if exist_r:
        if calculate_rms(padded_left_audio) >= calculate_rms(padded_right_audio):
            left_is_master = True
            master_audio = padded_left_audio
            slave_audio = padded_right_audio
        else:
            left_is_master = False
            master_audio = padded_right_audio
            slave_audio = padded_left_audio
    else:
        master_audio = padded_left_audio
        slave_audio = None

    del left_audio, right_audio, padded_left_audio, padded_right_audio

    if exist_r:
        # デバッグ用
        print("left_is_master", left_is_master)

    # srは44.1khz, 48kHz仮定
    win_size = 1024 * 2
    if pitch >= 1.0:
        s_step = win_size // 4
        a_step = s_step / pitch
    else:
        a_step = win_size // 4
        s_step = a_step * pitch
    # デバッグ用
    print("a_step:", a_step, "s_step:", s_step, "\n")
    
    # ループ回数
    loop_times = int((len(master_audio) + a_step - 1) / a_step)

    audio_size = int(len(master_audio) * pitch + win_size)
    new_master_audio = np.zeros(audio_size, dtype=np.float64)
    if exist_r:
        new_slave_audio = np.zeros(audio_size, dtype=np.float64)
    else:
        new_slave_audio = None
    wsum = np.zeros(audio_size, dtype=np.float64)

    peak_indices = np.array([0])
    # 最初のフレームの処理
    master_seg_audio, slave_seg_audio = audio_to_segment(master_audio, slave_audio, win_size, a_step, 0)
    new_master_audio, new_slave_audio, wsum = segment_to_audio(master_seg_audio, slave_seg_audio, win_size, s_step, new_master_audio, new_slave_audio, wsum, 0)
    prev_master_fft = scipy.fft.rfft(master_seg_audio)
    new_phase_spec = np.angle(prev_master_fft)

    epsilon = 1e-8
    for i in tqdm(range(1, loop_times), desc="フェーズボコーダ処理"):
        master_seg_audio, slave_seg_audio = audio_to_segment(master_audio, slave_audio, win_size, a_step, i)
        master_fft = scipy.fft.rfft(master_seg_audio)
        new_master_fft, new_phase_spec, peak_indices = edit_phase(prev_master_fft, master_fft, a_step, s_step, new_phase_spec, peak_indices, i)
        master_ifft = scipy.fft.irfft(new_master_fft)
        prev_master_fft = master_fft
        if exist_r:
            slave_fft = scipy.fft.rfft(slave_seg_audio)
            mask = np.abs(master_fft) > epsilon
            len_slave_fft = len(slave_fft)
            amp_ratio = np.zeros(len_slave_fft, dtype=np.float64)
            phase_diff = np.zeros(len_slave_fft, dtype=np.float64)
            amp_ratio[mask] = np.abs(slave_fft[mask]) / np.abs(master_fft[mask])
            phase_diff[mask] = np.angle(slave_fft[mask]) - np.angle(master_fft[mask])
            new_slave_fft = amp_ratio * np.abs(new_master_fft) * np.exp(1j * (new_phase_spec + phase_diff))
            slave_ifft = scipy.fft.irfft(new_slave_fft)
        else:
            slave_ifft = None
        new_master_audio, new_slave_audio, wsum = segment_to_audio(master_ifft, slave_ifft, win_size, s_step, new_master_audio, new_slave_audio, wsum, i)

    # wsumで割るだけ
    new_master_audio, new_slave_audio = normalize(new_master_audio, new_slave_audio, wsum)

    # 切り出し量はそれらしい量。int(4096 * pitch)ではなさそう？
    new_master_audio = pad(new_master_audio, -int(2048 * pitch))
    if exist_r:
        new_slave_audio = pad(new_slave_audio, -int(2048 * pitch))
    else:
        new_slave_audio = None

    if exist_r:
        if left_is_master:
            new_left_audio = new_master_audio
            new_right_audio = new_slave_audio
        else:
            new_left_audio = new_slave_audio
            new_right_audio = new_master_audio
    else:
        new_left_audio = new_master_audio
        new_right_audio = None

    del new_master_audio, new_slave_audio, wsum
    
    return new_left_audio, new_right_audio


# Wave読み込み
def read_wav(file_path):
    # サウンドファイルを読み込む
    data, samplerate = sf.read(file_path, dtype=np.float64)

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

    # データを正規化
    #data = data / np.max(np.abs(data))

    # 32bit浮動小数型に変換
    data = data.astype(np.float32)

    # 最大値と最小値に収める (浮動小数点誤差を考慮)
    #data = np.clip(data, -1, 1)

    # ファイルを書き出す
    sf.write(file_path, data, samplerate, subtype='FLOAT')


def main():
    # Wavファイル選択
    input_path_obj = choose_file()

    print()
    is_rate = False
    # キー指定
    while True:
        print("To specify a rate, please prefix the number with an 'r'.")
        try:
            value = input('Pitch (-48 ~ 48) : ')
        except EOFError:
            sys.exit()
        value = value.strip() # 空白文字のみの入力は無視する
        if value != '':
            if value[0] == "r":
                is_rate = True
                value = value[1:]
            try: # 数字かチェック
                value_float = float(value)
                if value_float <= 48 and value_float >= -48: #ピッチの範囲
                    if is_rate:
                        if 0 < value_float <= 4: #rateの範囲
                            pitch = value_float
                            break
                        else:
                            print("rateは0以上4以下までです\n")
                    else:
                        pitch = round(1.059463094359295264561825294946 ** value_float, 12)
                        break
                else:
                    print("ピッチは-48以上48以下までです\n")
            except ValueError:
                print("数字を入力してください\n")

    if is_rate:
        key_name = "rate" + str(value_float)
    else:
        if pitch > 1:
            key_name = "pitch+" + str(value_float)
        else:
            key_name = "pitch" + str(value_float)

    # ファイル名重複チェック
    base_name = input_path_obj.stem
    output_path_obj = input_path_obj.with_name(f"{base_name}_pv." + str(key_name) + "_.wav")

    if output_path_obj.exists():
        base_name = output_path_obj.stem
        for i in range(1, 1000):
            output_path_obj = output_path_obj.with_name(f"{base_name} ({i}).wav")
            if not output_path_obj.exists():
                break
        else:
            sys.exit("ファイル名が重複しすぎているため作成できません")

    # 確認用
    print("\ninput:", input_path_obj.name)
    print("output:", output_path_obj.name, "\n")

    # Wav読み込み
    data_l, data_r, sr = read_wav(input_path_obj)

    new_data_l, new_data_r = phase_vocoder(data_l, data_r, pitch)

    # リサンプリング
    if Settings.resampling == 1:
        new_data_l, new_data_r = resampling(new_data_l, new_data_r, sr, 1/pitch)

    #書き出し
    write_wav(output_path_obj, new_data_l, new_data_r, sr)


if __name__ == "__main__":
    main()
