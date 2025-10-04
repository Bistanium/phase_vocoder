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
    # 1にするとピッチシフトモードになる
    # disabled:0  enabled:1
    resampling = 0
    # 0埋め倍数
    # 2, 4 推奨
    pad_multiple = 2
    # 解析に使うfftサイズ
    fft_size = 1024 * 2
    # %を表記しない
    # 50, 75, 87.5 など
    overlap_rate = 75
    # FFT時の窓関数にハン窓ではなくコサイン窓を使う
    # 低音域の再現性が上がるがトランジェントに弱くなる
    use_cosine_window = False
    # トランジェントで位相推定をリセットするか
    # 不完全なのでFalseの方が良い結果を示すことが多い
    phase_reset = False


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
        return np.array([[0, frameSize]], dtype=np.int64), np.array([np.argmax(data)], dtype=np.int64)

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
def detect_peaks(data, n_min=2, n_max=20):
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


def detect_transients(magnitude_prev, magnitude, threshold=0.125):
    prev_frame_sums = np.sum(magnitude_prev)
    frame_sums = np.sum(magnitude)
    if prev_frame_sums < 1e-2 or frame_sums < 1e-2:
        return False

    # 負の増加を除いた差分を計算
    diff = (magnitude - magnitude_prev) / prev_frame_sums
    positive_diff = np.maximum(diff, 0)

    # 線形重みを作成（低域: 0.5 → 高域: 1）
    weights = np.linspace(0.5, 1, len(magnitude))

    weighted_diff = positive_diff * weights

    flux = np.sum(weighted_diff)

    # しきい値を使ってトランジェントを検出
    transient = flux > threshold

    return transient


@njit(cache=True)
def _mpn(regions, peak_indices, phase_spectrogram, new_phase_spectrogram, synthesis_phase, betas):
    len_peak_indices = len(peak_indices)
    for j in range(len_peak_indices):
        peak = peak_indices[j]
        # ピークの占める領域
        region = regions[j]
        phase_spectrogram_peak = phase_spectrogram[peak]
        # synthesis_phaseはピークのときのみの値しかないため
        synthesis_phase_peak = synthesis_phase[j]

        beta = betas[j]

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


def modify_phase(prev_spectrogram, spectrogram, a_step, s_step, prev_new_phase_spectrogram, i, b_is_transient, bb_is_transient):
    prev_magnitude_spectrogram, prev_phase_spectrogram = analyse_complex_spectrogram(prev_spectrogram)
    magnitude_spectrogram, phase_spectrogram = analyse_complex_spectrogram(spectrogram)

    is_transient = detect_transients(prev_magnitude_spectrogram, magnitude_spectrogram)

    half_frame_size = len(phase_spectrogram)
    # half_n=n/2+1よりnを求める
    frame_size = (half_frame_size - 1) * 2
    # 位相をリセットするならTrue
    phase_reset = Settings.phase_reset

    # トランジェント検出の精度が悪いので真のトランジェントが得られそうな条件を用意
    true_transient = is_transient and b_is_transient and not bb_is_transient

    # スペクトラムごとに領域とピークを検出
    n_min, n_max = 2, 2
    t_regions, t_peak_indices = divide_into_regions(magnitude_spectrogram, n_min, n_max)

    if true_transient:
        if phase_reset:
            # 最も強いピークを位相を合わせる基準にする
            peak_indices = np.array([np.argmax(magnitude_spectrogram)], dtype=np.int64)
        else:
            # 高域はピーク判定をキツくしてトランジェント保持を行う
            n_min = 4
            n_max = int(64 * Settings.pad_multiple * Settings.fft_size / 1024)  # 良い感じに聞こえるような値
            regions, peak_indices = divide_into_regions(magnitude_spectrogram, n_min, n_max)
    else:
        regions, peak_indices = t_regions, t_peak_indices

    # フレーム間の間隔の計算
    analysis_hop = int(a_step * i + 0.5) - int(a_step * (i - 1) + 0.5)
    synthesis_hop = int(s_step * i + 0.5) - int(s_step * (i - 1) + 0.5)

    # 各周波数ビンに対応する角周波数
    omega = 2 * np.pi * peak_indices / frame_size
    # フレーム間の周波数の位相の変化と対応する周波数のビンから計算される位相の変化との差
    delta_phi = WrapToPi(phase_spectrogram[peak_indices] - prev_phase_spectrogram[peak_indices] - omega * analysis_hop)
    # omegaを補正して位相の変化がより正確になるようにする
    instantaneous_frequency = omega + delta_phi / analysis_hop
    # フレーム間がsynthesis_hopになったときにどのくらい位相が変化したか
    synthesis_phase = prev_new_phase_spectrogram[peak_indices] + instantaneous_frequency * synthesis_hop

    # すべてのビンを書き換えるのでemptyでも良い
    new_phase_spectrogram = np.empty(half_frame_size, dtype=np.float64)

    # betaはそれらしい値にしてある
    if np.max(magnitude_spectrogram) > 1e-6:
        alpha = 6 # 6~8
        betas = np.power(magnitude_spectrogram[peak_indices] / np.max(magnitude_spectrogram[peak_indices]), 1/alpha, dtype=np.float64)
    else:
        betas = np.full(len(peak_indices), 1, dtype=np.int64)

    # 前と今がトランジェントのとき、今の位相リセット
    if true_transient:
        if phase_reset:
            # トランジェントのとき位相差をそのまま用いる
            new_phase_spectrogram = synthesis_phase + phase_spectrogram - phase_spectrogram[peak_indices]
        else:
            betas = np.full(len(peak_indices), 1, dtype=np.int64)
            # 自然さを優先してある程度のトランジェント崩壊を許容
            new_phase_spectrogram = _mpn(regions, peak_indices, phase_spectrogram, new_phase_spectrogram, synthesis_phase, betas)
    else:
        # トランジェントではないとき
        new_phase_spectrogram = _mpn(regions, peak_indices, phase_spectrogram, new_phase_spectrogram, synthesis_phase, betas)

    new_spectrogram = reconstruct_spectrogram(magnitude_spectrogram, new_phase_spectrogram)

    return new_spectrogram, new_phase_spectrogram, is_transient, b_is_transient


def audio_to_segment(data_l, data_r, win_size, a_step, i):
    len_data = len(data_l)
    if Settings.use_cosine_window:
        win_func = scipy.signal.windows.cosine(win_size, sym=False)
    else:
        win_func = scipy.signal.windows.hann(win_size, sym=False)

    pad_size = win_size * (Settings.pad_multiple - 1)

    # abcd → abcdYyxX → cdYyxXab
    # np.rollを使ったほうが良い結果が得られる
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


def segment_to_audio(data_ls, data_rs, win_size, s_step, ret_l, ret_r, wsum, i, master_seg_audio):
    if Settings.use_cosine_window:
        win_func = scipy.signal.windows.cosine(win_size, sym=False)
    else:
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
    epsilon = 1e-6
    pos = wsum > epsilon
    ret_l[pos] /= wsum[pos]
    if ret_r is not None:
        ret_r[pos] /= wsum[pos]

    return ret_l, ret_r


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
    win_size = int(Settings.fft_size)
    overlap = int(100 / (100 - Settings.overlap_rate) + 0.5)
    if pitch >= 1.0:
        s_step = win_size // overlap
        a_step = s_step / pitch
    else:
        a_step = win_size // overlap
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

    epsilon = 1e-6
    b_is_transient = False
    bb_is_transient = False

    # 最初のフレームの処理
    master_seg_audio, slave_seg_audio = audio_to_segment(master_audio, slave_audio, win_size, a_step, 0)
    new_master_audio, new_slave_audio, wsum = segment_to_audio(
        master_seg_audio, slave_seg_audio, win_size, s_step, new_master_audio, new_slave_audio, wsum, 0, master_seg_audio
    )
    prev_master_fft = scipy.fft.rfft(master_seg_audio)
    new_phase_spec = np.angle(prev_master_fft)

    for i in tqdm(range(1, loop_times), desc="フェーズボコーダ処理"):
        master_seg_audio, slave_seg_audio = audio_to_segment(master_audio, slave_audio, win_size, a_step, i)
        master_fft = scipy.fft.rfft(master_seg_audio)
        new_master_fft, new_phase_spec, b_is_transient, bb_is_transient = modify_phase(
            prev_master_fft, master_fft, a_step, s_step, new_phase_spec, i, b_is_transient, bb_is_transient
        )
        master_ifft = scipy.fft.irfft(new_master_fft)
        prev_master_fft = master_fft
        if exist_r:
            slave_fft = scipy.fft.rfft(slave_seg_audio)
            # 振幅が小さすぎる周波数ビンは位相が不安定らしいので計算しない
            mask = np.abs(master_fft) > epsilon
            len_slave_fft = len(slave_fft)
            amp_ratio = np.zeros(len_slave_fft, dtype=np.float64)
            phase_diff = np.zeros(len_slave_fft, dtype=np.float64)
            amp_ratio[mask] = np.abs(slave_fft[mask]) / np.abs(master_fft[mask])
            phase_diff[mask] = np.angle(slave_fft[mask]) - np.angle(master_fft[mask])
            # 直交形式に変換
            new_slave_fft = amp_ratio * np.abs(new_master_fft) * np.exp(1j * (new_phase_spec + phase_diff))
            slave_ifft = scipy.fft.irfft(new_slave_fft)
        else:
            slave_ifft = None
        new_master_audio, new_slave_audio, wsum = segment_to_audio(
            master_ifft, slave_ifft, win_size, s_step, new_master_audio, new_slave_audio, wsum, i, master_seg_audio
        )

    # wsum(窓関数補正値)で割るだけ
    new_master_audio, new_slave_audio = normalize(new_master_audio, new_slave_audio, wsum)

    # 最初のゼロ埋め分の削除。正確な値がわからないがint(4096 * pitch)ではなさそう？
    new_master_audio = pad(new_master_audio, -int(2048 * pitch))
    if exist_r:
        new_slave_audio = pad(new_slave_audio, -int(2048 * pitch))
    else:
        new_slave_audio = None

    # leftとrightに割り振る
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


def choose_file():
    # ファイル選択
    while True:
        fTyp = [("Audio File", ".wav"), ("wav", ".wav")]
        input_name = tkinter.filedialog.askopenfilename(filetypes = fTyp)
        # pathlibはosより何かと便利らしい
        input_path_obj = Path(input_name)

        if input_name:
            # 拡張子のみ得る
            extension = input_path_obj.suffix
            if extension == ".wav":
                break
            else:
                sys.exit("ファイルが正しくありません")
        else:
            sys.exit()

    return input_path_obj


# Wav読み込み
def read_wav(file_path):
    # ファイルを読み込み
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

    # 64bitのままでもいいが多分過剰
    data = data.astype(np.float32)

    # 最大値と最小値に収める (丸め誤差を考慮)
    #data = np.clip(data, -1, 1)

    # ファイルを書き出す
    sf.write(file_path, data, samplerate, subtype='FLOAT')


def crate_output_path_obj(temp_output_path_obj, max=100):
    output_path_obj = temp_output_path_obj
    if output_path_obj.exists():
        # 拡張子以外を得る
        base_name = output_path_obj.stem
        for i in range(1, max):
            # 名前の後ろに数字を加える
            output_path_obj = output_path_obj.with_name(f"{base_name} ({i}).wav")
            # 同じファイル名のものがなければ終わり
            if not output_path_obj.exists():
                break
        else:
            sys.exit("ファイル名が重複しすぎているため作成できません")
    
    return output_path_obj


# リサンプリング
def resampling(left_audio, right_audio, sr, rate):
    original_sr = int(sr * 1e2 + 0.5)
    target_sr = int(sr * rate * 1e2 + 0.5)
    # 最大公約数で割っておいたほうが速いらしい
    r = Fraction(original_sr, target_sr).limit_denominator()
    sr_orig = r.numerator
    sr_new = r.denominator

    # 引数は処理が速くて音質が悪くないものを選んだ
    resampled_left_audio = resampy.resample(left_audio, sr_orig=sr_orig, sr_new=sr_new, filter='sinc_window', num_zeros=32)
    if right_audio is not None:
        resampled_right_audio = resampy.resample(right_audio, sr_orig=sr_orig, sr_new=sr_new, filter='sinc_window', num_zeros=32)
    else:
        resampled_right_audio = None

    return resampled_left_audio, resampled_right_audio


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
                            pitch = round(value_float, 12)
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
    temp_output_path_obj = input_path_obj.with_name(f"{base_name}_pv." + str(key_name) + "_.wav")
    # ファイル名に重複が起きないようにする
    output_path_obj = crate_output_path_obj(temp_output_path_obj)

    # 確認用
    print("\ninput:", input_path_obj.name)
    print("output:", output_path_obj.name, "\n")

    # Wav読み込み
    data_l, data_r, sr = read_wav(input_path_obj)

    new_data_l, new_data_r = phase_vocoder(data_l, data_r, pitch)

    # リサンプリング
    if Settings.resampling == 1:
        new_data_l, new_data_r = resampling(new_data_l, new_data_r, sr, 1/pitch)

    # 書き出し
    write_wav(output_path_obj, new_data_l, new_data_r, sr)


if __name__ == "__main__":
    main()
