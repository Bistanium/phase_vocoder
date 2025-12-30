# Phase Vocoder in Python

[https://github.com/Bistanium/phase_vocoder](https://github.com/Bistanium/phase_vocoder)

フェーズボコーダを使うとタイムストレッチとピッチシフトが可能になります。

## 必要なライブラリ
| Library   | Version |
|-----------|---------|
| numba     | 0.61.2  |
| numpy     | 2.2.6   |
| resampy   | 0.4.3   |
| scipy     | 1.16.1  |
| soundfile | 0.13.1  |
| tqdm      | 4.67.1  |

## 使い方
1. `start.bat` を実行します。
2. 処理したいwavファイルを選択します。
3. 目的のピッチの値やタイムストレッチ比を入力します。

## 追記
- このプログラムには `resampling` というユーザが好きに変更できるところがあります。  
  - デフォルトでは `resampling = 0` に設定されていて、この状態ではタイムストレッチが行われます。  
  - ピッチシフトを行いたい場合は `resampling = 1` に変更してください。

<img width="340" height="131" alt="image" src="https://github.com/user-attachments/assets/a7a66c22-e4b4-4088-b03d-918c3bea3860" />
<img width="339" height="133" alt="image" src="https://github.com/user-attachments/assets/6104ac94-9fee-4a5a-b5e7-6513b68aa601" />

## 結果
- 以下の二つの画像のうち、上にある画像は元の波形の画像で、下にある画像は1.5倍にタイムストレッチした波形の画像です。
<img width="1737" height="637" alt="image" src="https://github.com/user-attachments/assets/95e906ee-4860-4677-a1b3-fadb147f2550" />

## 参照
- フェーズボコーダーによるタイムストレッチ - C# [https://qiita.com/takayoshi1968/items/f7644df1f58dc1152bd9](https://qiita.com/takayoshi1968/items/f7644df1f58dc1152bd9)
- Time-scale Modification using the Phase Vocoder [https://old.iem.at/projekte/dsp/hammer/hammer.pdf](https://old.iem.at/projekte/dsp/hammer/hammer.pdf)
