# Phase Vocoder in Python

[https://github.com/Bistanium/phase_vocoder](https://github.com/Bistanium/phase_vocoder)

Phase vocoders enable time stretching and pitch shifting.

## Required Libraries
| Library   | Version |
|-----------|---------|
| numba     | 0.61.2  |
| numpy     | 2.2.6   |
| resampy   | 0.4.3   |
| scipy     | 1.16.1  |
| soundfile | 0.13.1  |
| tqdm      | 4.67.1  |

## How to Use
1. Run the `start.bat` file.
2. Select a wav file (any file you want to process).
3. Enter a value (time-stretch rate or target pitch).

## Additional note
- WAV files must have a sampling rate of either 44.1 kHz or 48 kHz.
- There is a section in the program where the user can modify the `resampling` variable.  
  - By default, `resampling` is set to 0, which performs time stretching.  
  - To perform pitch shifting instead, set `resampling` to 1.

<img width="340" height="131" alt="image" src="https://github.com/user-attachments/assets/a7a66c22-e4b4-4088-b03d-918c3bea3860" />
<img width="339" height="133" alt="image" src="https://github.com/user-attachments/assets/6104ac94-9fee-4a5a-b5e7-6513b68aa601" />

## Result
- The waveform shown on top is the original, and the one below is the result of time-stretching the signal by a factor of 1.5.
<img width="1737" height="637" alt="image" src="https://github.com/user-attachments/assets/95e906ee-4860-4677-a1b3-fadb147f2550" />

## Reference
- フェーズボコーダーによるタイムストレッチ - C# [https://qiita.com/takayoshi1968/items/f7644df1f58dc1152bd9](https://qiita.com/takayoshi1968/items/f7644df1f58dc1152bd9)
- Time-scale Modification using the Phase Vocoder [https://old.iem.at/projekte/dsp/hammer/hammer.pdf](https://old.iem.at/projekte/dsp/hammer/hammer.pdf)
