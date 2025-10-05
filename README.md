# Phase Vocoder
Phase vocoders enable time stretching and pitch shifting.

[https://github.com/Bistanium/phase_vocoder](https://github.com/Bistanium/phase_vocoder)

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

<img width="350" height="152" alt="image" src="https://github.com/user-attachments/assets/aa58f653-6b8e-4f77-9d55-fbc37e45b137" />
<img width="347" height="150" alt="image" src="https://github.com/user-attachments/assets/b57d4368-34c7-433f-ab8a-cba775095d61" />

## Result
The waveform shown on top is the original, and the one below is the result of time-stretching the signal by a factor of 1.5.
<img width="1735" height="633" alt="image" src="https://github.com/user-attachments/assets/263a81f9-aedd-425a-9a30-44724468b9c9" />

## Reference
- フェーズボコーダーによるタイムストレッチ - C# [https://qiita.com/takayoshi1968/items/f7644df1f58dc1152bd9](https://qiita.com/takayoshi1968/items/f7644df1f58dc1152bd9)
- Time-scale Modification using the Phase Vocoder [https://old.iem.at/projekte/dsp/hammer/hammer.pdf](https://old.iem.at/projekte/dsp/hammer/hammer.pdf)
