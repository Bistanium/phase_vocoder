# Phase Vocoder
Phase Vocoder implemented in Python.  

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
There is a section in the program where the user can modify the `resampling` variable.  
- By default, `resampling` is set to 0, which performs time stretching.  
- To perform pitch shifting instead, set `resampling` to 1.

<img width="322" height="133" alt="resampling-0" src="https://github.com/user-attachments/assets/6302a244-ab95-4b52-a16d-d0b2ab7efaeb" />
<img width="318" height="129" alt="image" src="https://github.com/user-attachments/assets/0405bb9d-b68b-4e77-88a4-3665d49cf8be" />

## Result
The waveform of the original audio:
<img width="1737" height="339" alt="audio-x" src="https://github.com/user-attachments/assets/94107c79-134d-410e-a022-a97d59793487" />

The waveform after time-stretching:
<img width="1736" height="341" alt="audio-x_r1.5" src="https://github.com/user-attachments/assets/e5d5bb8e-762f-4b96-aa64-ca99a12d1b84" />

## Reference
- フェーズボコーダーによるタイムストレッチ - C# [https://qiita.com/takayoshi1968/items/f7644df1f58dc1152bd9](https://qiita.com/takayoshi1968/items/f7644df1f58dc1152bd9)
- Time-scale Modification using the Phase Vocoder [https://old.iem.at/projekte/dsp/hammer/hammer.pdf](https://old.iem.at/projekte/dsp/hammer/hammer.pdf)
