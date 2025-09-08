# Phase_Vocoder
Phase Vocoder implemented in Python.  
This project was developed with reference to [フェーズボコーダーによるタイムストレッチ](https://qiita.com/takayoshi1968/items/f7644df1f58dc1152bd9)

## Required Libraries
- numba
- numpy
- resampy
- scipy
- soundfile
- tqdm

## How to Use
1. Run the `start.bat` file.
2. Select a wav file (any file you want to process).
3. Enter a value (time-stretch rate or target pitch).

## Additional note
There is a section in the program where the user can modify the `resampling` variable.  
- By default, `resampling` is set to 0, which performs time stretching.  
- To perform pitch shifting instead, set `resampling` to 1.

<img width="630" height="154" alt="resampling-0" src="https://github.com/user-attachments/assets/9d6d7ed5-74b9-4a1d-b522-b5bf632ffb9a" />
<img width="628" height="150" alt="resampling-1" src="https://github.com/user-attachments/assets/ef7e4cf0-5966-45ad-bf69-6739f8cbce04" />

## Result
The waveform of the original audio:
<img width="1737" height="339" alt="audio-x" src="https://github.com/user-attachments/assets/94107c79-134d-410e-a022-a97d59793487" />

The waveform after time-stretching:
<img width="1736" height="341" alt="audio-x_r1.5" src="https://github.com/user-attachments/assets/e5d5bb8e-762f-4b96-aa64-ca99a12d1b84" />
