@echo off

SET CONDA_PATH=.\Miniconda3

REM     base    
CALL %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%

SET KMP_DUPLICATE_LIB_OK=TRUE

python app.py ^
 --transport virtualcam ^
 --max_session 2 ^
 --tts gpt-sovits ^
 --model wav2lip384 ^
 --avatar_id wav2lip_avatar1 ^
 --customvideo_config data/custom_config.json ^
 --TTS_SERVER http://127.0.0.1:9880 ^
 --REF_FILE "D:\AI\GPT-SoVITS\dw.wav" ^
 --REF_TEXT 真的是太超值了!家人们!千万不要错过呀!

cmd /k