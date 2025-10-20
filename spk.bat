@echo off

SET CONDA_PATH=.\Miniconda3

REM 激活base环境
CALL %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%

SET KMP_DUPLICATE_LIB_OK=TRUE

python .\wav2lip\genavatar.py --video_path .\train_data\%1.mp4 --img_size 256 --face_det_batch_size 6


cmd /k