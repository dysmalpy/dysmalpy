set PYTHONPATH="C:\Program Files\dysmalpy"

call "C:\Program Files\anaconda3\Scripts\activate.bat" "C:\Program Files\anaconda3"

call conda activate dysmalpy_env

python "C:\Program Files\dysmalpy\dysmalpy\fitting_wrappers\dysmalpy_fit_single.py"

pause
