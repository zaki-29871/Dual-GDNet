call conda activate py38
cd GDNet/extensions
@RD /S /Q build
python ./setup.py install --user
pause