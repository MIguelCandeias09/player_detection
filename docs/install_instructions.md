Install Anaconda Python 3.12
For all users


If anaconda navigator doesnt open, 
Delete folder C:\Users\vishe\AppData\Roaming\Python\Python312\

In navigator
create cuda environment
	with python 3.11.11


pip install opencv-python
pip install supervision
pip install ultralytics
pip install umap-learn
pip install transformers
pip install sentencepiece
pip install protobuf

conda install pytorch torchvision cudatoolkit -c pytorch


Upon error: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
rename C:\ProgramData\anaconda3\envs\cuda\Lib\site-packages\torch\liblibiomp5md.dll  to  libiomp5md.dll.BACK

conda activate cuda
