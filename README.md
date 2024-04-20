python -m venv imageClassification
source imageClassification/bin/activate

pip install ipykernel

python -m ipykernel install —name=imageClassification

jupyter kernelspec list


ls -l . | grep -v '^d' | wc -l
python renameDataFiles.py 