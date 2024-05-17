python -m venv imageClassification
source imageClassification/bin/activate

pip install ipykernel

python -m ipykernel install —name=imageClassification

jupyter kernelspec list


ls -l . | grep -v '^d' | wc -l
python renameDataFiles.py 



python process_and_classify_image.py data/downloaded_images/Landscape/dark/dark18.jpg
python process_and_classify_image.py data/downloaded_images/Landscape/light/light33.jpg