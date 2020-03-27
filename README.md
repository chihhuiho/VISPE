# View Invariant Stochastic Prototype Embeddings (VISPE)
Code for Exploit Clues from Views Self Supervised and Regularized Learning for 3D Object Recognition. This work is published in CVPR 2020

# Usage
##  Evaluate pretrained model
1. Clone the project to directory DIR
```
git clone https://github.com/chihhuiho/VISPE.git
```
2. Initiate the conda environment
```
conda env create -f environment.yml -n VISPE
conda activate VISPE
```
3. Download the Modelnet dataset.
```
sh download.sh
```
4. Download the pretrained model from [here](https://drive.google.com/file/d/1PQV91Rpk6Ha3yGVHTSgqUiK5gQOD3Uvr/view?usp=sharing) and place it in the "model" folder
5. Evaluate the pretrained model
```
python main.py --load_pretrain --evaluate
```
##  Train your own model
1. Run our code from scratch
```
python main.py
```

# Ackowledgement
Please email to Chih-Hui (John) Ho (chh279@eng.ucsd.edu) if further issues are encountered.
