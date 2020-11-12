# FCNA
Locating Transcription factor binding sites (TFBSs) by fully convolutional network
## Requirements

+ Pytorch 1.1 
+ Python 3.6
+ CUDA 9.0
+ Python packages: biopython, sklearn

## Data preparation
(1) Downloading hg19.fa from http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/, and put it into /your path/hg19/.

(2) Pre-processing datasets.
+ Usage:
  ```
  bash process.sh <data path>
  ```

## Implementation 
**Running FCNA**
+ Usage: 
  ```
  bash run.sh <data path>
  ```
 
**Locating TFBSs**
+ Usage: 
  ```
  bash locate.sh <data path> <trained model path>
  ```
**Predicting motifs**
+ Usage: 
  ```
  bash motif.sh <data path> <trained model path>
  ```
  
**Refining the prediction performance**
+ Usage:  
  Firstly encoding the located regions;
  Secondly running FCNAR on them.
  
