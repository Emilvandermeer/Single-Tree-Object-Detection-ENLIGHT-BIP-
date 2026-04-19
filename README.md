# Single Tree Object Detection Enlight BIP

## Cloning into repository
This project is for a Faster R-CNN algorithm that predicts individual trees using bounding boxes.

Run commands will be given for use in the bash terminal with Linux.

To clone the repository do:
```
git clone https://github.com/Emilvandermeer/Single-Tree-Object-Detection-ENLIGHT-BIP-.git
```
Then navigate to the newly cloned repo with:
```
cd Single-Tree-Object-Detection-ENLIGHT-BIP
```
Navigate to the directory where all the model files are located:
```
cd BIP_proj
```
Sync environments so that all libary dependencies are handled:
```
uv sync
```
Activate the virtual environment:
```
source .venv/bin/activate
```
## Running the code - Training the model from scratch

To train the model from scratch, use the command:

```
uv run train.py
```
It will automatically downlaod and unzip all of the NEON tree dataset files if they are not already present. The total files downloaded to your device are ~8gb. Per epoch train loss, val loss and training time will be shown.

## Running the code - Testing the model we trained

If you do not want to run the model from scratch and only test it, you can use the following drive link which contains the model we trained:

```
https://drive.google.com/drive/u/0/folders/14__Ka8ONCYCeUuLhmcBx1FJGhbzpuV9T 
```

Drag and drop all of the files in the drive folder into
```
Single-Tree-Object-Detection-ENLIGHT-BIP-/BIP_proj/checkpoints
```
Then you can run all of the same tests we used with 
```
uv run test.py
```
It will automatically downlaod and unzip all of the NEON tree dataset files if they are not already present. The total files downloaded to your device are ~8gb.

Metrics will be outputted in the terminal.