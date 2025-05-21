# A self-supervised multi-layer network of Rectified Spectral Units (ReSUs)
This repository contains the code to implement simulations in ``A self-supervised 
multi-layer network of Rectified Spectral Units (ReSUs)''

## Requirements
To install requirements, run the following command:
```bash
conda create -n resu python=3.12
conda activate resu
pip install -r requirements.txt
```

## Dataset
The natural scene dataset is from [Meyer et al 2014](https://pub.uni-bielefeld.de/record/2689637). The preprocessing of the raw images is decribed in https://elifesciences.org/articles/47579. To replicate the results in this paper, 
- Download the MATLAB code from https://github.com/ClarkLabCode/SynapticModel. Before running the code, the `rootDataPath` and `sceneSourcePath` directories must be set in the `SetConfiguration` function, or provided as arguments to that function. Then run the command `config = SetConfiguration(rootDataPath, sceneSourcePath)` to set the paths. Next, run `params = SetModelParameters` to get the parameters for the model.

- Download the Meyer 2014 dataset, which is a set of `.rar` archives, each containing a set of .mat files. Extract the .mat files from the archives and copy the resulting `.mat` files into a folder named `imageData` within the `sceneSourcePath` root directory.

- To convert the images into contrast profiles. First run `ConvertNaturalScenesToContrast(config,params)` function inside `utlil` directory. This will save the constrast as a .mat file within th root directory. Rename this .mat file as `contrast_scene.mat` and copy it to the `data` folder in this repository.

## Scripts
- **OU process**: to simulate an examplar  OU process, run the notebok `OU_process.ipynb`. The output will be saved in the `results` folder.

- **Gaussian processes**: to simulate Gaussian processes with different kernels, run the notebok `Gaussian_processes.ipynb`. The output will be saved in the `results` folder.

- **natural scene**: to simulate the filters derived from CCA of contrast profiles from natural scenes, run the notebok `filter_natural_scene.ipynb`. The output will be saved in the `results` folder.

- **ReSU**: to simulate a two-layer ReSU network on natural scenes, run the notebook `ReSU_motion_detection.ipynb`. The output will be saved in the `results` folder.


## Results
All the figures in the paper can be reproduced by running the notebok `figures_ReSU.ipynb`.
