# Below commands will allow you to create an environment to run our code

Tested on Linux machine.

```bash

conda create -n 0shotCLIP python=3.7 -y

conda activate 0shotCLIP 

conda install pytorch cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge tqdm terminaltables -y
conda install -c anaconda h5py -y

# Might be necessary 
conda deactivate
```

