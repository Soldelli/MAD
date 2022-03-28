# Below commands will allow you to create an environment to run our code

Tested on Linux machine.

```bash
conda create --name vlg python=3.7 -y

conda activate vlg

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install -c anaconda pyyaml==5.3.1 h5py==2.10.0 scipy==1.1.0 -y

pip install yacs==0.1.8
pip install tensorboard wandb multimethod 

# Might be necessary 
conda deactivate
```

