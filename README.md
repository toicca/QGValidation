# QGValidation
Repository for QG validation and scale factors

This repository uses coffea to read nanoAOD files and produce data and MC validation plots
To install the required packages first run:

```bash
source init_setup.sh
```
This command should install everything needed. If you already installed the required packages, you just need to source ```setup.sh``` everytime you start a new session

# Running the code

The script ```analysis_data.py``` performs the event selection with coffea. To test the script run:

```bash
cd $COFFEAHOME/scripts
python analysis_data.py --maxchunk 2 --version test root://cms-xrd-global.cern.ch//store/group/phys_jetmet/JMEnanoV01/UL17/DoubleMuon/Run2017B-09Aug2019_UL2017-v1_JMEnanoV1/201026_100435/0000/step1_NANO_10.root
```
If everything works, there should be a root files called ```test.root``` under the folder ```pods```. 
Run 

```bash
python analysis_data.py -h
```

for additional options. To check the definitions for the objects used during the selection, open the file ```coffea_utils.py```.

To instead load a list of files, first create a list under the folder ```data``` and load it from the main analysis code.

Finally, an options to run the code with ```parsl``` support is also given. The current code is set to be run in a ```slurm``` system and needs to be modified in case a different batch system is used.
