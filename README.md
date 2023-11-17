# QG Validation
Repository for QG tagger validation and scale factors

This framework uses Python and coffea to read JMENano files and produce data and MC validation plots.
**Note: Currently the processing can only be done locally. LXPLUS implementation is yet to be done.**

The easiest solution is to install the required Python packages is to use [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).

Assuming that you are using Miniconda, initialze the setup by running:
```bash
source init_setup.sh
```
This command should install everything needed. This only needs to be done once. When starting a new session just run:
```bash
source activate_setup.sh
```
This script will set the correct shell environment variables.

# Quark-Gluon Likelihood (QGL)

The QGL value for each jet needs to be calculated separately, since the JMENano samples do not necessarily have the correct QGL training. The QGL trainings are located in `utils/QGL/`. Note that they are here in ROOT format, which needs to be converted to JSON with the `utils/QGL/convert_qgl_pdf_to_json.py` script.
```bash
cd $COFFEAHOME/utils/QGL/
python convert_qgl_pdf_to_json.py QGL_FILE.root
```
The JSON files are not directly contained in the repository because of their relatively large size. Running the file conversion script is quite slow, but it only needs to be done once per training.

# Running an analysis

There are multiple components involved in running the dijet/Z+jets analyses.

* The `$COFFEAHOME/scripts/run_qg_analysis.py` script is used for starting the processing.
* The Z+jets and dijet channel analyses are defined in `$COFFEAHOME/scripts/qg_processors.py`.
* The object selections are defined in `$COFFEAHOME/scripts/analysis_utils.py`.
* The processing script can be easily run with config files located in `$COFFEAHOME/scripts/config`.
* The list of files to be processed are contained as .txt files in `$COFFEAHOME/data`.

Certain necessary auxiliary files are located in `$COFFEAHOME/utils`. These include jet energy corrections, PU profiles, and QGL trainings, as well as the `utils.json` file that contains the MC sample cross sections, effective luminosities, and the numbers of generated events in each MC sample.

As of right now, the year-dependent variables are unfortunately hard-coded into the `qg_processors.py` script. This will be changed in the near-future.
