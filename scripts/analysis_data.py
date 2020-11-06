from coffea import hist
import uproot
from coffea.util import awkward
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea.processor import run_parsl_job
from coffea.processor.parsl.parsl_executor import parsl_executor
from coffea.lookup_tools import extractor
from coffea.lumi_tools import LumiMask
import numpy as np
import matplotlib.pyplot as plt

import sys
import re
import h5py as h5
from optparse import OptionParser
import parsl
import os

from parsl.configs.local_threads import config
from parsl.providers import LocalProvider,SlurmProvider
from parsl.channels import LocalChannel,SSHChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname
import json, yaml
import copy
import coffea_utils as cutils




class Channel():
    def __init__(self, name):
        self.name = name
        self.sel=None
        self.jets=None
        self.muons = None

    def apply_sel(self):                
        self.muons = self.muons[self.sel]
        self.jets = self.jets[self.sel]
        

class SignalProcessor(processor.ProcessorABC):
    def __init__(self):        
        #sys.path.append(os.path.join(os.environ['COFFEAHOME'],'scripts'))    

        #Define the histograms to save
        dataset_axis = hist.Cat("dataset", "")                
            

        self.run_dict = {}
        dict_accumulator = {}
        
        list_accumulator = { #Output TTrees will contain the following variables
            'njets': processor.list_accumulator(),
            'weight': processor.list_accumulator(),
            'QGL': processor.list_accumulator(),
            
            'Dimuon_mass':processor.list_accumulator(),
            'Dimuon_pt':processor.list_accumulator(),
            'Jet_PartonFlavour':processor.list_accumulator(),
            'Jet_HadronFlavour':processor.list_accumulator(),
            'Jet_pt':processor.list_accumulator(),
            'Jet_eta':processor.list_accumulator(),
            'Jet_phi':processor.list_accumulator(),
            'Jet_mass':processor.list_accumulator(),
        }
        
        dict_accumulator['TTree'] = processor.dict_accumulator(list_accumulator)

        
        dict_accumulator['cutflow']= processor.defaultdict_accumulator(int) #Keep track on objects

        self._accumulator = processor.dict_accumulator( dict_accumulator )
    
    
    @property
    def accumulator(self):
        
        return self._accumulator
    
        
    def process(self, df):        
        output = self.accumulator.identity()
        dataset = df["dataset"]
        #Properties per object
        jets = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt=df['Jet_pt'].content,            
            #pt=df['Jet_pt_nom'].content,            
            eta=df['Jet_eta'].content,
            phi=df['Jet_phi'].content,
            #mass=df['Jet_mass_nom'].content,
            mass=df['Jet_mass'].content,
            jet_id=df['Jet_jetId'].content, #this one is bitwise for some reason
            partonFlavour= df['Jet_partonFlavour'].content if 'Jet_partonFlavour' in df else np.ones(df['Jet_jetId'].content.shape), #dummy flag for data
            hadronFlavour= df['Jet_hadronFlavour'].content if 'Jet_hadronFlavour' in df else np.ones(df['Jet_jetId'].content.shape), #dummy flag for data
            cleanMask = df['Jet_cleanmask'].content,
            pu_id = df['Jet_puId'].content,

            qgl=df['Jet_qgl'].content,

            )

        muons = JaggedCandidateArray.candidatesfromcounts(
            df['nMuon'],
            pt=df['Muon_pt'].content,
            eta=df['Muon_eta'].content,
            phi=df['Muon_phi'].content,
            mass=df['Muon_mass'].content,
            charge=df['Muon_charge'].content,
            iso=df['Muon_pfRelIso04_all'].content,
            dxy=df['Muon_dxy'].content,
            dz=df['Muon_dz'].content,            
            isTight=df['Muon_tightId'].content,
            isMedium=df['Muon_mediumId'].content,
            isLoose=df['Muon_looseId'].content,
            )

        electrons = JaggedCandidateArray.candidatesfromcounts(
            df['nElectron'],
            pt=df['Electron_pt'].content,
            eta=df['Electron_eta'].content,
            phi=df['Electron_phi'].content,
            mass=df['Electron_mass'].content,
            charge=df['Electron_charge'].content,
            dxy=df['Electron_dxy'].content,
            iso=df['Electron_pfRelIso03_all'].content,
            dz=df['Electron_dz'].content,
            cutbased=df['Electron_cutBased'].content,
            )

        if 'data' not in dataset:            
            lumi = np.ones(electrons.size,dtype=np.bool_) #dummy lumi mask for MC
        else:
            if 'COFFEADATA' not in os.environ:
                print("ERROR: Enviroment variables not set. Run setup.sh first!")
                sys.exit()
            json_path = os.path.join(os.environ['COFFEADATA'],'json')

            run = df["run"]
            lumiblock = df["luminosityBlock"]
            lumi_mask = LumiMask('{}/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'.format(json_path)) #FixMe: Add option for different years
            lumi = lumi_mask(run,lumiblock)
                    

            
        #input()
        # This keeps track of how many events there are, as well as how many of each object exist in this events.
        output['cutflow']['all events'] += muons.size
        output['cutflow']['all jets'] += jets.counts.sum()
        output['cutflow']['all muons'] += muons.counts.sum()
        output['cutflow']['all electrons'] += electrons.counts.sum()
        
        # # # # # # # # # # #
        # TRIGGER SELECTION #
        # # # # # # # # # # #
        
        #FixMe: year dependent flags

            
        trigger_mask = (df['HLT_IsoMu24']==1)  #IsoTrk was missing from my ntuples


        # # # # # # # # # # #
        # OBJECT SELECTIONS #
        # # # # # # # # # # #
        
        # Note that we don't want to down-select our jets (we wish to preserve jaggedness). The JaggedArray does this if more than one argument is provided for the mask. Since jets.mass > -1 is arbitrarily true, we add it to avoid this down-selection.

        muons,_ = cutils.ObjSelection(muons,'medium_muon',2017)        
        electrons,_ = cutils.ObjSelection(electrons,'electron',2017)        
        jets, _ = cutils.ObjSelection(jets,'jet',2017)        
                

        # Now we want to make sure no jets are within 0.4 delta-R of any muon.
        # We cross jets with both fakeable muons, keeping it nested so it groups together by each jet and we can check if each jet is far enough away from every lepton. Conversely, the clean_mask should also do the same, but will keep this example for future reference

        cross_mu = jets['p4'].cross(muons['p4'], nested=True)
        # delta_r is a built-in function for TLorentzVectors, so we'll make use of it.
        check_jmu = (cross_mu.i0.delta_r(cross_mu.i1) > 0.4).all()        
        jets = jets[(check_jmu)&(jets.mass>-1)]


        
    
        # This tells us how many objects pass each of our cuts.
        output['cutflow']['cleaned electrons'] += electrons.counts.sum()
        output['cutflow']['cleaned muons'] += muons.counts.sum()
        output['cutflow']['cleaned jets'] += jets.counts.sum()
        
        # # # # # # # # # #
        # EVENT SELECTION #
        # # # # # # # # # #
        

        jetMask = (jets.counts>=1) 
        eventMask = (jetMask) & (lumi) & (electrons.counts == 0) & (muons.counts == 2)  #Event selection. FixMe: Currently, only checking z+jets, so dijets need to be implemented

        z_jets = Channel("ZJ")
        z_jets.muons = muons
        z_jets.jets = jets
        z_jets.sel = eventMask
        z_jets.apply_sel()
        
        #Z+Jets specific selection
        
        leading_jet_pt = z_jets.jets.pt[:,0]
        z_cand = z_jets.muons.distincts()
        z_charge = z_jets.muons[:,0].charge*z_jets.muons[:,1].charge
            
        cross = z_jets.jets['p4'].cross(z_cand['p4'])                        
        z_jet_deltaphi = cross.i0.delta_phi(cross.i1)


        mass_mask = (z_cand.mass >70) & (z_cand.mass <110)
        charge_mask = z_charge<0
        deltaphi_mask = np.abs(z_jet_deltaphi) > 2.1
        deltaphi_mask = deltaphi_mask[:,0] #Only care about leading jet

        subleading_mask = z_jets.jets.mass[:,0]>-1  #Start with always true
        subleading_mask[z_jets.jets.counts>=2]= (z_jets.jets.pt[z_jets.jets.counts>=2,1]/z_cand.pt[z_jets.jets.counts>=2]) < 0.3 #Only apply cut if second jet exists

        
        dilepton_mask = (mass_mask.flatten()) & (charge_mask.flatten()) & (deltaphi_mask) & (subleading_mask.flatten())
        z_jets.sel = dilepton_mask
        z_jets.apply_sel()
        output['cutflow']['Dilepton selection'] += z_jets.jets.size

        if z_jets.jets.size>0:

            if 'data' not in dataset:
                weights = (df['genWeight'])[(eventMask)]
                weights = weights[(dilepton_mask)]
                jet_PartonFlavour = z_jets.jets.partonFlavour[:,0]
                jet_HadronFlavour = z_jets.jets.hadronFlavour[:,0]
            else:
                weights=np.ones(z_jets.jets.size)
                jet_PartonFlavour=np.ones(z_jets.jets.size)
                jet_HadronFlavour=np.ones(z_jets.jets.size)


                        

            #Saves information to TTrees
            output['TTree']['njets']+=z_jets.jets.counts.tolist()
            output['TTree']['weight']+=weights.tolist()
            output['TTree']['QGL']+=z_jets.jets.qgl[:,0].tolist()
            output['TTree']['Jet_PartonFlavour']+=jet_PartonFlavour.tolist()
            output['TTree']['Jet_HadronFlavour']+=jet_HadronFlavour.tolist()
            output['TTree']['Jet_pt']+=z_jets.jets.pt[:,0].tolist()
            output['TTree']['Jet_eta']+=z_jets.jets.eta[:,0].tolist()
            output['TTree']['Jet_phi']+=z_jets.jets.phi[:,0].tolist()
            output['TTree']['Jet_mass']+=z_jets.jets.mass[:,0].tolist()            
            output['TTree']['Dimuon_mass']+=z_cand[(dilepton_mask)].mass.tolist()
            output['TTree']['Dimuon_pt']+=z_cand[(dilepton_mask)].pt.tolist()
    
                        
        return output

    def postprocess(self, accumulator):
        return accumulator


        
parser = OptionParser(usage="%prog [opt]  inputFiles")

parser.add_option("--samples",dest="samples", type="string", default='data', help="Specify which default samples to run [data/mc/all]. Default: data")
parser.add_option("--base_folder", type="string", default="UL17", help="Folder which the data is stored. Default: %default") #
parser.add_option("--cpu",  dest="cpu", type="long", default=1, help="Number of cpus to use. Default %default")
parser.add_option("--chunk",  dest="chunk", type="long",  default=200000, help="Chunk size. Default %default")
parser.add_option("--maxchunk",  dest="maxchunk", type="long",  default=2e6, help="Maximum number of chunks. Default %default")
parser.add_option("--version",  dest="version", type="string",  default="", help="nametag to append to output file.")


#If using parsl for job submission
parser.add_option("--mem",  dest="mem", type="long", default=1100, help="Memory required in mb")
parser.add_option("--blocks",  dest="blocks", type="long", default=500, help="number of blocks. Default %default")
parser.add_option("-q", "--queue",  dest="queue", type="string", default="quick", help="Which queue to send the jobs. Default: %default")
parser.add_option("--walltime",  dest="walltime", type="string", default="0:59:50", help="Max time for job run. Default %default")
parser.add_option("--parsl",  dest="parsl", action="store_true",  default=False, help="Run with parsl. Default: False")


(opt, args) = parser.parse_args()

samples = opt.samples

if len(args) < 1:
    if 'COFFEADATA' not in os.environ:
        print("ERROR: Enviroment variables not set. Run setup.sh first!")
        sys.exit()
    #You can list the samples in a .txt file to be loaded. I will leave a simple one for example purposes
    if samples == 'data': 
        file_name = os.path.join(os.environ['COFFEADATA'],'SingleMuon2017RunC.txt')
        fout_name = 'data_{}.root'.format(opt.version)
    else:
        sys.error("ERROR: You must specify what kind of dataset you want to run [--samples]")
    
    print('Loading sets from file')    
    files = []
    with open(os.path.join(file_name),'r') as fread:
        files = fread.readlines()
        files = [x.strip() for x in files] 
        idx = np.arange(len(files))
        np.random.shuffle(idx)
        files = np.array(files)[idx]
                    
else:
    files = args
    fout_name = '{}.root'.format(opt.version)


fileset={}
data_names = cutils.data_names
for f in files:
    if any(name in f for name in data_names):
        name = 'data'
    else:
        name = 'mc'

    if name in fileset:
        fileset[name].append(f)
    else:
        fileset[name] = [f]

for sample in fileset:
    print('Files: {0}, sample name: {1}'.format(fileset[sample],sample))

    

#These options are batch dependent, in my case, I had a slurm batch system. if other job submission is used, just ignore everything inside the if statement
if opt.parsl:
    sched_options = '''
    #SBATCH --account=wn 
    #SBATCH --partition=gpu    
    #SBATCH --cpus-per-task=%d
    #SBATCH --mem=%d
    ''' % (opt.cpu,opt.mem) 

    slurm_htex = Config(
        executors=[
            HighThroughputExecutor(
                label="coffea_parsl_slurm",
                #worker_debug=True,
                address=address_by_hostname(),
                prefetch_capacity=0,  
                heartbeat_threshold=60,
                cores_per_worker=1,
                max_workers=opt.cpu,
                provider=SlurmProvider(
                    channel=LocalChannel(),
                    init_blocks=opt.blocks,
                    min_blocks = opt.blocks-20,                     
                    max_blocks=opt.blocks+20,
                    exclusive  = False,
                    parallelism=1,
                    nodes_per_block=1,
                    partition=opt.queue,
                    scheduler_options=sched_options,   # Enter scheduler_opt if needed
                    walltime=opt.walltime
                ),
            )
        ],
        initialize_logging=False,
        app_cache = True,
        retries=5,
        strategy=None,
    )

    dfk = parsl.load(slurm_htex)




    output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=SignalProcessor(),
                                      executor=processor.parsl_executor,
                                      executor_args={'config':None, 'retries':5,'tailtimeout':60,'xrootdtimeout':60},
                                      chunksize = opt.chunk
    )


else:
    output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=SignalProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers':opt.cpu},
                                      maxchunks =opt.maxchunk,
                                      chunksize = opt.chunk,
                                      
    )



for flow in output['cutflow']:
    print(flow, output['cutflow'][flow])



    

if not os.path.exists(os.path.join(os.environ['COFFEAHOME'],"pods")):
    os.makedirs(os.path.join(os.environ['COFFEAHOME'],"pods"))
    fout = uproot.recreate(os.path.join(os.environ['COFFEAHOME'],"pods",fout_name))
else:    
    fout = uproot.recreate(os.path.join(os.environ['COFFEAHOME'],"pods",fout_name))

for var in output:
    if var == 'cutflow':continue
    if 'TTree' in var:
        branch_dict = {}
        for branch in output[var]:
            branch_dict[branch] =  "float32"
            
        fout['Events'] = uproot.newtree(branch_dict)
        fout['Events'].extend(output[var])

if opt.parsl:
    parsl.dfk().cleanup()
    parsl.clear()
