import uproot
import vector
import coffea
import correctionlib
import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import coffea_utils as cutils

from coffea import processor
from coffea.lookup_tools import extractor
from coffea.lumi_tools import LumiMask
from optparse import OptionParser

def find_qgl_bin(bins_dict, value):
    if (value < bins_dict[0]) or (value > bins_dict[len(bins_dict)-1]):
        return -1
    bin_num = 0
    while value > bins_dict[bin_num+1]:
        bin_num = bin_num+1
    return bin_num

class Channel():
    def __init__(self, name):
        self.name = name
        self.sel=None
        self.jets=None

    def apply_sel(self):                
        self.jets = self.jets[self.sel]
        
class SignalProcessor(processor.ProcessorABC):
    def __init__(self, puppi):        
        self._accumulator = {
            'cutflow' : processor.defaultdict_accumulator(int), 
            'weight': processor.column_accumulator(np.array([])),
            'Jet1_PartonFlavour': processor.column_accumulator(np.array([])),
            'Jet1_HadronFlavour': processor.column_accumulator(np.array([])),
            'Jet1_pt': processor.column_accumulator(np.array([])),
            'Jet1_eta': processor.column_accumulator(np.array([])),
            'Jet1_phi': processor.column_accumulator(np.array([])),
            'Jet1_mass': processor.column_accumulator(np.array([])),
            'Jet1_qgl_new': processor.column_accumulator(np.array([])),
            'Jet1_qgl_axis2': processor.column_accumulator(np.array([])),
            'Jet1_qgl_ptD': processor.column_accumulator(np.array([])),
            'Jet1_qgl_mult': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavUDS': processor.column_accumulator(np.array([])),
            'Jet1_particleNetAK4_QvsG': processor.column_accumulator(np.array([])),
            'Jet2_PartonFlavour': processor.column_accumulator(np.array([])),
            'Jet2_HadronFlavour': processor.column_accumulator(np.array([])),
            'Jet2_pt': processor.column_accumulator(np.array([])),
            'Jet2_eta': processor.column_accumulator(np.array([])),
            'Jet2_phi': processor.column_accumulator(np.array([])),
            'Jet2_mass': processor.column_accumulator(np.array([])),
            'Jet2_qgl_new': processor.column_accumulator(np.array([])),
            'Jet2_qgl_axis2': processor.column_accumulator(np.array([])),
            'Jet2_qgl_ptD': processor.column_accumulator(np.array([])),
            'Jet2_qgl_mult': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavUDS': processor.column_accumulator(np.array([])),
            'Jet2_particleNetAK4_QvsG': processor.column_accumulator(np.array([])),
            'LHE_HT': processor.column_accumulator(np.array([])),
            'LHE_HTIncoming': processor.column_accumulator(np.array([])),
            'nEvents': processor.value_accumulator(int),
            }

        self.puppi = puppi
        self._qgl_evaluator = correctionlib.highlevel.CorrectionSet.from_file('/ssd-home/kimmokal/QGval/QGValidation/scripts/pdfQG_AK4chs_13TeV_UL17_ghosts.corr.json')
        self._qgl_file = uproot.open("/ssd-home/kimmokal/QGval/pdfs/pdfQG_AK4chs_13TeV_UL17_ghosts.root")
    
    @property
    def accumulator(self):
        return self._accumulator

    @property
    def qgl_evaluator(self):
        return self._qgl_evaluator

    @property
    def qgl_file(self):
        return self._qgl_file
        
    def process(self, df):        
        output = self.accumulator
        dataset = df.metadata['dataset']

        vector.register_awkward()
        # Properties per object
        if self.puppi:
            jets = ak.zip(
                {
                "pt": df.JetPuppi_pt,
                "eta": df.JetPuppi_eta,
                "phi": df.JetPuppi_phi,
                "mass": df.JetPuppi_mass,
                "jet_id": df.JetPuppi_jetId, #this one is bitwise for some reason
                "partonFlavour": df.JetPuppi_partonFlavour if 'JetPuppi_partonFlavour' in df.fields else ak.ones_like(df.JetPuppi_jetId), #dummy flag for data
                "hadronFlavour": df.JetPuppi_hadronFlavour if 'JetPuppi_hadronFlavour' in df.fields else ak.ones_like(df.JetPuppi_jetId),
                "axis2": df.JetPuppi_qgl_axis2,
                "ptD": df.JetPuppi_qgl_ptD,
                "mult": df.JetPuppi_qgl_mult,
                "rho": df.fixedGridRhoFastjetCentral,
                "deepFlavQG": df.JetPuppi_btagDeepFlavUDS,
                "particleNetQG": df.JetPuppi_particleNetAK4_QvsG,
                "LHE_HT": df.LHE_HT if 'LHE_HT' in df.fields else ak.ones_like(df.JetPuppi_jetId),
                "LHE_HTIncoming" : df.LHE_HTIncoming if 'LHE_HTIncoming' in df.fields else ak.ones_like(df.JetPuppi_jetId),
                }, with_name="Momentum4D"
                )
        else:
            jets = ak.zip(
                {
                "pt": df.Jet_pt,
                "eta": df.Jet_eta,
                "phi": df.Jet_phi,
                "mass": df.Jet_mass,
                "jet_id": df.Jet_jetId, #this one is bitwise for some reason
                "partonFlavour": df.Jet_partonFlavour if 'Jet_partonFlavour' in df.fields else ak.ones_like(df.Jet_jetId), #dummy flag for data
                "hadronFlavour": df.Jet_hadronFlavour if 'Jet_hadronFlavour' in df.fields else ak.ones_like(df.Jet_jetId),
                "pu_id": df.Jet_puId,
                "axis2": df.Jet_qgl_axis2,
                "ptD": df.Jet_qgl_ptD,
                "mult": df.Jet_qgl_mult,
                "rho": df.fixedGridRhoFastjetCentral,
                "deepFlavQG": df.Jet_btagDeepFlavUDS,
                "particleNetQG": df.Jet_particleNetAK4_QvsG,
                "LHE_HT": df.LHE_HT if 'LHE_HT' in df.fields else ak.ones_like(df.Jet_jetId),
                "LHE_HTIncoming" : df.LHE_HTIncoming if 'LHE_HTIncoming' in df.fields else ak.ones_like(df.Jet_jetId),
                }, with_name="Momentum4D"
                )

        if 'data' not in dataset:            
            lumi = np.ones(len(jets), dtype=np.bool_) #dummy lumi mask for MC
        else:
            if 'COFFEADATA' not in os.environ:
                print("ERROR: Enviroment variables not set. Run setup.sh first!")
                sys.exit()
            json_path = os.path.join(os.environ['COFFEADATA'],'json')

            run = df["run"]
            lumiblock = df["luminosityBlock"]
            lumi_mask = LumiMask('{}/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'.format(json_path)) #FixMe: Add option for different years
            lumi = lumi_mask(run,lumiblock)

        nEvents = len(df.HLT_ZeroBias)

        # This keeps track of how many events there are, as well as how many of each object exist in this events.
        output['cutflow']['all events'] += nEvents
        output['cutflow']['all chs jets'] += ak.num(jets).to_numpy().sum()

        # # # # # # # # # # #
        # TRIGGER SELECTION #
        # # # # # # # # # # #

        trigger_mask = (df.HLT_ZeroBias==1)

        # # # # # # # #
        # MET FILTERS #
        # # # # # # # #

        MET_mask = (
                (df.Flag_goodVertices == 1) & (df.Flag_globalSuperTightHalo2016Filter == 1) & (df.Flag_HBHENoiseFilter==1) &
                (df.Flag_HBHENoiseIsoFilter==1) & (df.Flag_EcalDeadCellTriggerPrimitiveFilter==1) & (df.Flag_BadPFMuonFilter==1) &
                (df.Flag_BadPFMuonDzFilter==1) & (df.Flag_eeBadScFilter==1) & (df.Flag_ecalBadCalibFilter==1)
                )

        # # # # # # # # # # #
        # OBJECT SELECTIONS #
        # # # # # # # # # # #

        jets, _ = cutils.ObjSelection(jets,'jet_dijet',2017)

        # This tells us how many objects pass each of our cuts.
        output['cutflow']['cleaned chs jets'] += ak.num(jets).to_numpy().sum()

        # # # # # # # # # #
        # EVENT SELECTION #
        # # # # # # # # # #

        jetMask = (ak.num(jets) >= 2) 

        eventMask = (jetMask) & (lumi) & (trigger_mask) & (MET_mask)

        dijet = Channel("dijet")
        dijet.jets = jets
        dijet.sel = eventMask
        dijet.apply_sel()

        deltaphi_mask = np.abs(dijet.jets[:,0].deltaphi(dijet.jets[:,1])) > 2.7

        subsubleading_mask = np.asarray(dijet.jets.mass[:,0] > -1) # Start with always true

        leading_jet_pt = dijet.jets["pt"][ak.num(dijet.jets) >= 3, 0]
        subleading_jet_pt = dijet.jets["pt"][ak.num(dijet.jets) >= 3, 1]
        subsubleading_jet_pt = dijet.jets["pt"][ak.num(dijet.jets) >= 3, 2]
        subsubleading_mask[ak.num(dijet.jets) >= 3] = np.asarray(subsubleading_jet_pt < (leading_jet_pt + subleading_jet_pt)*0.5)

        leading_jet_pt = dijet.jets["pt"][:,0]
        subleading_jet_pt = dijet.jets["pt"][:,1]
        pt_balance_mask = np.abs((leading_jet_pt-subleading_jet_pt)/(leading_jet_pt+subleading_jet_pt)) < 0.7

        dijet_mask = (deltaphi_mask) & (subsubleading_mask) & (pt_balance_mask)
        dijet.sel = dijet_mask
        dijet.apply_sel()
        output['cutflow']['Dijet selection'] += len(dijet.jets)

        qgl_file = self.qgl_file
        qgl_rho_bins = qgl_file["rhoBins"].members["fElements"]
        qgl_eta_bins = qgl_file["etaBins"].members["fElements"]
        qgl_pt_bins = qgl_file["ptBins"].members["fElements"]

        qgl_rho_dict = {}
        qgl_eta_dict = {}
        qgl_pt_dict = {}

        for i in range(len(qgl_rho_bins)):
            qgl_rho_dict[i] = qgl_rho_bins[i]
        for i in range(len(qgl_eta_bins)):
            qgl_eta_dict[i] = qgl_eta_bins[i]
        for i in range(len(qgl_pt_bins)):
            qgl_pt_dict[i] = qgl_pt_bins[i]

        find_rho_bin = lambda x : find_qgl_bin(qgl_rho_dict, x)
        find_eta_bin = lambda x : find_qgl_bin(qgl_eta_dict, x)
        find_pt_bin = lambda x : find_qgl_bin(qgl_pt_dict, x)

        dijet.jets["rho_bin"] = list(map(find_rho_bin, dijet.jets["rho"][:,0]))
        dijet.jets["eta_bin"] = list(map(find_eta_bin, np.abs(dijet.jets["eta"][:,0])))
        dijet.jets["pt_bin"] = list(map(find_pt_bin, dijet.jets["pt"][:,0]))

        qgl_evaluator = self.qgl_evaluator
        def compute_jet_qgl(jet):
            jet_rho_bin = jet["rho_bin"]
            jet_eta_bin = jet["eta_bin"]
            jet_pt_bin = jet["pt_bin"]

            if (jet_rho_bin < 0 or jet_eta_bin < 0 or jet_pt_bin < 0):
                return -1.
            if jet["axis2"] <= 0:
                jet_axis2 = 0.
            else:
                jet_axis2 = -np.log(jet["axis2"])
            jet_mult = jet["mult"]
            jet_ptD = jet["ptD"]
            
            quark_likelihood = 1.
            gluon_likelihood = 1.

            for var in ["axis2", "ptD", "mult"]:
                quark_string = "{var_name}/{var_name}_quark_eta{bin1}_pt{bin2}_rho{bin3}".format(
                        var_name=var, bin1=jet_eta_bin, bin2=jet_pt_bin, bin3=jet_rho_bin)
                gluon_string = "{var_name}/{var_name}_gluon_eta{bin1}_pt{bin2}_rho{bin3}".format(
                        var_name=var, bin1=jet_eta_bin, bin2=jet_pt_bin, bin3=jet_rho_bin)
                
                if var == "axis2":
                    input_var = jet_axis2
                if var == "ptD":
                    input_var = jet_ptD
                if var == "mult":
                    input_var = float(jet_mult)

                var_quark_likelihood = qgl_evaluator[quark_string].evaluate(input_var)
                var_gluon_likelihood = qgl_evaluator[gluon_string].evaluate(input_var)

                if (var_quark_likelihood < 0) or (var_gluon_likelihood < 0):
                    return -1.

                quark_likelihood = quark_likelihood*var_quark_likelihood
                gluon_likelihood = gluon_likelihood*var_gluon_likelihood
        
            return round(quark_likelihood/(quark_likelihood+gluon_likelihood), 3)

        leading_jets = dijet.jets[:,0].to_list()
        subleading_jets = dijet.jets[:,1].to_list()

        jet1_qgl_new = list(map(compute_jet_qgl, leading_jets))
        jet2_qgl_new = list(map(compute_jet_qgl, subleading_jets))

        n_dijet = len(dijet.jets)
        if n_dijet > 0:
            if 'data' not in dataset:
                weights = df['genWeight'][eventMask]
                weights = weights[dijet_mask].to_numpy()
                jet1_PartonFlavour = dijet.jets["partonFlavour"][:,0].to_numpy()
                jet1_HadronFlavour = dijet.jets["hadronFlavour"][:,0].to_numpy()
                jet2_PartonFlavour = dijet.jets["partonFlavour"][:,1].to_numpy()
                jet2_HadronFlavour = dijet.jets["hadronFlavour"][:,1].to_numpy()
            else:
                weights = np.ones(len(dijet.jets))
                jet1_PartonFlavour = np.ones(len(dijet.jets))
                jet1_HadronFlavour = np.ones(len(dijet.jets))
                jet2_PartonFlavour = np.ones(len(dijet.jets))
                jet2_HadronFlavour = np.ones(len(dijet.jets))

            output['weight'] += processor.column_accumulator(np.reshape(weights, (n_dijet,)))
            output['Jet1_PartonFlavour'] += processor.column_accumulator(np.reshape(jet1_PartonFlavour, (n_dijet,)))
            output['Jet1_HadronFlavour'] += processor.column_accumulator(np.reshape(jet1_HadronFlavour, (n_dijet,)))
            output['Jet1_pt'] += processor.column_accumulator(np.reshape(dijet.jets["pt"][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_eta'] += processor.column_accumulator(np.reshape(dijet.jets["eta"][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_phi'] += processor.column_accumulator(np.reshape(dijet.jets["phi"][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_mass'] += processor.column_accumulator(np.reshape(dijet.jets["mass"][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_qgl_new'] += processor.column_accumulator(np.reshape(jet1_qgl_new, (n_dijet,)))
            output['Jet1_qgl_axis2'] += processor.column_accumulator(np.reshape(dijet.jets["axis2"][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_qgl_ptD'] += processor.column_accumulator(np.reshape(dijet.jets["ptD"][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_qgl_mult'] += processor.column_accumulator(np.reshape(dijet.jets["mult"][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavUDS'] += processor.column_accumulator(np.reshape(dijet.jets["deepFlavQG"][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_particleNetAK4_QvsG'] += processor.column_accumulator(np.reshape(dijet.jets["particleNetQG"][:,0].to_numpy(), (n_dijet,)))
            output['Jet2_PartonFlavour'] += processor.column_accumulator(np.reshape(jet2_PartonFlavour, (n_dijet,)))
            output['Jet2_HadronFlavour'] += processor.column_accumulator(np.reshape(jet2_HadronFlavour, (n_dijet,)))
            output['Jet2_pt'] += processor.column_accumulator(np.reshape(dijet.jets["pt"][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_eta'] += processor.column_accumulator(np.reshape(dijet.jets["eta"][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_phi'] += processor.column_accumulator(np.reshape(dijet.jets["phi"][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_mass'] += processor.column_accumulator(np.reshape(dijet.jets["mass"][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_qgl_new'] += processor.column_accumulator(np.reshape(jet2_qgl_new, (n_dijet,)))
            output['Jet2_qgl_axis2'] += processor.column_accumulator(np.reshape(dijet.jets["axis2"][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_qgl_ptD'] += processor.column_accumulator(np.reshape(dijet.jets["ptD"][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_qgl_mult'] += processor.column_accumulator(np.reshape(dijet.jets["mult"][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavUDS'] += processor.column_accumulator(np.reshape(dijet.jets["deepFlavQG"][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_particleNetAK4_QvsG'] += processor.column_accumulator(np.reshape(dijet.jets["particleNetQG"][:,1].to_numpy(), (n_dijet,)))
            output['LHE_HT'] += processor.column_accumulator(np.reshape(dijet.jets["LHE_HT"][:,1].to_numpy(), (n_dijet,)))
            output['LHE_HTIncoming'] += processor.column_accumulator(np.reshape(dijet.jets["LHE_HTIncoming"][:,1].to_numpy(), (n_dijet,)))
            output['nEvents'] += nEvents

        return output

    def postprocess(self, accumulator):
        return accumulator


def main():
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--samples", dest="samples", type="string", default='data', help="Specify which default samples to run [data/mc/all]. Default: data")
    parser.add_option("--base_folder", type="string", default="UL17", help="Folder which the data is stored. Default: %default") #
    parser.add_option("--workers", dest="workers", type="int", default=1, help="Number of workers to use. Default %default")
    parser.add_option("--chunk", dest="chunk", type="int", default=1e4, help="Chunk size. Default %default")
    parser.add_option("--maxchunk", dest="maxchunk", type="int", default=None, help="Maximum number of chunks. Default %default")
    parser.add_option("--out", dest="out", type="string", default="", help="Name for output file.")
    parser.add_option("--puppi", dest="puppi", action="store_true", default=False, help="Option for processing PUPPI jets. Default %default")
    (opt, args) = parser.parse_args()
    samples = opt.samples
    
    if len(args) < 1:
        if 'COFFEADATA' not in os.environ:
            print("ERROR: Enviroment variables not set. Run setup.sh first!")
            sys.exit()
        #You can list the samples in a .txt file to be loaded. I will leave a simple one for example purposes
        if samples == 'data':
            # file_name = os.path.join(os.environ['COFFEADATA'], 'ZeroBiasUL17_test.txt')
            file_name = os.path.join(os.environ['COFFEADATA'], 'ZeroBiasUL17.txt')
            fout_name = 'data_{}.root'.format(opt.out)
        elif samples == 'mc':
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_test.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_100to200_pythia8_UL17_test.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_binned_pythia8_UL17.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_50to100_pythia8_UL17.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_100to200_pythia8_UL17.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_200to300_pythia8_UL17.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_300to500_pythia8_UL17.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_500to700_pythia8_UL17.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_700to1000_pythia8_UL17.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_1000to1500_pythia8_UL17.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_1500to2000_pythia8_UL17.txt')
            file_name = os.path.join(os.environ['COFFEADATA'], 'QCD_HT_2000toInf_pythia8_UL17.txt')
            fout_name = 'mc_{}.root'.format(opt.out)
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
        fout_name = '{}.root'.format(opt.out)
    
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

    job_executor = processor.FuturesExecutor(workers=opt.workers)
    run = processor.Runner(
                        executor = job_executor,
                        maxchunks = opt.maxchunk,
                        chunksize = opt.chunk,
                        )
    output = run(fileset, treename='Events', processor_instance=SignalProcessor(opt.puppi))
    
    for flow in output['cutflow']:
        print(flow, output['cutflow'][flow])
    del output['cutflow']

    misc = {'nEvents' : [output['nEvents'].value]}
    del output['nEvents']

    if not os.path.exists(os.path.join(os.environ['COFFEAHOME'],"pods")):
        os.makedirs(os.path.join(os.environ['COFFEAHOME'],"pods"))
    
    branch_dict = {}
    for var in output:
        branch_dict[var] = np.float32
        output[var] = output[var].value

    with uproot.recreate(os.path.join(os.environ['COFFEAHOME'],"pods",fout_name)) as fout:
        fout.mktree("Events", branch_dict)
        fout["Events"].extend(output)

        fout.mktree('Misc', {'nEvents' : np.uint})
        fout['Misc'].extend(misc)
    
if __name__ == "__main__":
    main()
