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
        self.muons = None

    def apply_sel(self):                
        self.muons = self.muons[self.sel]
        self.jets = self.jets[self.sel]
        
class SignalProcessor(processor.ProcessorABC):
    def __init__(self):        
        self._accumulator = {
            'cutflow' : processor.defaultdict_accumulator(int), 
            'weight': processor.column_accumulator(np.array([])),
            'Jet_PartonFlavour': processor.column_accumulator(np.array([])),
            'Jet_HadronFlavour': processor.column_accumulator(np.array([])),
            'Jet_pt': processor.column_accumulator(np.array([])),
            'Jet_eta': processor.column_accumulator(np.array([])),
            'Jet_phi': processor.column_accumulator(np.array([])),
            'Jet_mass': processor.column_accumulator(np.array([])),
            'Jet_qgl': processor.column_accumulator(np.array([])),
            'Jet_qgl_new': processor.column_accumulator(np.array([])),
            'Jet_qgl_axis2': processor.column_accumulator(np.array([])),
            'Jet_qgl_ptD': processor.column_accumulator(np.array([])),
            'Jet_qgl_mult': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavUDS': processor.column_accumulator(np.array([])),
            'Jet_particleNetAK4_QvsG': processor.column_accumulator(np.array([])),
            'Dimuon_mass': processor.column_accumulator(np.array([])),
            'Dimuon_pt': processor.column_accumulator(np.array([])),
            'Dimuon_eta': processor.column_accumulator(np.array([])),
            'nEvents': processor.value_accumulator(int),
            }

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
        jets = ak.zip(
            {
            "pt": df.Jet_pt,
            "eta": df.Jet_eta,
            "phi": df.Jet_phi,
            "mass": df.Jet_mass,
            "jet_id": df.Jet_jetId, #this one is bitwise for some reason
            "partonFlavour": df.Jet_partonFlavour if 'Jet_partonFlavour' in df.fields else ak.ones_like(df.Jet_jetId), #dummy flag for data
            "hadronFlavour": df.Jet_hadronFlavour if 'Jet_hadronFlavour' in df.fields else ak.ones_like(df.Jet_jetId),
            "cleanMask": df.Jet_cleanmask,
            "pu_id": df.Jet_puId,
            "qgl": df.Jet_qgl,
            "axis2": df.Jet_qgl_axis2,
            "ptD": df.Jet_qgl_ptD,
            "mult": df.Jet_qgl_mult,
            "rho": df.fixedGridRhoFastjetCentral,
            "deepFlavQG": df.Jet_btagDeepFlavUDS,
            "particleNetQG": df.Jet_particleNetAK4_QvsG,
            }, with_name="Momentum4D"
            )

        muons = ak.zip(
            {
            "pt": df.Muon_pt,
            "eta": df.Muon_eta,
            "phi": df.Muon_phi,
            "mass": df.Muon_mass,
            "charge": df.Muon_charge,
            "iso": df.Muon_pfRelIso04_all,
            "dxy": df.Muon_dxy,
            "dz": df.Muon_dz,            
            "isTight": df.Muon_tightId,
            "isMedium": df.Muon_mediumId,
            "isLoose": df.Muon_looseId
            }, with_name="Momentum4D"
            )

        electrons = ak.zip(
            {
            "pt": df.Electron_pt,
            "eta": df.Electron_eta,
            "phi": df.Electron_phi,
            "mass": df.Electron_mass,
            "charge": df.Electron_charge,
            "dxy": df.Electron_dxy,
            "iso": df.Electron_pfRelIso03_all,
            "dz": df.Electron_dz,
            "cutbased": df.Electron_cutBased
            }, with_name="Momentum4D"
            )

        if 'data' not in dataset:            
            lumi = np.ones(len(electrons), dtype=np.bool_) #dummy lumi mask for MC
        else:
            if 'COFFEADATA' not in os.environ:
                print("ERROR: Enviroment variables not set. Run setup.sh first!")
                sys.exit()
            json_path = os.path.join(os.environ['COFFEADATA'],'json')

            run = df["run"]
            lumiblock = df["luminosityBlock"]
            lumi_mask = LumiMask('{}/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'.format(json_path)) #FixMe: Add option for different years
            lumi = lumi_mask(run,lumiblock)
                    
        nEvents = len(df.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8)

        #input()
        # This keeps track of how many events there are, as well as how many of each object exist in this events.
        output['cutflow']['all events'] += nEvents
        output['cutflow']['all chs jets'] += ak.num(jets).to_numpy().sum()
        output['cutflow']['all muons'] += ak.num(muons).to_numpy().sum()
        output['cutflow']['all electrons'] += ak.num(electrons).to_numpy().sum()

        # # # # # # # # # # #
        # TRIGGER SELECTION #
        # # # # # # # # # # #
        
        trigger_mask = (df.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8==1)

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
        
        # Note that we don't want to down-select our jets (we wish to preserve jaggedness). The JaggedArray does this if more than one argument is provided for the mask. Since jets.mass > -1 is arbitrarily true, we add it to avoid this down-selection.

        muons, muon_selection = cutils.ObjSelection(muons,'muon',2017)
        electrons,_ = cutils.ObjSelection(electrons,'electron',2017)
        jets, _ = cutils.ObjSelection(jets,'jet',2017)

        # Now we want to make sure no jets are within 0.4 delta-R of any muon.
        # We cross jets with both fakeable muons, keeping it nested so it groups together by each jet and we can check if each jet is far enough away from every lepton. Conversely, the clean_mask should also do the same, but will keep this example for future reference
        cross_jmu = ak.cartesian([jets, muons], nested=True)
        check_jmu = ak.all((cross_jmu.slot0.deltaR(cross_jmu.slot1) > 0.4), axis=-1)
        jets = jets[(check_jmu)&(jets.mass>-1)]

        # This tells us how many objects pass each of our cuts.
        output['cutflow']['cleaned electrons'] += ak.num(electrons).to_numpy().sum()
        output['cutflow']['cleaned muons'] += ak.num(muons).to_numpy().sum()
        output['cutflow']['cleaned chs jets'] += ak.num(jets).to_numpy().sum()

        # # # # # # # # # #
        # EVENT SELECTION #
        # # # # # # # # # #

        jetMask = (ak.num(jets) >= 1) 
        electronMask = (ak.num(electrons) == 0)
        muonMask = (ak.num(muons) == 2)

        eventMask = (jetMask) & (electronMask) & (muonMask) & (lumi) & (trigger_mask) & (MET_mask)

        z_jets = Channel("ZJ")
        z_jets.muons = muons
        z_jets.jets = jets
        z_jets.sel = eventMask
        z_jets.apply_sel()

        # Z+Jets specific selection
        leading_jet_pt = z_jets.jets["pt"][:,0]
        z_charge = z_jets.muons[:,0].charge+z_jets.muons[:,1].charge

        z_cand = ak.combinations(z_jets.muons, 2)
        z_cand = z_cand.slot0 + z_cand.slot1

        cross_jz = ak.cartesian([z_jets.jets, z_cand])
        z_jet_deltaphi = cross_jz.slot0.deltaphi(cross_jz.slot1)

        mass_mask = (z_cand.mass > 71.2) & (z_cand.mass < 111.2)
        charge_mask = z_charge == 0
        deltaphi_mask = z_jet_deltaphi > 2.7
        deltaphi_mask = deltaphi_mask[:,0] # Only care about leading jet

        subleading_mask = np.asarray(z_jets.jets.mass[:,0] > -1) # Start with always true
        subleading_jet_pt = z_jets.jets["pt"][ak.num(z_jets.jets) >= 2, 1]
        z_pt = z_cand.pt[ak.num(z_jets.jets) >= 2, 0]
        subleading_mask[ak.num(z_jets.jets) >= 2] = np.asarray((subleading_jet_pt / z_pt) < 1.0)

        z_pt_mask = z_cand.pt > 15

        dilepton_mask = (ak.flatten(mass_mask)) & (charge_mask) & (deltaphi_mask) & (subleading_mask) & (ak.flatten(z_pt_mask))
        z_jets.sel = dilepton_mask
        z_jets.apply_sel()
        output['cutflow']['Dilepton selection'] += len(z_jets.jets)

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

        z_jets.jets["rho_bin"] = list(map(find_rho_bin, z_jets.jets["rho"][:,0]))
        z_jets.jets["eta_bin"] = list(map(find_eta_bin, np.abs(z_jets.jets["eta"][:,0])))
        z_jets.jets["pt_bin"] = list(map(find_pt_bin, z_jets.jets["pt"][:,0]))

        qgl_evaluator = self.qgl_evaluator
        def compute_jet_qgl(jet):
            jet_rho_bin = jet["rho_bin"]
            jet_eta_bin = jet["eta_bin"]
            jet_pt_bin = jet["pt_bin"]

            if (jet_rho_bin < 0 or jet_eta_bin < 0 or jet_pt_bin < 0):
                return -1.
            if jet["axis2"] <= 0:
                jet_axis2 = 0
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

        qgl_jets = z_jets.jets[:,0].to_list()
        z_jets.jets["qgl_new"] = list(map(compute_jet_qgl, qgl_jets))

        n_z_jets = len(z_jets.jets)
        if n_z_jets > 0:
            if 'data' not in dataset:
                weights = df['genWeight'][eventMask]
                weights = weights[dilepton_mask]
                weights = np.reshape(weights.to_numpy(), (n_z_jets,))
                jet_PartonFlavour = np.reshape(z_jets.jets.partonFlavour[:,0].to_numpy(), (n_z_jets,))
                jet_HadronFlavour = np.reshape(z_jets.jets.hadronFlavour[:,0].to_numpy(), (n_z_jets,))
            else:
                weights = np.ones(len(z_jets.jets))
                jet_PartonFlavour = np.ones(len(z_jets.jets))
                jet_HadronFlavour = np.ones(len(z_jets.jets))

            output['weight'] += processor.column_accumulator(weights)
            output['Jet_PartonFlavour'] += processor.column_accumulator(jet_PartonFlavour)
            output['Jet_HadronFlavour'] += processor.column_accumulator(jet_HadronFlavour)
            output['Jet_pt'] += processor.column_accumulator(np.reshape(z_jets.jets["pt"][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_eta'] += processor.column_accumulator(np.reshape(z_jets.jets["eta"][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_phi'] += processor.column_accumulator(np.reshape(z_jets.jets["phi"][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_mass'] += processor.column_accumulator(np.reshape(z_jets.jets["mass"][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl'] += processor.column_accumulator(np.reshape(z_jets.jets["qgl"][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_new'] += processor.column_accumulator(np.reshape(z_jets.jets["qgl_new"][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_axis2'] += processor.column_accumulator(np.reshape(z_jets.jets["axis2"][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_ptD'] += processor.column_accumulator(np.reshape(z_jets.jets["ptD"][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_mult'] += processor.column_accumulator(np.reshape(z_jets.jets["mult"][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavUDS'] += processor.column_accumulator(np.reshape(z_jets.jets["deepFlavQG"][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_particleNetAK4_QvsG'] += processor.column_accumulator(np.reshape(z_jets.jets["particleNetQG"][:,0].to_numpy(), (n_z_jets,)))
            dimuon_mass = np.reshape(z_cand[(dilepton_mask)].mass.to_numpy(), (n_z_jets,))
            dimuon_pt = np.reshape(z_cand[(dilepton_mask)].pt.to_numpy(), (n_z_jets,))
            dimuon_eta = np.reshape(z_cand[(dilepton_mask)].eta.to_numpy(), (n_z_jets,))
            output['Dimuon_mass'] += processor.column_accumulator(dimuon_mass)
            output['Dimuon_pt'] += processor.column_accumulator(dimuon_pt)
            output['Dimuon_eta'] += processor.column_accumulator(dimuon_eta)
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
    parser.add_option("--version", dest="version", type="string", default="", help="nametag to append to output file.")

    (opt, args) = parser.parse_args()
    samples = opt.samples
    
    if len(args) < 1:
        if 'COFFEADATA' not in os.environ:
            print("ERROR: Enviroment variables not set. Run setup.sh first!")
            sys.exit()
        #You can list the samples in a .txt file to be loaded. I will leave a simple one for example purposes
        if samples == 'data':
            file_name = os.path.join(os.environ['COFFEADATA'], 'DoubleMuon2017_test.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'DoubleMuon2017.txt')
            fout_name = 'data_{}.root'.format(opt.version)
        elif samples == 'mc':
            file_name = os.path.join(os.environ['COFFEADATA'], 'DYJets_test_b.txt')
            # file_name = os.path.join(os.environ['COFFEADATA'], 'DYJetsToLL_M-50_pythia8_UL17.txt')
            fout_name = 'mc_{}.root'.format(opt.version)
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

    job_executor = processor.FuturesExecutor(workers=opt.workers)
    run = processor.Runner(
                        executor = job_executor,
                        maxchunks = opt.maxchunk,
                        chunksize = opt.chunk,
                        )
    output = run(fileset, treename='Events', processor_instance=SignalProcessor())
    
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
        fout.mktree('Events', branch_dict)
        fout['Events'].extend(output)

        fout.mktree('Misc', {'nEvents' : np.uint})
        fout['Misc'].extend(misc)

if __name__ == "__main__":
    main()
