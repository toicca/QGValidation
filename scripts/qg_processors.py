import uproot
import vector
import coffea
import correctionlib
import sys
import os
import json
import gzip
import cachetools
import copy

import numpy as np
import awkward as ak
import analysis_utils as utils

from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.lookup_tools import extractor
from coffea.lumi_tools import LumiMask
from coffea.jetmet_tools import JetResolution, JetResolutionScaleFactor, CorrectedJetsFactory, JECStack
from correctionlib.convert import from_uproot_THx

def find_qgl_bin(bins_dict, value):
    if (value < bins_dict[0]) or (value > bins_dict[len(bins_dict)-1]):
        return -1
    bin_num = 0
    while value > bins_dict[bin_num+1]:
        bin_num = bin_num+1
    return bin_num

def compute_jet_qgl(jet, qgl_evaluator):
    jet_rho_bin = jet['rho_bin']
    jet_eta_bin = jet['eta_bin']
    jet_pt_bin = jet['pt_bin']

    if (jet_rho_bin < 0 or jet_eta_bin < 0 or jet_pt_bin < 0):
        return -1.

    jet_mult = jet['qgl_mult']
    jet_ptD = jet['qgl_ptD']
    jet_axis2 = jet['qgl_axis2']
    jet_axis2 = -np.log(jet_axis2) if jet_axis2 > 0 else 0. 
    
    quark_likelihood = 1.
    gluon_likelihood = 1.

    var_names = ['mult', 'ptD', 'axis2']
    for var in var_names:
        quark_string = '{var_name}/{var_name}_quark_eta{bin1}_pt{bin2}_rho{bin3}'.format(
                var_name=var, bin1=jet_eta_bin, bin2=jet_pt_bin, bin3=jet_rho_bin)
        gluon_string = '{var_name}/{var_name}_gluon_eta{bin1}_pt{bin2}_rho{bin3}'.format(
                var_name=var, bin1=jet_eta_bin, bin2=jet_pt_bin, bin3=jet_rho_bin)
        if var == 'mult':
            input_var = float(jet_mult)
        if var == 'ptD':
            input_var = jet_ptD
        if var == 'axis2':
            input_var = jet_axis2

        var_quark_likelihood = qgl_evaluator[quark_string].evaluate(input_var)
        var_gluon_likelihood = qgl_evaluator[gluon_string].evaluate(input_var)

        if (var_quark_likelihood < 0) or (var_gluon_likelihood < 0):
            return -1.

        quark_likelihood *= var_quark_likelihood
        gluon_likelihood *= var_gluon_likelihood

    return round(quark_likelihood/(quark_likelihood+gluon_likelihood), 5)

class Channel():
    def __init__(self, name):
        self.name = name
        self.sel=None
        self.jets=None
        self.muons=None

    def apply_sel(self):                
        self.jets = self.jets[self.sel]
        if self.muons is not None:
            self.muons = self.muons[self.sel]
        
class DijetProcessor(processor.ProcessorABC):
    def __init__(self, puppi, jes_up, jes_down, jer_up, jer_down):        
        self.output = {
            'cutflow' : processor.defaultdict_accumulator(int), 
            'weight': processor.column_accumulator(np.array([])),
            'PU_weight': processor.column_accumulator(np.array([])),
            'PU_weight_up': processor.column_accumulator(np.array([])),
            'PU_weight_down': processor.column_accumulator(np.array([])),
            'FSR_weight_up': processor.column_accumulator(np.array([])),
            'FSR_weight_down': processor.column_accumulator(np.array([])),
            'ISR_weight_up': processor.column_accumulator(np.array([])),
            'ISR_weight_down': processor.column_accumulator(np.array([])),
            'L1prefiring_weight': processor.column_accumulator(np.array([])),
            'L1prefiring_weight_up': processor.column_accumulator(np.array([])),
            'L1prefiring_weight_down': processor.column_accumulator(np.array([])),
            'gluon_weight': processor.column_accumulator(np.array([])),
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
            'Jet1_btagDeepFlavQG': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavG': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavB': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavCvB': processor.column_accumulator(np.array([])),
            'Jet1_btagDeepFlavCvL': processor.column_accumulator(np.array([])),
            'Jet1_particleNetAK4_QvsG': processor.column_accumulator(np.array([])),
            'Jet1_particleNetAK4_B': processor.column_accumulator(np.array([])),
            'Jet1_particleNetAK4_CvsB': processor.column_accumulator(np.array([])),
            'Jet1_particleNetAK4_CvsL': processor.column_accumulator(np.array([])),
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
            'Jet2_btagDeepFlavQG': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavG': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavB': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavCvB': processor.column_accumulator(np.array([])),
            'Jet2_btagDeepFlavCvL': processor.column_accumulator(np.array([])),
            'Jet2_particleNetAK4_QvsG': processor.column_accumulator(np.array([])),
            'Jet2_particleNetAK4_B': processor.column_accumulator(np.array([])),
            'Jet2_particleNetAK4_CvsB': processor.column_accumulator(np.array([])),
            'Jet2_particleNetAK4_CvsL': processor.column_accumulator(np.array([])),
            'alpha' : processor.column_accumulator(np.array([])),
            'fixedGridRhoFastjetAll': processor.column_accumulator(np.array([])),
            'LHE_HT': processor.column_accumulator(np.array([])),
            'LHE_HTIncoming': processor.column_accumulator(np.array([])),
            'Generator_binvar': processor.column_accumulator(np.array([])),
            'PV_npvs': processor.column_accumulator(np.array([])),
            'PV_npvsGood': processor.column_accumulator(np.array([])),
            'Pileup_nTrueInt': processor.column_accumulator(np.array([])),
            }

        self.puppi = puppi
        self.jes_up = jes_up
        self.jes_down = jes_down
        self.jer_up = jer_up
        self.jer_down = jer_down

        jer_mc_SF = 'Summer19UL17_JRV2_MC_SF_AK4PFchs'
        jer_mc_PtResolution = 'Summer19UL17_JRV2_MC_PtResolution_AK4PFchs'
        jec_mc_L1FastJet = 'Summer19UL17_V5_MC_L1FastJet_AK4PFchs'
        jec_mc_L2Relative = 'Summer19UL17_V5_MC_L2Relative_AK4PFchs'
        jec_mc_UncSources = 'RegroupedV2_Summer19UL17_V5_MC_UncertaintySources_AK4PFchs'

        coffea_base_path = os.environ['COFFEAHOME']
        jerc_extractor = extractor()
        jerc_extractor.add_weight_sets([
            '* * {}/utils/JERC/{}.txt'.format(coffea_base_path, jer_mc_SF),
            '* * {}/utils/JERC/{}.txt'.format(coffea_base_path, jer_mc_PtResolution),
            '* * {}/utils/JERC/{}.txt'.format(coffea_base_path, jec_mc_L1FastJet),
            '* * {}/utils/JERC/{}.txt'.format(coffea_base_path, jec_mc_L2Relative),
            '* * {}/utils/JERC/{}.junc.txt'.format(coffea_base_path, jec_mc_UncSources),
            ])
        jerc_extractor.finalize()
        jerc_evaluator = jerc_extractor.make_evaluator()

        self.JER_SF = JetResolutionScaleFactor(**{jer_mc_SF : jerc_evaluator[jer_mc_SF]})
        self.JER_PtResolution = JetResolution(**{jer_mc_PtResolution : jerc_evaluator[jer_mc_PtResolution]})

        jec_stack_names = [key for key in jerc_evaluator.keys()]
        jec_inputs = {name: jerc_evaluator[name] for name in jec_stack_names}
        jec_stack = JECStack(jec_inputs)

        name_map = jec_stack.blank_name_map
        name_map["JetPt"] = "pt"
        name_map["JetMass"] = "mass"
        name_map["JetEta"] = "eta"
        name_map["JetA"] = "area"
        name_map["ptGenJet"] = "pt_gen"
        name_map["ptRaw"] = "pt_raw"
        name_map["massRaw"] = "mass_raw"
        name_map["Rho"] = "rho"

        self.jet_corrector = CorrectedJetsFactory(name_map, jec_stack)

        self.qgl_evaluator = correctionlib.CorrectionSet.from_file('{}/utils/QGL/pdfQG_AK4chs_13TeV_UL17_ghosts.corr.json'.format(coffea_base_path))
        self.qgl_file = uproot.open('{}/utils/QGL/pdfQG_AK4chs_13TeV_UL17_ghosts.root'.format(coffea_base_path))
        self.pileup_weights = from_uproot_THx('{}/utils/pileup/PU_weights_HLT_ZeroBias_69200ub_pythia8_UL17.root:weight'.format(coffea_base_path), flow='clamp')
        self.pileup_weights_down = from_uproot_THx('{}/utils/pileup/PU_weights_HLT_ZeroBias_66000ub_pythia8_UL17.root:weight'.format(coffea_base_path), flow='clamp')
        self.pileup_weights_up = from_uproot_THx('{}/utils/pileup/PU_weights_HLT_ZeroBias_72400ub_pythia8_UL17.root:weight'.format(coffea_base_path), flow='clamp')
    
    def process(self, events):
        output = self.output
        dataset = events.metadata['dataset']

        if dataset in utils.data_names:
            filetype = 'data'
        elif dataset in utils.mc_names:
            filetype = 'mc'

        HLT = events.HLT

        if self.puppi:
            jets = events.JetPuppi
        else:
            jets = events.Jet

        if hasattr(events, 'GenJet'):
            gen_jets = events.GenJet

        if filetype == 'mc':            
            lumi_mask = np.ones(len(jets), dtype=np.bool_) # dummy lumi mask for MC
        else:
            run = events['run']
            lumiblock = events['luminosityBlock']
            goldenJSON_path = os.path.join(os.environ['COFFEAHOME'],'data','json')
            lumi_path = '{}/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'.format(goldenJSON_path) # FixMe: Add option for different years
            lumi_mask = LumiMask(lumi_path)(run, lumiblock) 

        nEvents = len(HLT.ZeroBias)
        event_rho = events.fixedGridRhoFastjetAll
        PV_npvs = events.PV.npvs
        PV_npvsGood = events.PV.npvsGood
        Pileup_nTrueInt = events.Pileup.nTrueInt if 'nTrueInt' in events.Pileup.fields else ak.ones_like(PV_npvs)
        if 'Generator' in events.fields:
            Generator_binvar = events.Generator.binvar if 'binvar' in events.Generator.fields else ak.ones_like(PV_npvs)
        if 'LHE' in events.fields:
            LHE_HT = events.LHE.HT if 'HT' in events.LHE.fields else ak.ones_like(PV_npvs)
            LHE_HTIncoming = events.LHE.HTIncoming if 'HTIncoming' in events.LHE.fields else ak.ones_like(PV_npvs)
            LHEPart_pdgId = events.LHEPart.pdgId if 'pdgId' in events.LHEPart.fields else ak.ones_like(PV_npvs)
            LHEPart_status = events.LHEPart.status if 'status' in events.LHEPart.fields else ak.ones_like(PV_npvs)
        PSWeight = events.PSWeight if 'PSWeight' in events.fields else ak.ones_like(PV_npvs)

        # print('fsr:murfac=0.5: {}'.format(PSWeight[0][2]))
        # print('fsr:murfac=2.0: {}'.format(PSWeight[0][3]))
        # print('isr:murfac=0.5: {}'.format(PSWeight[0][24]))
        # print('isr:murfac=2.0: {}'.format(PSWeight[0][25]))

        # Cutflow to keep track of number of events and jets
        output['cutflow']['all events'] += nEvents
        output['cutflow']['all jets'] += ak.num(jets).to_numpy().sum()

        # # # # # # # # # # #
        # TRIGGER SELECTION #
        # # # # # # # # # # #

        trigger_mask = (HLT.ZeroBias==1)

        # # # # # # # #
        # MET FILTERS #
        # # # # # # # #

        MET_mask = (
                (events.Flag.goodVertices == 1) & (events.Flag.globalSuperTightHalo2016Filter == 1) & (events.Flag.HBHENoiseFilter==1) &
                (events.Flag.HBHENoiseIsoFilter==1) & (events.Flag.EcalDeadCellTriggerPrimitiveFilter==1) & (events.Flag.BadPFMuonFilter==1) &
                (events.Flag.BadPFMuonDzFilter==1) & (events.Flag.eeBadScFilter==1) & (events.Flag.ecalBadCalibFilter==1)
                )

        # # # # # # # # # # #
        # OBJECT SELECTIONS #
        # # # # # # # # # # #

        jets, _ = utils.ObjSelection(jets,'jet',2017)
        output['cutflow']['cleaned jets'] += ak.num(jets).to_numpy().sum()

        # # # # # # # # # #
        # EVENT SELECTION #
        # # # # # # # # # #

        jet_mask = (ak.num(jets) >= 2)

        event_mask = jet_mask & lumi_mask & trigger_mask & MET_mask

        # TO-DO: Change implementation of cuts to use coffea.PackedSelection, available in newer Coffea version

        if 'GenVtx' in events.fields:
            vtx_mask = np.abs(events.GenVtx.z - events.PV.z) < 0.2
            event_mask = event_mask & vtx_mask

        event_rho = event_rho[event_mask]
        PV_npvs = PV_npvs[event_mask]
        PV_npvsGood = PV_npvsGood[event_mask]
        Pileup_nTrueInt = Pileup_nTrueInt[event_mask]
        Generator_binvar = Generator_binvar[event_mask]
        LHE_HT = LHE_HT[event_mask]
        LHE_HTIncoming = LHE_HTIncoming[event_mask]
        LHEPart_pdgId = LHEPart_pdgId[event_mask]
        LHEPart_status = LHEPart_status[event_mask]
        PSWeight = PSWeight[event_mask]

        dijet = Channel('dijet')
        dijet.jets = jets
        dijet.gen_jets = gen_jets
        dijet.sel = event_mask
        dijet.apply_sel()

        # Apply JECs to MC jets
        if filetype == 'mc':
            # Impose the requirement delta_r < 0.2 between reco jets and gen jets for the JER "scaling method"
            genjet_delta_r = dijet.jets.delta_r(dijet.jets.matched_gen)
            matched_gen_pt = ak.where(genjet_delta_r > 0.2, 0., dijet.jets.matched_gen['pt'])

            # Add the missing variables needed for energy corrections
            dijet.jets['pt_raw'] = (1 - dijet.jets['rawFactor'])*dijet.jets['pt']
            dijet.jets['mass_raw'] = (1 - dijet.jets['rawFactor'])*dijet.jets['mass']
            dijet.jets['pt_gen'] = ak.values_astype(ak.fill_none(matched_gen_pt, 0.), np.float32)
            dijet.jets['rho'] = ak.broadcast_arrays(event_rho, dijet.jets['pt'])[0]

            jerc_cache = cachetools.Cache(np.inf)
            dijet.jets = self.jet_corrector.build(dijet.jets, jerc_cache)

            # Apply JES and JER variations before event selection
            if self.jes_up:
                dijet.jets = dijet.jets.JES_Total.up
            elif self.jes_down:
                dijet.jets = dijet.jets.JES_Total.down

            if self.jer_up:
                dijet.jets = dijet.jets.JER.up
            elif self.jer_down:
                dijet.jets = dijet.jets.JER.down

            # Sort again by jet pT
            sorted_pt_arg = ak.argsort(dijet.jets['pt'], ascending=False)
            dijet.jets = dijet.jets[sorted_pt_arg]

        # GenJet matching
        leading_jet_genJetIdx = dijet.jets[:,0].genJetIdx
        subleading_jet_genJetIdx = dijet.jets[:,1].genJetIdx
        genjet_match_mask = ak.all([(leading_jet_genJetIdx >= 0), (subleading_jet_genJetIdx >= 0)], axis=0)

        deltaphi_mask = np.abs(dijet.jets[:,0].delta_phi(dijet.jets[:,1])) > 2.7

        subsubleading_mask = np.asarray(dijet.jets.mass[:,0] > -1) # Start with always true
        leading_jet_pt = dijet.jets['pt'][ak.num(dijet.jets) >= 3, 0]
        subleading_jet_pt = dijet.jets['pt'][ak.num(dijet.jets) >= 3, 1]
        subsubleading_jet_pt = dijet.jets['pt'][ak.num(dijet.jets) >= 3, 2]
        dijet_pt_average = (leading_jet_pt + subleading_jet_pt)*0.5
        subsubleading_mask[ak.num(dijet.jets) >= 3] = np.asarray(subsubleading_jet_pt < dijet_pt_average)

        alpha = np.zeros(len(dijet.jets))
        alpha[ak.num(dijet.jets) >= 3] = subsubleading_jet_pt / dijet_pt_average

        leading_jet_pt = dijet.jets['pt'][:,0]
        subleading_jet_pt = dijet.jets['pt'][:,1]
        pt_balance_mask = (leading_jet_pt-subleading_jet_pt)/(leading_jet_pt+subleading_jet_pt) < 0.7

        # Apply dijet selection
        dijet_mask = deltaphi_mask & subsubleading_mask & pt_balance_mask & genjet_match_mask
        dijet.sel = dijet_mask
        dijet.apply_sel()
        output['cutflow']['Dijet selection'] += len(dijet.jets)

        alpha = alpha[dijet_mask]
        PV_npvs = PV_npvs[dijet_mask]
        PV_npvsGood = PV_npvsGood[dijet_mask]
        Pileup_nTrueInt = Pileup_nTrueInt[dijet_mask]
        Generator_binvar = Generator_binvar[dijet_mask]
        LHE_HT = LHE_HT[dijet_mask]
        LHE_HTIncoming = LHE_HTIncoming[dijet_mask]
        LHEPart_pdgId = LHEPart_pdgId[dijet_mask]
        LHEPart_status = LHEPart_status[dijet_mask]
        PSWeight = PSWeight[dijet_mask]

        # Calculate the QGL value for each jet
        qgl_file = self.qgl_file
        qgl_rho_bins = qgl_file['rhoBins'].members['fElements']
        qgl_eta_bins = qgl_file['etaBins'].members['fElements']
        qgl_pt_bins = qgl_file['ptBins'].members['fElements']

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

        # Only pick the variables relevant for computing QGL
        qgl_variables = ['pt', 'eta', 'rho', 'qgl_axis2', 'qgl_mult', 'qgl_ptD']
        qgl_evaluator = self.qgl_evaluator

        leading_jets = ak.zip({var: dijet.jets[:,0][var] for var in qgl_variables})
        subleading_jets = ak.zip({var: dijet.jets[:,1][var] for var in qgl_variables})

        leading_jets['rho_bin'] = np.fromiter(map(find_rho_bin, leading_jets['rho']), dtype=int)
        leading_jets['eta_bin'] = np.fromiter(map(find_eta_bin, np.abs(leading_jets['eta'])), dtype=int)
        leading_jets['pt_bin'] = np.fromiter(map(find_pt_bin, leading_jets['pt']), dtype=int)

        subleading_jets['rho_bin'] = np.fromiter(map(find_rho_bin, subleading_jets['rho']), dtype=int)
        subleading_jets['eta_bin'] = np.fromiter(map(find_eta_bin, np.abs(subleading_jets['eta'])), dtype=int)
        subleading_jets['pt_bin'] = np.fromiter(map(find_pt_bin, subleading_jets['pt']), dtype=int)

        jet1_qgl_new = np.fromiter(map(lambda jet: compute_jet_qgl(jet, qgl_evaluator), leading_jets), dtype=float)
        jet2_qgl_new = np.fromiter(map(lambda jet: compute_jet_qgl(jet, qgl_evaluator), subleading_jets), dtype=float)

        # Add weights and save output
        n_dijet = len(dijet.jets)
        if n_dijet > 0:
            if filetype == 'mc':
                with open(os.path.join(os.environ['COFFEAHOME'],'utils','utils.json'),'r') as utils_json_file:
                    utils_json = json.load(utils_json_file)

                xsecs = utils_json['UL17']['xsec']
                nGenEvents = utils_json['UL17']['nGenEvents']
                lumi = utils_json['UL17']['lumi']

                sample_name = next((x for x in xsecs.keys() if dataset in x), None)
                if sample_name is None:
                    sys.exit('ERROR: xsec for sample not found in the JSON file.')
                if sample_name not in nGenEvents.keys():
                    sys.exit('ERROR: nGenEvents for sample not found in the JSON file.')
                sample_xsec = xsecs[sample_name]*1000 # Convert from pb to fb
                sample_nGenEvents = nGenEvents[sample_name]
                sample_lumi = lumi['ZeroBias']
                mc_scale = sample_lumi*sample_xsec / sample_nGenEvents

                weights = events['genWeight'][event_mask]
                weights = weights[dijet_mask].to_numpy()*mc_scale

                pileup_weight_evaluator = self.pileup_weights.to_evaluator()
                pileup_weight_up_evaluator = self.pileup_weights_up.to_evaluator()
                pileup_weight_down_evaluator = self.pileup_weights_down.to_evaluator()
                PU_weight = pileup_weight_evaluator.evaluate(Pileup_nTrueInt)
                PU_weight_up = pileup_weight_up_evaluator.evaluate(Pileup_nTrueInt)
                PU_weight_down = pileup_weight_down_evaluator.evaluate(Pileup_nTrueInt)

                nPSWeight = [len(x) for x in events.PSWeight]
                if len(np.unique(nPSWeight))==1 and np.unique(nPSWeight[0])==44:
                    FSR_weight_down = PSWeight[:,2].to_numpy()
                    FSR_weight_up = PSWeight[:,3].to_numpy()
                    ISR_weight_down = PSWeight[:,24].to_numpy()
                    ISR_weight_up = PSWeight[:,25].to_numpy()
                else:
                    sys.exit('ERROR! PSWeights not assigned correctly, as there aren\'t 44 weights.')

                L1prefiring_weight = events.L1PreFiringWeight.Nom[event_mask][dijet_mask].to_numpy()
                L1prefiring_weight_up = events.L1PreFiringWeight.Up[event_mask][dijet_mask].to_numpy()
                L1prefiring_weight_down = events.L1PreFiringWeight.Dn[event_mask][dijet_mask].to_numpy()

                LHEPart_isGluon = np.abs(LHEPart_pdgId == 21)
                LHEPart_isIncoming = LHEPart_status == -1
                gluon_weights = np.power(1.10, ak.num(LHEPart_pdgId[LHEPart_isGluon & LHEPart_isIncoming])).to_numpy()

                jet1_PartonFlavour = dijet.jets['partonFlavour'][:,0].to_numpy()
                jet1_HadronFlavour = dijet.jets['hadronFlavour'][:,0].to_numpy()
                jet2_PartonFlavour = dijet.jets['partonFlavour'][:,1].to_numpy()
                jet2_HadronFlavour = dijet.jets['hadronFlavour'][:,1].to_numpy()
            else:
                weights = np.ones(len(dijet.jets))
                PU_weight = np.ones(len(dijet.jets))
                PU_weight_up = np.ones(len(dijet.jets))
                PU_weight_down = np.ones(len(dijet.jets))
                gluon_weights = np.ones(len(dijet.jets))
                FSR_weight_up = np.ones(len(dijet.jets))
                FSR_weight_down = np.ones(len(dijet.jets))
                ISR_weight_up = np.ones(len(dijet.jets))
                ISR_weight_down = np.ones(len(dijet.jets))
                L1prefiring_weight = np.ones(len(dijet.jets))
                L1prefiring_weight_up = np.ones(len(dijet.jets))
                L1prefiring_weight_down = np.ones(len(dijet.jets))
                jet1_PartonFlavour = np.ones(len(dijet.jets))
                jet1_HadronFlavour = np.ones(len(dijet.jets))
                jet2_PartonFlavour = np.ones(len(dijet.jets))
                jet2_HadronFlavour = np.ones(len(dijet.jets))

            output['weight'] += processor.column_accumulator(np.reshape(weights, (n_dijet,)))
            output['PU_weight'] += processor.column_accumulator(np.reshape(PU_weight, (n_dijet,)))
            output['PU_weight_up'] += processor.column_accumulator(np.reshape(PU_weight_up, (n_dijet,)))
            output['PU_weight_down'] += processor.column_accumulator(np.reshape(PU_weight_down, (n_dijet,)))
            output['FSR_weight_up'] += processor.column_accumulator(np.reshape(FSR_weight_up, (n_dijet,)))
            output['FSR_weight_down'] += processor.column_accumulator(np.reshape(FSR_weight_down, (n_dijet,)))
            output['ISR_weight_up'] += processor.column_accumulator(np.reshape(ISR_weight_up, (n_dijet,)))
            output['ISR_weight_down'] += processor.column_accumulator(np.reshape(ISR_weight_down, (n_dijet,)))
            output['L1prefiring_weight'] += processor.column_accumulator(np.reshape(L1prefiring_weight, (n_dijet,)))
            output['L1prefiring_weight_up'] += processor.column_accumulator(np.reshape(L1prefiring_weight_up, (n_dijet,)))
            output['L1prefiring_weight_down'] += processor.column_accumulator(np.reshape(L1prefiring_weight_down, (n_dijet,)))
            output['gluon_weight'] += processor.column_accumulator(np.reshape(gluon_weights, (n_dijet,)))
            output['Jet1_PartonFlavour'] += processor.column_accumulator(np.reshape(jet1_PartonFlavour, (n_dijet,)))
            output['Jet1_HadronFlavour'] += processor.column_accumulator(np.reshape(jet1_HadronFlavour, (n_dijet,)))
            output['Jet1_pt'] += processor.column_accumulator(np.reshape(dijet.jets['pt'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_eta'] += processor.column_accumulator(np.reshape(dijet.jets['eta'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_phi'] += processor.column_accumulator(np.reshape(dijet.jets['phi'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_mass'] += processor.column_accumulator(np.reshape(dijet.jets['mass'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_qgl_new'] += processor.column_accumulator(np.reshape(jet1_qgl_new, (n_dijet,)))
            output['Jet1_qgl_axis2'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_axis2'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_qgl_ptD'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_ptD'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_qgl_mult'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_mult'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavUDS'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavUDS'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavQG'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavQG'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavG'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavG'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavB'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavB'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavCvB'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavCvB'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_btagDeepFlavCvL'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavCvL'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_particleNetAK4_QvsG'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_QvsG'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_particleNetAK4_B'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_B'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_particleNetAK4_CvsB'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_CvsB'][:,0].to_numpy(), (n_dijet,)))
            output['Jet1_particleNetAK4_CvsL'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_CvsL'][:,0].to_numpy(), (n_dijet,)))
            output['Jet2_PartonFlavour'] += processor.column_accumulator(np.reshape(jet2_PartonFlavour, (n_dijet,)))
            output['Jet2_HadronFlavour'] += processor.column_accumulator(np.reshape(jet2_HadronFlavour, (n_dijet,)))
            output['Jet2_pt'] += processor.column_accumulator(np.reshape(dijet.jets['pt'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_eta'] += processor.column_accumulator(np.reshape(dijet.jets['eta'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_phi'] += processor.column_accumulator(np.reshape(dijet.jets['phi'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_mass'] += processor.column_accumulator(np.reshape(dijet.jets['mass'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_qgl_new'] += processor.column_accumulator(np.reshape(jet2_qgl_new, (n_dijet,)))
            output['Jet2_qgl_axis2'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_axis2'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_qgl_ptD'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_ptD'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_qgl_mult'] += processor.column_accumulator(np.reshape(dijet.jets['qgl_mult'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavUDS'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavUDS'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavQG'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavQG'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavG'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavG'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavB'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavB'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavCvB'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavCvB'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_btagDeepFlavCvL'] += processor.column_accumulator(np.reshape(dijet.jets['btagDeepFlavCvL'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_particleNetAK4_QvsG'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_QvsG'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_particleNetAK4_B'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_B'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_particleNetAK4_CvsB'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_CvsB'][:,1].to_numpy(), (n_dijet,)))
            output['Jet2_particleNetAK4_CvsL'] += processor.column_accumulator(np.reshape(dijet.jets['particleNetAK4_CvsL'][:,1].to_numpy(), (n_dijet,)))
            output['alpha'] += processor.column_accumulator(np.reshape(alpha, (n_dijet,)))
            output['fixedGridRhoFastjetAll'] += processor.column_accumulator(np.reshape(dijet.jets['rho'][:,0].to_numpy(), (n_dijet,)))
            output['LHE_HT'] += processor.column_accumulator(np.reshape(LHE_HT.to_numpy(), (n_dijet,)))
            output['LHE_HTIncoming'] += processor.column_accumulator(np.reshape(LHE_HTIncoming.to_numpy(), (n_dijet,)))
            output['Generator_binvar'] += processor.column_accumulator(np.reshape(Generator_binvar.to_numpy(), (n_dijet,)))
            output['PV_npvs'] += processor.column_accumulator(np.reshape(PV_npvs.to_numpy(), (n_dijet,)))
            output['PV_npvsGood'] += processor.column_accumulator(np.reshape(PV_npvsGood.to_numpy(), (n_dijet,)))
            output['Pileup_nTrueInt'] += processor.column_accumulator(np.reshape(Pileup_nTrueInt.to_numpy(), (n_dijet,)))
        return output

    def postprocess(self, accumulator):
        return accumulator

# Processor for z+jets analysis
class ZmmProcessor(processor.ProcessorABC):
    def __init__(self, puppi, jes_up, jes_down):
        self.output = {
            'cutflow' : processor.defaultdict_accumulator(int), 
            'weight': processor.column_accumulator(np.array([])),
            'PU_weight': processor.column_accumulator(np.array([])),
            'PU_weight_up': processor.column_accumulator(np.array([])),
            'PU_weight_down': processor.column_accumulator(np.array([])),
            'FSR_weight_up': processor.column_accumulator(np.array([])),
            'FSR_weight_down': processor.column_accumulator(np.array([])),
            'ISR_weight_up': processor.column_accumulator(np.array([])),
            'ISR_weight_down': processor.column_accumulator(np.array([])),
            'L1prefiring_weight': processor.column_accumulator(np.array([])),
            'L1prefiring_weight_up': processor.column_accumulator(np.array([])),
            'L1prefiring_weight_down': processor.column_accumulator(np.array([])),
            'gluon_weight': processor.column_accumulator(np.array([])),
            'Jet_PartonFlavour': processor.column_accumulator(np.array([])),
            'Jet_HadronFlavour': processor.column_accumulator(np.array([])),
            'Jet_pt': processor.column_accumulator(np.array([])),
            'Jet_eta': processor.column_accumulator(np.array([])),
            'Jet_phi': processor.column_accumulator(np.array([])),
            'Jet_mass': processor.column_accumulator(np.array([])),
            'Jet_qgl_new': processor.column_accumulator(np.array([])),
            'Jet_qgl_axis2': processor.column_accumulator(np.array([])),
            'Jet_qgl_ptD': processor.column_accumulator(np.array([])),
            'Jet_qgl_mult': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavUDS': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavQG': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavG': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavB': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavCvB': processor.column_accumulator(np.array([])),
            'Jet_btagDeepFlavCvL': processor.column_accumulator(np.array([])),
            'Jet_particleNetAK4_QvsG': processor.column_accumulator(np.array([])),
            'Jet_particleNetAK4_B': processor.column_accumulator(np.array([])),
            'Jet_particleNetAK4_CvsB': processor.column_accumulator(np.array([])),
            'Jet_particleNetAK4_CvsL': processor.column_accumulator(np.array([])),
            'Dimuon_mass': processor.column_accumulator(np.array([])),
            'Dimuon_pt': processor.column_accumulator(np.array([])),
            'Dimuon_eta': processor.column_accumulator(np.array([])),
            'alpha': processor.column_accumulator(np.array([])),
            'fixedGridRhoFastjetAll': processor.column_accumulator(np.array([])),
            'PV_npvs': processor.column_accumulator(np.array([])),
            'PV_npvsGood': processor.column_accumulator(np.array([])),
            'Pileup_nTrueInt': processor.column_accumulator(np.array([])),
            }

        self.puppi = puppi
        self.jes_up = jes_up
        self.jes_down = jes_down

        coffea_base_path = os.environ['COFFEAHOME']
        jes_unc = 'Summer19UL17_V5_MC_Total_AK4PFchs'
        self.jes_evaluator = correctionlib.CorrectionSet.from_file('{}/utils/JERC/jet_jerc_UL17.json.gz'.format(coffea_base_path))[jes_unc]
        self.qgl_evaluator = correctionlib.CorrectionSet.from_file('{}/utils/QGL/pdfQG_AK4chs_13TeV_UL17_ghosts.corr.json'.format(coffea_base_path))
        self.qgl_file = uproot.open('{}/utils/QGL/pdfQG_AK4chs_13TeV_UL17_ghosts.root'.format(coffea_base_path))
        self.pileup_weights = from_uproot_THx('{}/utils/pileup/PU_weights_goldenJSON_69200ub_pythia8_UL17.root:weight'.format(coffea_base_path), flow='clamp')
        self.pileup_weights_down = from_uproot_THx('{}/utils/pileup/PU_weights_goldenJSON_66000ub_pythia8_UL17.root:weight'.format(coffea_base_path), flow='clamp')
        self.pileup_weights_up = from_uproot_THx('{}/utils/pileup/PU_weights_goldenJSON_72400ub_pythia8_UL17.root:weight'.format(coffea_base_path), flow='clamp')
        
    def process(self, events):
        output = self.output
        dataset = events.metadata['dataset']

        if dataset in utils.data_names:
            filetype = 'data'
        elif dataset in utils.mc_names:
            filetype = 'mc'

        vector.register_awkward()
        # Properties per object
        if self.puppi:
            jets = ak.zip(
                {
                'pt': events.JetPuppi_pt,
                'eta': events.JetPuppi_eta,
                'phi': events.JetPuppi_phi,
                'mass': events.JetPuppi_mass,
                'jet_id': events.JetPuppi_jetId, 
                'partonFlavour': events.JetPuppi_partonFlavour if 'Jet_partonFlavour' in events.fields else ak.ones_like(events.Jet_jetId), #dummy flag for data
                'hadronFlavour': events.JetPuppi_hadronFlavour if 'Jet_hadronFlavour' in events.fields else ak.ones_like(events.Jet_jetId),
                'axis2': events.JetPuppi_qgl_axis2,
                'ptD': events.JetPuppi_qgl_ptD,
                'mult': events.JetPuppi_qgl_mult,
                'rho': events.fixedGridRhoFastjetAll,
                'deepFlavQG': events.JetPuppi_btagDeepFlavQG,
                'deepFlavUDS': events.JetPuppi_btagDeepFlavUDS,
                'deepFlavG': events.JetPuppi_btagDeepFlavG,
                'deepFlavB': events.JetPuppi_btagDeepFlavB,
                'deepFlavCvB': events.JetPuppi_btagDeepFlavCvB,
                'deepFlavCvL': events.JetPuppi_btagDeepFlavCvL,
                'particleNetAK4_QvsG': events.JetPuppi_particleNetAK4_QvsG,
                'particleNetAK4_B': events.JetPuppi_particleNetAK4_B,
                'particleNetAK4_CvsB': events.JetPuppi_particleNetAK4_CvsB,
                'particleNetAK4_CvsL': events.JetPuppi_particleNetAK4_CvsL,
                'genJetIdx': events.JetPuppi_genJetIdx if 'JetPuppi_genJetIdx' in events.fields else ak.ones_like(events.Jet_jetId), 
                }, with_name='Momentum4D'
                )
        else:
            jets = ak.zip(
                {
                'pt': events.Jet_pt,
                'eta': events.Jet_eta,
                'phi': events.Jet_phi,
                'mass': events.Jet_mass,
                'jet_id': events.Jet_jetId,
                'partonFlavour': events.Jet_partonFlavour if 'Jet_partonFlavour' in events.fields else ak.ones_like(events.Jet_jetId),
                'hadronFlavour': events.Jet_hadronFlavour if 'Jet_hadronFlavour' in events.fields else ak.ones_like(events.Jet_jetId),
                'axis2': events.Jet_qgl_axis2,
                'ptD': events.Jet_qgl_ptD,
                'mult': events.Jet_qgl_mult,
                'rho': events.fixedGridRhoFastjetAll,
                'deepFlavQG': events.Jet_btagDeepFlavQG,
                'deepFlavUDS': events.Jet_btagDeepFlavUDS,
                'deepFlavG': events.Jet_btagDeepFlavG,
                'deepFlavB': events.Jet_btagDeepFlavB,
                'deepFlavCvB': events.Jet_btagDeepFlavCvB,
                'deepFlavCvL': events.Jet_btagDeepFlavCvL,
                'particleNetAK4_QvsG': events.Jet_particleNetAK4_QvsG,
                'particleNetAK4_B': events.Jet_particleNetAK4_B,
                'particleNetAK4_CvsB': events.Jet_particleNetAK4_CvsB,
                'particleNetAK4_CvsL': events.Jet_particleNetAK4_CvsL,
                'genJetIdx': events.Jet_genJetIdx if 'Jet_genJetIdx' in events.fields else ak.ones_like(events.Jet_jetId), 
                }, with_name='Momentum4D'
                )

        muons = ak.zip(
            {
            'pt': events.Muon_pt,
            'eta': events.Muon_eta,
            'phi': events.Muon_phi,
            'mass': events.Muon_mass,
            'charge': events.Muon_charge,
            'iso': events.Muon_pfRelIso04_all,
            'dxy': events.Muon_dxy,
            'dz': events.Muon_dz,            
            'isTight': events.Muon_tightId,
            'isMedium': events.Muon_mediumId,
            'isLoose': events.Muon_looseId
            }, with_name='Momentum4D'
            )

        electrons = ak.zip(
            {
            'pt': events.Electron_pt,
            'eta': events.Electron_eta,
            'phi': events.Electron_phi,
            'mass': events.Electron_mass,
            'charge': events.Electron_charge,
            'dxy': events.Electron_dxy,
            'iso': events.Electron_pfRelIso03_all,
            'dz': events.Electron_dz,
            'cutbased': events.Electron_cutBased
            }, with_name='Momentum4D'
            )

        if filetype == 'mc':
            lumi_mask = np.ones(len(electrons), dtype=np.bool_) #dummy lumi mask for MC
        else:
            run = events['run']
            lumiblock = events['luminosityBlock']
            goldenJSON_path = os.path.join(os.environ['COFFEAHOME'],'data','json')
            lumi_path = '{}/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'.format(goldenJSON_path) #FixMe: Add option for different years
            lumi_mask = LumiMask(lumi_path)(run,lumiblock)
                    
        nEvents = len(events.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8)
        PV_npvs = events.PV_npvs
        PV_npvsGood = events.PV_npvsGood
        Pileup_nTrueInt = events.Pileup_nTrueInt if 'Pileup_nTrueInt' in events.fields else ak.ones_like(events.PV_npvs)
        LHEPart_pdgId = events.LHEPart_pdgId if 'LHEPart_pdgId' in events.fields else ak.ones_like(events.PV_npvs)
        LHEPart_status = events.LHEPart_status if 'LHEPart_status' in events.fields else ak.ones_like(events.PV_npvs)
        PSWeight = events.PSWeight if 'PSWeight' in events.fields else ak.ones_like(events.PV_npvs)

        # This keeps track of how many events there are, as well as how many of each object exist in this events
        output['cutflow']['all events'] += nEvents
        output['cutflow']['all chs jets'] += ak.num(jets).to_numpy().sum()
        output['cutflow']['all muons'] += ak.num(muons).to_numpy().sum()
        output['cutflow']['all electrons'] += ak.num(electrons).to_numpy().sum()

        # # # # # # # # # # #
        # TRIGGER SELECTION #
        # # # # # # # # # # #
        
        trigger_mask = (events.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8==1)

        # # # # # # # #
        # MET FILTERS #
        # # # # # # # #

        MET_mask = (
                (events.Flag_goodVertices == 1) & (events.Flag_globalSuperTightHalo2016Filter == 1) & (events.Flag_HBHENoiseFilter==1) &
                (events.Flag_HBHENoiseIsoFilter==1) & (events.Flag_EcalDeadCellTriggerPrimitiveFilter==1) & (events.Flag_BadPFMuonFilter==1) &
                (events.Flag_BadPFMuonDzFilter==1) & (events.Flag_eeBadScFilter==1) & (events.Flag_ecalBadCalibFilter==1)
                )

        # # # # # # # # # # #
        # OBJECT SELECTIONS #
        # # # # # # # # # # #

        muons, muon_selection = utils.ObjSelection(muons,'muon',2017)
        electrons,_ = utils.ObjSelection(electrons,'electron',2017)
        jets, _ = utils.ObjSelection(jets,'jet',2017)

        # Now we want to make sure no jets are within 0.4 delta-R of any muon.
        cross_jmu = ak.cartesian([jets, muons], nested=True)
        check_jmu = ak.all((cross_jmu.slot0.deltaR(cross_jmu.slot1) > 0.4), axis=-1)
        jets = jets[(check_jmu)&(jets.mass>-1)]

        output['cutflow']['cleaned electrons'] += ak.num(electrons).to_numpy().sum()
        output['cutflow']['cleaned muons'] += ak.num(muons).to_numpy().sum()
        output['cutflow']['cleaned chs jets'] += ak.num(jets).to_numpy().sum()

        # # # # # # # # # #
        # EVENT SELECTION #
        # # # # # # # # # #

        jet_mask = (ak.num(jets) >= 1) 
        electron_mask = (ak.num(electrons) == 0)
        muon_mask = (ak.num(muons) == 2)

        PV_z = events.PV_z
        if 'GenVtx_z' in events.fields:
            GenVtx_z = events.GenVtx_z
            vtx_mask = np.abs(GenVtx_z - PV_z) < 0.2
        else:
            vtx_mask = np.ones_like(len(PV_z), dtype=bool)

        event_mask = jet_mask & electron_mask & muon_mask & lumi_mask & trigger_mask & MET_mask & vtx_mask
        # event_mask = jet_mask & electron_mask & muon_mask & lumi_mask & trigger_mask & MET_mask

        PV_npvs = PV_npvs[event_mask]
        PV_npvsGood = PV_npvsGood[event_mask]
        Pileup_nTrueInt = Pileup_nTrueInt[event_mask]
        LHEPart_pdgId = LHEPart_pdgId[event_mask]
        LHEPart_status = LHEPart_status[event_mask]
        PSWeight = PSWeight[event_mask]

        z_jets = Channel('zmm')
        z_jets.muons = muons
        z_jets.jets = jets
        z_jets.sel = event_mask
        z_jets.apply_sel()

        if filetype == 'mc':
            # Apply JES uncertainties before event selection
            if self.jes_up or self.jes_down:
                jes_evaluator = self.jes_evaluator
                flattened_jets, num_jets = ak.flatten(z_jets.jets), ak.num(z_jets.jets) # until correctionlib handles ak arrays we must flatten & unflatten
                if self.jes_up:
                    pt_var_up = 1 + jes_evaluator.evaluate(flattened_jets['eta'], flattened_jets['pt'])
                    flattened_corr_pt_up = flattened_jets['pt']*pt_var_up
                    corr_pt = ak.unflatten(flattened_corr_pt_up, num_jets)
                    z_jets.jets['pt'] = corr_pt
                if self.jes_down:
                    pt_var_down = 1 - jes_evaluator.evaluate(flattened_jets['eta'], flattened_jets['pt'])
                    flattened_corr_pt_down = flattened_jets['pt']*pt_var_down
                    corr_pt = ak.unflatten(flattened_corr_pt_down, num_jets)
                    z_jets.jets['pt'] = corr_pt
                # Sort again by jet pT
                sorted_pt_arg = ak.argsort(z_jets.jets['pt'], ascending=False)
                z_jets.jets = z_jets.jets[sorted_pt_arg]

        leading_jet_pt = z_jets.jets['pt'][:,0]
        z_charge = z_jets.muons[:,0].charge+z_jets.muons[:,1].charge
        charge_mask = z_charge == 0

        z_cand = ak.combinations(z_jets.muons, 2)
        z_cand = z_cand.slot0 + z_cand.slot1

        cross_jz = ak.cartesian([z_jets.jets, z_cand])
        z_jet_deltaphi = cross_jz.slot0.deltaphi(cross_jz.slot1)
        deltaphi_mask = np.abs(z_jet_deltaphi[:,0]) > 2.7

        mass_mask = (z_cand.mass > 71.2) & (z_cand.mass < 111.2)

        subleading_mask = np.asarray(z_jets.jets.mass[:,0] > -1) # Start with always true
        subleading_jet_pt = z_jets.jets['pt'][ak.num(z_jets.jets) >= 2, 1]
        z_pt = z_cand.pt[ak.num(z_jets.jets) >= 2, 0]
        subleading_mask[ak.num(z_jets.jets) >= 2] = np.asarray((subleading_jet_pt / z_pt) < 1.0)

        alpha = np.zeros(len(z_jets.jets))
        alpha[ak.num(z_jets.jets) >= 2] = subleading_jet_pt / z_pt

        z_pt_mask = z_cand.pt > 15

        genjet_match_mask = z_jets.jets[:,0].genJetIdx >= 0

        # Apply Z+Jets selection
        # dilepton_mask = ak.flatten(mass_mask) & charge_mask & deltaphi_mask & subleading_mask & ak.flatten(z_pt_mask)
        dilepton_mask = ak.flatten(mass_mask) & charge_mask & deltaphi_mask & subleading_mask & ak.flatten(z_pt_mask) & genjet_match_mask
        z_jets.sel = dilepton_mask
        z_jets.apply_sel()
        output['cutflow']['Dilepton selection'] += len(z_jets.jets)

        alpha = alpha[dilepton_mask]
        PV_npvs = PV_npvs[dilepton_mask]
        PV_npvsGood = PV_npvsGood[dilepton_mask]
        Pileup_nTrueInt = Pileup_nTrueInt[dilepton_mask]
        LHEPart_pdgId = LHEPart_pdgId[dilepton_mask]
        LHEPart_status = LHEPart_status[dilepton_mask]
        PSWeight = PSWeight[dilepton_mask]

        # Calculate the QGL value for each leading jet
        qgl_file = self.qgl_file
        qgl_rho_bins = qgl_file['rhoBins'].members['fElements']
        qgl_eta_bins = qgl_file['etaBins'].members['fElements']
        qgl_pt_bins = qgl_file['ptBins'].members['fElements']

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

        z_jets.jets['rho_bin'] = list(map(find_rho_bin, z_jets.jets['rho'][:,0]))
        z_jets.jets['eta_bin'] = list(map(find_eta_bin, np.abs(z_jets.jets['eta'][:,0])))
        z_jets.jets['pt_bin'] = list(map(find_pt_bin, z_jets.jets['pt'][:,0]))

        qgl_evaluator = self.qgl_evaluator
        leading_jets = z_jets.jets[:,0].to_list()
        z_jets.jets['qgl_new'] = [compute_jet_qgl(jet, qgl_evaluator) for jet in leading_jets]

        n_z_jets = len(z_jets.jets)
        if n_z_jets > 0:
            if filetype == 'mc':
                with open(os.path.join(os.environ['COFFEAHOME'],'utils','utils.json'),'r') as utils_json_file:
                    utils_json = json.load(utils_json_file)

                xsecs = utils_json['UL17']['xsec']
                nGenEvents = utils_json['UL17']['nGenEvents']
                lumi = utils_json['UL17']['lumi']

                sample_name = next((x for x in xsecs.keys() if dataset in x), None)
                if sample_name is None:
                    sys.exit('ERROR: xsec for sample not found in the JSON file.')
                if sample_name not in nGenEvents.keys():
                    sys.exit('ERROR: nGenEvents for sample not found in the JSON file.')
                sample_xsec = xsecs[sample_name]*1000 # Convert from pb to fb
                sample_nGenEvents = nGenEvents[sample_name]
                sample_lumi = lumi['DoubleMuon']
                mc_scale = sample_lumi*sample_xsec / sample_nGenEvents

                weights = events['genWeight'][event_mask]
                weights = weights[dilepton_mask].to_numpy()*mc_scale

                pileup_weight_evaluator = self.pileup_weights.to_evaluator()
                pileup_weight_up_evaluator = self.pileup_weights_up.to_evaluator()
                pileup_weight_down_evaluator = self.pileup_weights_down.to_evaluator()
                PU_weight = pileup_weight_evaluator.evaluate(Pileup_nTrueInt)
                PU_weight_up = pileup_weight_up_evaluator.evaluate(Pileup_nTrueInt)
                PU_weight_down = pileup_weight_down_evaluator.evaluate(Pileup_nTrueInt)
                if len(np.unique(events.nPSWeight))==1 and np.unique(events.nPSWeight[0])==44:
                    FSR_weight_down = PSWeight[:,2].to_numpy()
                    FSR_weight_up = PSWeight[:,3].to_numpy()
                    ISR_weight_down = PSWeight[:,24].to_numpy()
                    ISR_weight_up = PSWeight[:,25].to_numpy()
                else:
                    sys.exit('ERROR! PSWeights not assigned correctly, as there aren\'t 44 weights.')

                L1prefiring_weight = events['L1PreFiringWeight_Nom'][event_mask][dilepton_mask].to_numpy()
                L1prefiring_weight_up = events['L1PreFiringWeight_Up'][event_mask][dilepton_mask].to_numpy()
                L1prefiring_weight_down = events['L1PreFiringWeight_Dn'][event_mask][dilepton_mask].to_numpy()

                LHEPart_isGluon = np.abs(LHEPart_pdgId == 21)
                LHEPart_isIncoming = LHEPart_status == -1
                gluon_weights = np.power(1.10, ak.num(LHEPart_pdgId[LHEPart_isGluon & LHEPart_isIncoming])).to_numpy()

                jet_PartonFlavour = z_jets.jets['partonFlavour'][:,0].to_numpy()
                jet_HadronFlavour = z_jets.jets['hadronFlavour'][:,0].to_numpy()
            else:
                weights = np.ones(len(z_jets.jets))
                PU_weight = np.ones(len(z_jets.jets))
                PU_weight_up = np.ones(len(z_jets.jets))
                PU_weight_down = np.ones(len(z_jets.jets))
                gluon_weights = np.ones(len(z_jets.jets))
                FSR_weight_up = np.ones(len(z_jets.jets))
                FSR_weight_down = np.ones(len(z_jets.jets))
                ISR_weight_up = np.ones(len(z_jets.jets))
                ISR_weight_down = np.ones(len(z_jets.jets))
                L1prefiring_weight = np.ones(len(z_jets.jets))
                L1prefiring_weight_up = np.ones(len(z_jets.jets))
                L1prefiring_weight_down = np.ones(len(z_jets.jets))
                jet_PartonFlavour = np.ones(len(z_jets.jets))
                jet_HadronFlavour = np.ones(len(z_jets.jets))

            output['weight'] += processor.column_accumulator(np.reshape(weights, (n_z_jets,)))
            output['PU_weight'] += processor.column_accumulator(np.reshape(PU_weight, (n_z_jets,)))
            output['PU_weight_up'] += processor.column_accumulator(np.reshape(PU_weight_up, (n_z_jets,)))
            output['PU_weight_down'] += processor.column_accumulator(np.reshape(PU_weight_down, (n_z_jets,)))
            output['FSR_weight_up'] += processor.column_accumulator(np.reshape(FSR_weight_up, (n_z_jets,)))
            output['FSR_weight_down'] += processor.column_accumulator(np.reshape(FSR_weight_down, (n_z_jets,)))
            output['ISR_weight_up'] += processor.column_accumulator(np.reshape(ISR_weight_up, (n_z_jets,)))
            output['ISR_weight_down'] += processor.column_accumulator(np.reshape(ISR_weight_down, (n_z_jets,)))
            output['L1prefiring_weight'] += processor.column_accumulator(np.reshape(L1prefiring_weight, (n_z_jets,)))
            output['L1prefiring_weight_up'] += processor.column_accumulator(np.reshape(L1prefiring_weight_up, (n_z_jets,)))
            output['L1prefiring_weight_down'] += processor.column_accumulator(np.reshape(L1prefiring_weight_down, (n_z_jets,)))
            output['gluon_weight'] += processor.column_accumulator(np.reshape(gluon_weights, (n_z_jets,)))
            output['Jet_PartonFlavour'] += processor.column_accumulator(np.reshape(jet_PartonFlavour, (n_z_jets,)))
            output['Jet_HadronFlavour'] += processor.column_accumulator(np.reshape(jet_HadronFlavour, (n_z_jets,)))
            output['Jet_pt'] += processor.column_accumulator(np.reshape(z_jets.jets['pt'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_eta'] += processor.column_accumulator(np.reshape(z_jets.jets['eta'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_phi'] += processor.column_accumulator(np.reshape(z_jets.jets['phi'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_mass'] += processor.column_accumulator(np.reshape(z_jets.jets['mass'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_new'] += processor.column_accumulator(np.reshape(z_jets.jets['qgl_new'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_axis2'] += processor.column_accumulator(np.reshape(z_jets.jets['axis2'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_ptD'] += processor.column_accumulator(np.reshape(z_jets.jets['ptD'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_qgl_mult'] += processor.column_accumulator(np.reshape(z_jets.jets['mult'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavUDS'] += processor.column_accumulator(np.reshape(z_jets.jets['deepFlavUDS'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavQG'] += processor.column_accumulator(np.reshape(z_jets.jets['deepFlavQG'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavG'] += processor.column_accumulator(np.reshape(z_jets.jets['deepFlavG'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavB'] += processor.column_accumulator(np.reshape(z_jets.jets['deepFlavB'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavCvB'] += processor.column_accumulator(np.reshape(z_jets.jets['deepFlavCvB'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_btagDeepFlavCvL'] += processor.column_accumulator(np.reshape(z_jets.jets['deepFlavCvL'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_particleNetAK4_QvsG'] += processor.column_accumulator(np.reshape(z_jets.jets['particleNetAK4_QvsG'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_particleNetAK4_B'] += processor.column_accumulator(np.reshape(z_jets.jets['particleNetAK4_B'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_particleNetAK4_CvsB'] += processor.column_accumulator(np.reshape(z_jets.jets['particleNetAK4_CvsB'][:,0].to_numpy(), (n_z_jets,)))
            output['Jet_particleNetAK4_CvsL'] += processor.column_accumulator(np.reshape(z_jets.jets['particleNetAK4_CvsL'][:,0].to_numpy(), (n_z_jets,)))
            output['Dimuon_mass'] += processor.column_accumulator(np.reshape(z_cand[(dilepton_mask)].mass.to_numpy(), (n_z_jets,)))
            output['Dimuon_pt'] += processor.column_accumulator(np.reshape(z_cand[(dilepton_mask)].pt.to_numpy(), (n_z_jets,)))
            output['Dimuon_eta'] += processor.column_accumulator(np.reshape(z_cand[(dilepton_mask)].eta.to_numpy(), (n_z_jets,)))
            output['alpha'] += processor.column_accumulator(np.reshape(alpha, (n_z_jets,)))
            output['fixedGridRhoFastjetAll'] += processor.column_accumulator(np.reshape(z_jets.jets['rho'][:,0].to_numpy(), (n_z_jets,)))
            output['PV_npvs'] += processor.column_accumulator(np.reshape(PV_npvs.to_numpy(), (n_z_jets,)))
            output['PV_npvsGood'] += processor.column_accumulator(np.reshape(PV_npvsGood.to_numpy(), (n_z_jets,)))
            output['Pileup_nTrueInt'] += processor.column_accumulator(np.reshape(Pileup_nTrueInt.to_numpy(), (n_z_jets,)))
        return output

    def postprocess(self, accumulator):
        return accumulator