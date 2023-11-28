import sys
import numpy as np

data_names = ['DoubleMuon','ZeroBias']
mc_names = ['DY','QCD_HT50to100','QCD_HT100to200','QCD_HT200to300','QCD_HT300to500','QCD_HT500to700','QCD_HT700to1000',
            'QCD_HT1000to1500','QCD_HT1500to2000','QCD_HT2000toInf','QCD_Pt_15to30','QCD_Pt_30to50','QCD_Pt_50to80','QCD_Pt_80to120',
            'QCD_Pt_120to170','QCD_Pt_170to300','QCD_Pt_300to470','QCD_Pt_470to600','QCD_Pt_600to800','QCD_Pt_800to1000',
            'QCD_Pt_1000to1400','QCD_Pt_1400to1800','QCD_Pt_1800to2400','QCD_Pt_2400to3200','QCD_Pt_3200toInf', 'QCD_Pt-15to7000']

def ObjSelection(obj,name,year=2016,selection=None):
    '''Apply standard selection to ojects.
    Inputs: 
    obj: jagged_array to apply the selection
    selection: bool mask for selection different than the standard
    Returns: obj array with selection applied '''

    if year not in [2016, 2017, 2018]:
        sys.exit('ERROR: Only 2016--2018 supported')

    obj_list = ['jet','muon','electron']
    
    if name not in obj_list and selection == None:
        sys.exit('ERROR: For non standard objects, provide a selection mask to be applied')

    if selection != None:
        return obj[selection]
    else:
        if year == 2016:
            if name == 'jet':
                selection = (
                    (np.abs(obj['eta']) < 4.7) &
                    (obj['pt'] > 20) &
                    (obj['jetId']==6) & # tight
                    (obj['mass'] > -1)
                )
            elif name == 'muon':
                selection = (
                    (np.abs(obj['eta']) < 2.3) &
                    (obj['pt'] > 20)  &
                    (obj['dxy'] < 0.2) &
                    (obj['dz'] < 0.5) &
                    (obj['iso'] < 0.15) &
                    (obj['tightId']) &
                    (obj['mass'] > -1)
                )
            elif name == 'electron':
                selection = (
                    (np.abs(obj['eta']) < 2.4) & (obj['pt'] > 10)  &
                    (
                        ((np.abs(obj['eta']) < 1.4442) & (np.abs(obj['dxy']) < 0.05) & (np.abs(obj['dz']) < 0.1) ) 
                        | ((np.abs(obj['eta']) > 1.5660) & (np.abs(obj['dxy']) < 0.1) & (np.abs(obj['dz']) < 0.2) )
                    ) & 
                    (obj['cutBased'] >= 1) &
                    (obj['mass'] > -1)        
                )
        elif year == 2017:
            if name == 'jet':
                selection = (
                    (np.abs(obj['eta']) < 4.7) &
                    (obj['pt'] > 20)  & 
                    (obj['jetId']==6) & # tight
                    (obj['mass'] > -1) 
                )
            elif name == 'muon':
                selection = (
                    (np.abs(obj['eta']) < 2.3) &
                    (obj['pt'] > 20) &
                    (obj['pfRelIso04_all'] < 0.15) &
                    (obj['tightId']) &
                    (obj['mass'] > -1)  
                )
            elif name == 'electron':
                selection = (
                    (np.abs(obj['eta']) < 2.4) & (obj['pt'] > 10)  &
                    (
                        ((np.abs(obj['eta']) < 1.4442) & (np.abs(obj['dxy']) < 0.05) & (np.abs(obj['dz']) < 0.1) ) 
                        | ((np.abs(obj['eta']) > 1.5660) & (np.abs(obj['dxy']) < 0.1) & (np.abs(obj['dz']) < 0.2) )
                    ) & 
                    (obj['cutBased'] >= 1) &
                    (obj['mass'] > -1)        
                )

        elif year == 2018:
            if name == 'jet':
                selection = (
                    (np.abs(obj['eta']) < 4.7) &
                    (obj['pt'] > 20)  & 
                    (obj['jetId']==6) & # tight
                    (obj['mass'] > -1) 
                )
            elif name == 'muon':
                selection = (
                    (np.abs(obj['eta']) < 2.3) &
                    (obj['pt'] > 20) &
                    (obj['pfRelIso04_all'] < 0.15) &
                    (obj['tightId']) &
                    (obj['mass'] > -1)  
                )
            elif name == 'electron':
                selection = (
                    (np.abs(obj['eta']) < 2.4) & (obj['pt'] > 10)  &
                    (
                        ((np.abs(obj['eta']) < 1.4442) & (np.abs(obj['dxy']) < 0.05) & (np.abs(obj['dz']) < 0.1) ) 
                        | ((np.abs(obj['eta']) > 1.5660) & (np.abs(obj['dxy']) < 0.1) & (np.abs(obj['dz']) < 0.2) )
                    ) & 
                    (obj['cutBased'] >= 1) &
                    (obj['mass'] > -1)        
                )

        return obj[selection], selection
    
