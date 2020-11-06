import sys
import numpy as np


data_names = [
    'SingleMuon',
    'DoubleMuon',
    'ZeroBias',
]

def ObjSelection(obj,name,year=2016,selection=None):
    '''Apply standard selection to ojects.
    Inputs: 
    obj: jagged_array to apply the selection
    name: [jet,electron,muon,PF,tight_muon]
    selection: bool mask for selection different than the standard

    Returns: obj array with selection applied '''

    if year != 2017 and year != 2016: #FixMe add other years
        sys.exit("ERROR: Only 2016 and 2017 supported")

    obj_list = ['jet','electron','muon','PF','tight_muon','medium_muon']
    
    if name not in obj_list and selection == None:
        sys.exit("ERROR: For non standard objects, provide a selection mask to be applied")

    if selection != None:
        return obj[selection]
    else:
        if year == 2016:
            if name == 'jet':
                selection = (
                    (np.abs(obj.eta) < 2.4) & (obj.pt > 30)  & 
                    (
                        (obj.pt >= 50) 
                        | ((obj.pt < 50) & (obj.pu_id>=6) )
                    ) &
                    ((obj.jet_id==7)) & #tight
                    #(jets.b_tag >= 0) &
                    (obj.mass > -1) 
                )
                
            elif name == 'muon':
                selection = (
                    (np.abs(obj.eta) < 2.4) & (obj.pt > 10)  &
                    (obj['dxy'] < 0.2) &
                    (obj['dz'] < 0.5) &
                    (obj['iso'] < 0.15) &
                    (obj['isLoose']) &
                    (obj.mass > -1)  
                )
                
            elif name == 'tight_muon':
                selection = (
                    (np.abs(obj.eta) < 2.4) & (obj.pt > 20)  &
                    (obj['dxy'] < 0.2) &
                    (obj['dz'] < 0.5) &
                    (obj['iso'] < 0.15) &
                    (obj['isTight']) &
                    (obj.mass > -1) 
                )
                
            elif name == 'electron':
                selection = (
                    (np.abs(obj.eta) < 2.4) & (obj.pt > 15)  &
                    (obj.iso < 0.15) & 
                    (
                        ((np.abs(obj.eta) < 1.4442) & (np.abs(obj['dxy']) < 0.05) & (np.abs(obj['dz']) < 0.1) ) 
                        | ((np.abs(obj.eta) > 1.5660) & (np.abs(obj['dxy']) < 0.1) & (np.abs(obj['dz']) < 0.2) )
                    ) & 
                    (obj['cutbased']>=2) &
                    (obj.mass >-1)        
                )
                
        if year == 2017:
                
            if name == 'jet':
                selection = (
                    (np.abs(obj.eta) < 4.7) & 
                    (obj.pt > 30)  & 
                    ((obj.jet_id==6)) & #tight
                    (obj.mass > -1) 
                )
                                
            elif name == 'muon':
                selection = (
                    (np.abs(obj.eta) < 2.4) & (obj.pt > 15)  &
                    (obj['iso'] < 0.15) &
                    (obj['isLoose']) &
                    (obj.mass > -1)  
                )
                
            elif name == 'medium_muon':
                selection = (
                    (np.abs(obj.eta) < 2.5) & 
                    (obj.pt > 20)  &
                    (obj['iso'] < 0.15) &
                    (obj['isMedium']) &
                    (obj.mass > -1) 
                )

            elif name == 'tight_muon':
                selection = (
                    (np.abs(obj.eta) < 2.4) & 
                    (obj.pt > 20)  &
                    (obj['iso'] < 0.15) &
                    (obj['isTight']) &
                    (obj.mass > -1) 
                )
                
            elif name == 'electron':
                selection = (
                    (np.abs(obj.eta) < 2.4) & (obj.pt > 10)  &

                    (
                        ((np.abs(obj.eta) < 1.4442) & (np.abs(obj['dxy']) < 0.05) & (np.abs(obj['dz']) < 0.1) ) 
                        | ((np.abs(obj.eta) > 1.5660) & (np.abs(obj['dxy']) < 0.1) & (np.abs(obj['dz']) < 0.2) )
                    ) & 
                    (obj['cutbased']>=1) &
                    (obj.mass >-1)        
                )
        return obj[selection],selection
    
