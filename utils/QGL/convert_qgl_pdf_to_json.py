import uproot
import correctionlib
import gzip
import sys
import os

from pathlib import Path
from correctionlib import convert
from correctionlib.schemav2 import CorrectionSet, Correction, Binning, Variable
from coffea.lookup_tools import extractor

def main(qgl_file):
    corr_list = []
    
    ext1 = extractor()
    ext1.add_weight_sets([f'* * {qgl_file}'])
    ext1.finalize()
    evaluator1 = ext1.make_evaluator()
    
    num_keys = int(len(evaluator1.keys())/2)
    
    counter = 1
    skip = ['_error', 'hpt', 'heta']
    for key in evaluator1.keys():
        if any(x in key for x in skip):
            continue
        else:
            print(f'{key}, {counter}/{num_keys}')
            hist_path = f'{qgl_file}:{key}'
            hist = uproot.open(hist_path)
            corr = convert.from_histogram(hist)
    
            corr_new_data = Binning.parse_obj({
                    'nodetype' : 'binning',
                    'input' : key.split('/')[0],
                    'edges' : corr.data.edges,
                    'content' : corr.data.content,
                    'flow' : -1.0,
            })
    
            corr_new_variable = Variable.parse_obj({
                    'name' : key.split('/')[0],
                    'type' : 'real',
            })
    
            corr_new_name = Correction.parse_obj({
                    'version' : 2,
                    'name' : key,
                    'description' : corr.description,
                    'inputs' : [corr_new_variable], 
                    'output' : corr.output,
                    'data' : corr_new_data,
            })
    
            corr_list.append(corr_new_name)
            counter = counter + 1
    
    cset = CorrectionSet.parse_obj({
        'schema_version': 2,
        'corrections': corr_list,
    })
    
    save_path = os.path.join(os.environ['COFFEAHOME'],'utils','QGL')
    json_file_name = qgl_file.name.replace('.root', '.corr.json')

    with open(f'{save_path}/{json_file_name}', 'w') as fout:
        fout.write(cset.json(exclude_unset=True, indent=4))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('ERROR: Please provide an input file as the first argument: python convert_qgl_pdf_to_json.py <filename>')
    if 'COFFEAHOME' not in os.environ:
        sys.exit('ERROR: Enviroment variables not set. Run activate_setup.sh first!')

    qgl_file = Path(sys.argv[1])
    main(qgl_file)
