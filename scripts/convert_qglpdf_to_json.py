import uproot
import correctionlib
from correctionlib import convert
from correctionlib.schemav2 import CorrectionSet, Correction, Binning, Variable
from coffea.lookup_tools import extractor
import gzip

corr_list = []

ext1 = extractor()
ext1.add_weight_sets(["* * /ssd-home/kimmokal/QGval/pdfs/pdfQG_AK4chs_13TeV_UL17_ghosts.root"])
ext1.finalize()
evaluator1 = ext1.make_evaluator()

num_keys = int(len(evaluator1.keys())/2) - 2

counter = 1
skip = ['_error', 'hpt', 'heta']
for key in evaluator1.keys():
    if any(x in key for x in skip):
        continue
    else:
        print('{k}, {c}/{t}'.format(k=key, c=counter, t=num_keys))
        hist_path = '/ssd-home/kimmokal/QGval/pdfs/pdfQG_AK4chs_13TeV_UL17_ghosts.root:{hist}'.format(hist=key)
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
    "schema_version": 2,
    "corrections": corr_list,
})

with open('pdfQG_AK4chs_13TeV_UL17_ghosts.corr.json', 'w') as fout:
    fout.write(cset.json(exclude_unset=True, indent=4))
