"""Summarize timing and exact/tolerant comparison to the baseline two-update state."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference,candidate):
    paths_file=Path(candidate).parent/'probe-paths.json'
    paths=json.loads(paths_file.read_text()) if paths_file.exists() else {}
    with np.load(reference) as a,np.load(candidate) as b:
        if a.files!=b.files:return dict(compatible=False,reason='leaf count/names')
        exact=True;close=True;max_abs=0.;integer_differences=0;floating_differences=0
        parameter_max=0.;parameter_square=0.;parameter_elements=0
        for key in a.files:
            x,y=a[key],b[key]
            if x.shape!=y.shape or x.dtype!=y.dtype:
                return dict(compatible=False,reason=f'{key} shape/dtype')
            same=np.array_equal(x,y)
            exact &= same
            if np.issubdtype(x.dtype,np.inexact):
                floating_differences+=int(not same)
                if x.size:max_abs=max(max_abs,float(np.max(np.abs(x-y))))
                if '.params[' in paths.get(key,''):
                    delta=x.astype(np.float64)-y.astype(np.float64)
                    parameter_max=max(parameter_max,float(np.max(np.abs(delta))))
                    parameter_square+=float(np.sum(delta*delta));parameter_elements+=delta.size
                close &= bool(np.allclose(x,y,rtol=2e-4,atol=2e-5))
            elif not same:
                integer_differences+=1;close=False
        return dict(compatible=True,bitwise_equal=exact,close=close,max_abs_difference=max_abs,
                    differing_integer_leaves=integer_differences,differing_float_leaves=floating_differences,
                    parameter_max_abs_difference=parameter_max if parameter_elements else None,
                    parameter_rms_difference=(parameter_square/parameter_elements)**.5 if parameter_elements else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    baseline=args.root/'baseline'
    base=json.loads((baseline/'result.json').read_text())['warmed_median_seconds_per_update']
    rows=[]
    for directory in sorted(args.root.iterdir()):
        if not directory.is_dir() or not (directory/'result.json').exists():continue
        result=json.loads((directory/'result.json').read_text())
        if 'warmed_median_seconds_per_update' not in result:continue
        row=dict(case=directory.name,median_seconds=result['warmed_median_seconds_per_update'],
                 speedup=base/result['warmed_median_seconds_per_update'],compile_seconds=result['compile_seconds'],
                 peak_device_bytes=result.get('device_memory_stats',{}).get('peak_bytes_in_use'))
        if (directory/'probe-update002.npz').exists():
            row['two_update_agreement']=compare(baseline/'probe-update002.npz',directory/'probe-update002.npz')
        rows.append(row)
    (args.root/'summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
