import json, sys, traceback, importlib.metadata, platform
from pathlib import Path
out=Path(__file__).resolve().parent
result={'python':sys.executable,'python_version':platform.python_version(),'versions':{}}
for name in ['numpy','cuml-cu12','cupy-cuda12x','scikit-learn','torch','pyarrow']:
    try: result['versions'][name]=importlib.metadata.version(name)
    except Exception as e: result['versions'][name]=str(e)
try:
    import pickle, cupy as cp
    with open('/home/karri/projects/NIDSProject_Professor_Run/outputs/source_v2/models/rf_All49/model.pkl','rb') as f: model=pickle.load(f)
    classes=model.classes_
    if hasattr(classes,'get'): classes=classes.get()
    result.update(status='PASS',backend=type(model).__module__+'.'+type(model).__name__,classes=classes.tolist(),n_features_in=int(model.n_features_in_),cuda_devices=cp.cuda.runtime.getDeviceCount(),cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion())
    assert result['classes']==[0,1] and result['n_features_in']==49
except Exception:
    result['status']='LOAD_BLOCKED'; result['exception']=traceback.format_exc()
print(json.dumps(result,ensure_ascii=False,indent=2))
(out/'model_load_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
