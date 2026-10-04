from fixed_core import *
def savej(p,v):
    if p.exists():require(readj(p)==clean(v),'existing JSON mismatch '+str(p))
    else:writej(p,v)

def saven(p,**data):
    if p.exists():
        with np.load(p) as z:
            require(set(z.files)==set(data) and all(np.array_equal(z[k],v) for k,v in data.items()),'existing NPZ mismatch '+str(p))
    else:np.savez(p,**data)

def savecsv(p,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    text=frame.to_csv(index=False,na_rep='NA')
    if p.exists():require(p.read_text()==text,'existing CSV mismatch '+str(p))
    else:p.write_text(text)