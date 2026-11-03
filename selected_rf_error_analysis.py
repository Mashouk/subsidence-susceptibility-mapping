"""Error analysis of the SELECTED model (Random Forest, 100 trees, min_samples_leaf=5, seed 42) on the
block-grouped validation folds. Same code path as Part A (Setup to Run 2) of training_experiments.ipynb.
Run from the repository root:  python selected_rf_error_analysis.py"""
import numpy as np, pandas as pd, warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
SEED=42; CLASSES=["very_low","low","moderate","high","very_high"]; CID={c:i for i,c in enumerate(CLASSES)}
df=pd.read_csv("data/subsidence_dataset_processed.csv")
dev=df[df.susceptibility_class.notna() & (df.split=="train")].reset_index(drop=True)
y=dev.susceptibility_class.map(CID).to_numpy(); groups=dev.block_id.to_numpy()
NUM=["n_years_cropped","n_years_fallow","is_perennial_crop","land_use_changed","clay_pct","sand_pct","silt_pct","organic_matter_pct","ksat_um_per_s","available_water_capacity","elevation_m","slope_deg","gw_decline_m_per_yr_2016_2019","gw_mean_depth_m","gw_latest_depth_m","gw_depth_std_m","gw_n_measurements","dist_to_nearest_well_m","soil_missing"]
CAT=["drainagecl","hydgrp","taxclname","land_use_name_last_year"]; X=dev[NUM+CAT]
def mk(): return Pipeline([("prep",ColumnTransformer([("num",SimpleImputer(strategy="median"),NUM),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),CAT)])),("rf",RandomForestClassifier(n_estimators=100,min_samples_leaf=5,random_state=SEED,n_jobs=1))])
pred=np.zeros(len(y),int); proba=np.zeros((len(y),5))
for tr,va in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups):
    m=mk().fit(X.iloc[tr],y[tr]); pred[va]=m.predict(X.iloc[va]); proba[va]=m.predict_proba(X.iloc[va])
print("macro-F1 %.4f (must equal 0.3692)"%f1_score(y,pred,average="macro"))
conf=proba.max(1); wrong=pred!=y; miss=np.abs(y-pred)
print("wrong %d of %d | off by 1: %d | off by 2+: %d | very_low<->very_high: %d"%(wrong.sum(),len(y),(miss==1).sum(),(miss>=2).sum(),((y==0)&(pred==4)).sum()+((y==4)&(pred==0)).sum()))
print("mean confidence %.3f vs accuracy %.3f"%(conf.mean(),(~wrong).mean()))
for lo,hi in [(0,.35),(.35,.5),(.5,.65),(.65,1.01)]:
    s=(conf>=lo)&(conf<hi); print("conf %.2f-%.2f: %4d cells, accuracy %.3f"%(lo,hi,s.sum(),(~wrong[s]).mean() if s.sum() else float('nan')))
for t in (.65,.8):
    s=conf>=t; print("conf>=%.2f: %d cells, %d wrong (%.1f%%), blocks with a wrong one: %d"%(t,s.sum(),(wrong&s).sum(),100*(wrong&s).sum()/max(s.sum(),1),len(set(groups[wrong&s]))))
cw=pd.Series(groups[wrong&(conf>=.8)]).value_counts().head(4); print("largest blocks among wrong conf>=0.8:",cw.to_dict())
sup=pd.cut(dev.insar_n_points,[0,1,2,5,10,30],labels=["1","2","3-5","6-10","11+"])
print(pd.DataFrame({"cells":pd.Series(wrong).groupby(sup.values).size(),"error_rate":pd.Series(wrong).groupby(sup.values).mean()}).round(3))
print("share of cells predicted in each class:",np.bincount(pred,minlength=5).tolist())
