"""Fuzzy Mamdani untuk interpretasi risiko kestabilan jaringan."""
import numpy as np

INPUTS = {
 "p_unstable":{"universe":(0.,1.),"sets":{
  "Low":(0.,0.,.20,.40),"Medium":(.25,.45,.55,.75),
  "High":(.60,.75,.85,.95),"VeryHigh":(.85,.95,1.,1.)}},
 "stress":{"universe":(0.,1.),"sets":{
  "Low":(0.,0.,.25,.50),"Medium":(.25,.45,.55,.75),"High":(.50,.75,1.,1.)}}
}
OUTPUT = {"universe":(0.,100.),"sets":{
 "Secure":(0.,0.,15.,30.),"Alert":(20.,35.,45.,60.),
 "Unstable":(50.,65.,75.,88.),"Critical":(80.,90.,100.,100.)}}
RULES = [
 ({"p_unstable":"Low","stress":"Low"},"Secure"),
 ({"p_unstable":"Low","stress":"Medium"},"Secure"),
 ({"p_unstable":"Low","stress":"High"},"Alert"),
 ({"p_unstable":"Medium","stress":"Low"},"Alert"),
 ({"p_unstable":"Medium","stress":"Medium"},"Alert"),
 ({"p_unstable":"Medium","stress":"High"},"Unstable"),
 ({"p_unstable":"High","stress":"Low"},"Unstable"),
 ({"p_unstable":"High","stress":"Medium"},"Unstable"),
 ({"p_unstable":"High","stress":"High"},"Critical"),
 ({"p_unstable":"VeryHigh","stress":"Low"},"Critical"),
 ({"p_unstable":"VeryHigh","stress":"Medium"},"Critical"),
 ({"p_unstable":"VeryHigh","stress":"High"},"Critical")]
RECOMMENDATION = {
 "Secure":"Jaringan diperkirakan stabil. Lanjutkan operasi dan monitoring rutin.",
 "Alert":"Terdapat indikasi risiko. Tingkatkan pemantauan kondisi jaringan.",
 "Unstable":"Jaringan berpotensi tidak stabil. Evaluasi parameter operasi dan siapkan mitigasi.",
 "Critical":"Risiko sangat tinggi. Prioritaskan tindakan mitigasi dan evaluasi beban jaringan."}

def trapmf(x,a,b,c,d):
    x=np.asarray(x,dtype=float)
    left=np.ones_like(x) if a==b else np.clip((x-a)/(b-a),0,1)
    right=np.ones_like(x) if c==d else np.clip((d-x)/(d-c),0,1)
    return np.minimum(left,right)

def stress_index(tau_mean,g_mean):
    t=np.clip((float(tau_mean)-.5)/9.5,0,1)
    g=np.clip((float(g_mean)-.05)/.95,0,1)
    return float(.5*t+.5*g)

def infer(p_unstable,stress):
    p=float(np.clip(p_unstable,0,1)); s=float(np.clip(stress,0,1))
    universe=np.linspace(*OUTPUT["universe"],1001)
    agg=np.zeros_like(universe)
    for antecedent, consequent in RULES:
        vals={"p_unstable":p,"stress":s}
        strength=min(float(trapmf(vals[k],*INPUTS[k]["sets"][v])) for k,v in antecedent.items())
        mf=trapmf(universe,*OUTPUT["sets"][consequent])
        agg=np.maximum(agg,np.minimum(strength,mf))
    area=agg.sum()
    score=float((universe*agg).sum()/area) if area else 50.
    category=max(OUTPUT["sets"],key=lambda k:float(trapmf(score,*OUTPUT["sets"][k])))
    return {"score":score,"category":category,"universe":universe,"aggregated":agg,
            "recommendation":RECOMMENDATION[category]}
