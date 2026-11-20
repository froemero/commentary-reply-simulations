# Extract CPz/Pz/CP1/CP2 at 100 Hz + stimulus/response events + behaviour per UCAP participant.
import scipy.io as sio, numpy as np, os, sys
D = os.path.expanduser("~/mnt/Data"); OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_local", "ucap")
B = sio.loadmat(f"{D}/UCAP_export/behav.mat", squeeze_me=True, struct_as_record=False)["a1"]
beh = {k: np.asarray(getattr(B, k)) for k in ("VPNummer", "n_b", "RT", "err", "DeviantPosRL")}
stimtrig = {f"S{t}" for t in list(range(201, 209)) + list(range(211, 219))}
chans = ["CPz", "Pz", "CP1", "CP2"]; dec = 5
files = sorted(f for f in os.listdir(f"{D}/UCAP_clean") if f.endswith("_clean.set"))
for f in files:
    vp = f.split("_")[0]; out = f"{OUT}/{vp}.npz"
    if os.path.exists(out): continue
    m = sio.loadmat(f"{D}/UCAP_clean/{f}", squeeze_me=True, struct_as_record=False)["EEG"]
    labs = [c.labels for c in m.chanlocs]; idx = [labs.index(c) for c in chans]
    X = np.memmap(f"{D}/UCAP_clean/{m.data}", dtype=np.float32, mode="r", shape=(m.pnts, m.nbchan))
    Y = np.asarray(X[:, idx], dtype=np.float64); n = (m.pnts // dec) * dec
    Y = Y[:n].reshape(-1, dec, len(idx)).mean(axis=1)
    ev = m.event; types = [str(e.type) for e in ev]; lat = np.array([float(e.latency) for e in ev])
    stim_i = [i for i, t in enumerate(types) if t in stimtrig]
    sel = beh["VPNummer"] == int(vp)
    nb, rt, err = beh["n_b"][sel], beh["RT"][sel], beh["err"][sel]
    k = min(len(stim_i), sel.sum())
    s_lat, r_lat = [], []
    for i in stim_i[:k]:
        s_lat.append(lat[i])
        r_lat.append(lat[i + 1] if i + 1 < len(types) and types[i + 1] in ("S 98", "S 99") else np.nan)
    bnd = lat[[i for i, t in enumerate(types) if t == "boundary"]]
    np.savez_compressed(out, eeg=Y.astype(np.float32), chans=chans, srate=m.srate / dec,
        stim_lat=(np.array(s_lat) - 1) / dec, resp_lat=(np.array(r_lat) - 1) / dec,
        n_b=np.asarray(nb[:k], dtype=str), rt=np.asarray(rt[:k], dtype=float), err=np.asarray(err[:k], dtype=str),
        boundary=(bnd - 1) / dec, n_stim_events=len(stim_i), n_behav=int(sel.sum()))
    print(vp, len(stim_i), int(sel.sum()), flush=True)
