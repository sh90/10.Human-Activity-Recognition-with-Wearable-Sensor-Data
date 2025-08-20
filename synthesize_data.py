import math, numpy as np, pandas as pd

def synth_har(seconds=120, fs=50, seq=('sitting','walking','running'), seed=123):
    rng = np.random.default_rng(seed)
    rows = []; per = seconds // len(seq)
    t = 0
    for act in seq:
        for _ in range(per * fs):
            if act == 'sitting':
                ax = rng.normal(0.02, 0.03); ay = rng.normal(0.02, 0.03); az = rng.normal(1.0, 0.05)
            elif act == 'walking':
                f = 2.0
                ax = 0.4*math.sin(2*math.pi*f*(t/fs)) + rng.normal(0, 0.05)
                ay = 0.4*math.cos(2*math.pi*f*(t/fs)) + rng.normal(0, 0.05)
                az = 1.0 + 0.2*math.sin(2*math.pi*f*(t/fs)+0.5) + rng.normal(0, 0.05)
            else:  # running
                f = 3.5
                ax = 0.7*math.sin(2*math.pi*f*(t/fs)) + rng.normal(0, 0.08)
                ay = 0.7*math.cos(2*math.pi*f*(t/fs)) + rng.normal(0, 0.08)
                az = 1.0 + 0.35*math.sin(2*math.pi*f*(t/fs)+0.8) + rng.normal(0, 0.08)
            rows.append((t/fs, ax, ay, az, act)); t += 1
    return pd.DataFrame(rows, columns=['timestamp','ax','ay','az','label'])

df = synth_har(seconds=180, fs=50, seq=('sitting','walking','running'))
df.to_csv('my_synth.csv', index=False)


