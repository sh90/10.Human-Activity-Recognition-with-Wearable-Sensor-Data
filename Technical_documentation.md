## What is “time-series data”?

***Definition:*** Data collected as an ordered sequence over time. Each value is tied to a timestamp (e.g., every 20 ms).

***Examples:*** Heart rate every second, stock prices each minute, accelerometer readings 50 times per second.

## Why it’s special: 
Order matters. We care about patterns across time—rhythm, trends, repetitions—not just individual numbers.

https://www.influxdata.com/what-is-time-series-data/ 

## Key terms 

Sampling rate (Hz): How many measurements per second (e.g., 50 Hz = 50 samples/sec).

Window: A short slice of the stream (e.g., 2 seconds) used by the model to “hear” the rhythm.

Sequence: The ordered list of points in a window.

<img width="1543" height="793" alt="SamplingRate_Wearable_Demo_v2" src="https://github.com/user-attachments/assets/e9154a70-b408-4eb0-82df-d7d3f29b5b2c" />

## What are ax, ay, az?

They are the three accelerometer channels—acceleration along three perpendicular axes of your device:

ax: left–right

ay: forward–back

az: up–down (often shows gravity most strongly)

Think of the phone/watch as a tiny box with arrows pointing in 3 directions (x, y, z). The sensor reports how much the box accelerates along each arrow.
<img width="814" height="706" alt="Axes_Diagram" src="https://github.com/user-attachments/assets/cbc8d182-6933-4711-bcb7-73d8b4472d21" />

## Units & gravity

Reported in g (1 g ≈ gravity ≈ 9.81 m/s²) or directly in m/s².

If the device is still on a table, you’ll often see az ≈ 1 g (gravity), ax ≈ 0, ay ≈ 0—depending on orientation.

Sitting: small random wiggles; one axis often near ~1 g due to gravity.

Walking: smooth, periodic up-down/side-to-side pattern at a moderate frequency.

Running: same pattern but faster and larger amplitude.
<img width="1260" height="716" alt="TimeSeries_Sample" src="https://github.com/user-attachments/assets/88158603-f5c7-45c3-ac9d-9cd3a5815042" />


## How ax/ay/az look for different activities

Mini example (first 6 rows, 50 Hz)
time (s)	ax	ay	az	label
0.00	0.02	0.01	0.98	sitting
0.02	0.01	0.00	1.01	sitting
0.04	0.03	0.02	1.00	sitting
0.06	0.35	0.10	1.10	walking
0.08	0.30	0.15	1.05	walking
0.10	0.25	0.20	1.00	walking

(Numbers are illustrative.)

## Why we “window” time-series

The model needs a short context (e.g., 2 seconds) to sense the rhythm:

Split the stream into overlapping windows (e.g., 100 samples per window at 50 Hz).

Feed each window’s sequence [(ax,ay,az)_t=1 … (ax,ay,az)_t=100] to the model.

Model predicts a label for the whole window (sitting / walking / running).

## Gotchas

Orientation matters: Pocket vs wrist changes which axis sees gravity.

Noise is normal: Small random fluctuations; we handle it with scaling and augmentation.

Order matters: Shuffling time destroys meaning—always keep timestamps in order!

- Time‑series means *ordered over time*; order matters. We need a short **window** to hear the rhythm.
- Sampling rate: 50 Hz = 50 readings each second; more samples capture more detail but cost more battery.
- The accelerometer gives three channels: **ax, ay, az** — left/right, forward/back, up/down.
- When the device is still, **az ≈ 1g** due to gravity; ax, ay near zero (depends on orientation).
- **Sitting**: small wiggles; **Walking**: rhythmic waves; **Running**: faster and larger waves.
- Orientation matters (pocket vs wrist), so we train with augmentation (noise/scale/time‑warp).
