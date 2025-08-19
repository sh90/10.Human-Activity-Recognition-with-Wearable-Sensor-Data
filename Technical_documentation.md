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

## What are ax, ay, az?

They are the three accelerometer channels—acceleration along three perpendicular axes of your device:

ax: left–right

ay: forward–back

az: up–down (often shows gravity most strongly)

Think of the phone/watch as a tiny box with arrows pointing in 3 directions (x, y, z). The sensor reports how much the box accelerates along each arrow.

## Units & gravity

Reported in g (1 g ≈ gravity ≈ 9.81 m/s²) or directly in m/s².

If the device is still on a table, you’ll often see az ≈ 1 g (gravity), ax ≈ 0, ay ≈ 0—depending on orientation.

## How ax/ay/az look for different activities

Sitting: small random wiggles; one axis often near ~1 g due to gravity.

Walking: smooth, periodic up-down/side-to-side pattern at a moderate frequency.

Running: same pattern but faster and larger amplitude.
