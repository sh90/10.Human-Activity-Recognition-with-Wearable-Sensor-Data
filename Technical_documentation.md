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
