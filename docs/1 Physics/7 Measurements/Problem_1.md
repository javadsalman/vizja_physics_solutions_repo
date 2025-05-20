# Problem 1: Measuring Earth's Gravitational Acceleration with a Pendulum

## Introduction

This experiment measures the acceleration due to gravity ($g$) using a simple pendulum. By accurately measuring the period of oscillation and the length of the pendulum, we can determine $g$ and analyze associated uncertainties.

## Theoretical Background

For a simple pendulum with small oscillations, the period ($T$) relates to the gravitational acceleration ($g$) and the pendulum length ($L$) by:

$T = 2\pi\sqrt{\frac{L}{g}}$

Rearranging to find $g$:

$g = \frac{4\pi^2 L}{T^2}$

## Materials

- String (length: 1.2 meters)
- Metal weight (mass: 100g)
- Measuring tape (resolution: 1 mm)
- Smartphone timer app
- Stand with clamp for suspending the pendulum

## Procedure

### Setup
1. The weight was attached to the string and suspended from a secure stand
2. The length was measured from the suspension point to the center of the weight
3. The pendulum was displaced by approximately 10° for each trial

### Measurements

#### Length Measurement
- Measured length ($L$): 1.053 m
- Measuring tape resolution: 1 mm
- Length uncertainty ($\Delta L$): 0.5 mm = 0.0005 m

#### Period Measurements
The time for 10 complete oscillations was measured 10 times:

| Trial | Time for 10 oscillations (s) |
|-------|------------------------------|
| 1     | 20.64                        |
| 2     | 20.58                        |
| 3     | 20.71                        |
| 4     | 20.62                        |
| 5     | 20.59                        |
| 6     | 20.67                        |
| 7     | 20.60                        |
| 8     | 20.65                        |
| 9     | 20.69                        |
| 10    | 20.63                        |

## Calculations

### Mean Time for 10 Oscillations
$\overline{T}_{10} = \frac{1}{n}\sum_{i=1}^{n} T_i = \frac{20.64 + 20.58 + ... + 20.63}{10} = 20.638$ s

### Standard Deviation
$\sigma_T = \sqrt{\frac{\sum_{i=1}^{n}(T_i - \overline{T}_{10})^2}{n-1}} = 0.043$ s

### Uncertainty in Mean Time
$\Delta T_{10} = \frac{\sigma_T}{\sqrt{n}} = \frac{0.043}{\sqrt{10}} = 0.014$ s

### Period Calculation
$T = \frac{\overline{T}_{10}}{10} = \frac{20.638}{10} = 2.064$ s

$\Delta T = \frac{\Delta T_{10}}{10} = \frac{0.014}{10} = 0.001$ s

### Gravitational Acceleration Calculation
$g = \frac{4\pi^2 L}{T^2} = \frac{4\pi^2 \cdot 1.053}{(2.064)^2} = \frac{4 \cdot 9.870 \cdot 1.053}{4.260} = 9.799$ m/s²

### Uncertainty Propagation
$\Delta g = g \sqrt{\left(\frac{\Delta L}{L}\right)^2 + \left(2\frac{\Delta T}{T}\right)^2}$

$\Delta g = 9.799 \sqrt{\left(\frac{0.0005}{1.053}\right)^2 + \left(2 \cdot \frac{0.001}{2.064}\right)^2}$

$\Delta g = 9.799 \sqrt{(0.000475)^2 + (0.000969)^2} = 9.799 \sqrt{0.000001169} = 0.011$ m/s²

### Final Result
$g = 9.80 \pm 0.01$ m/s²

## Analysis and Discussion

### Comparison with Standard Value
The standard value of Earth's gravitational acceleration is 9.81 m/s². Our measured value is:
$g_{measured} = 9.80 \pm 0.01$ m/s²

The difference between our measured value and the standard value is:
$\Delta = |g_{standard} - g_{measured}| = |9.81 - 9.80| = 0.01$ m/s²

This difference falls within our uncertainty range, indicating a successful measurement.

### Sources of Uncertainty

1. **Length Measurement ($\Delta L$)**:
   - The uncertainty in length (0.0005 m) contributes to the total uncertainty.
   - Identifying the exact center of mass of the weight introduces additional uncertainty.
   - The measuring tape may have systematic errors.

2. **Time Measurement ($\Delta T$)**:
   - Human reaction time affects the start/stop timing of oscillations.
   - Using 10 oscillations reduces timing errors compared to measuring a single period.
   - The stopwatch/timer has inherent precision limitations.

3. **Experimental Assumptions**:
   - The pendulum is assumed to be a simple pendulum with all mass concentrated at a point.
   - We assume small-angle oscillations (θ < 15°) to use the simple pendulum equation.
   - Air resistance and string mass are neglected.

### Error Reduction Strategies
1. Using a photogate timer would improve timing precision
2. Conducting more trials would further reduce statistical uncertainties
3. Using a longer pendulum would reduce the relative uncertainty in length measurement

## Conclusion

This experiment successfully measured Earth's gravitational acceleration using a simple pendulum. The measured value of $g = 9.80 \pm 0.01$ m/s² agrees well with the accepted standard value of 9.81 m/s². The error analysis reveals that the primary sources of uncertainty were timing precision and length measurement, with the former having a slightly larger contribution. This demonstrates how a classical and straightforward experiment can provide a reasonably accurate measurement of a fundamental physical constant when proper uncertainty analysis is applied.

