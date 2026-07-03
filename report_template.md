# Modeling Human Activity States Using Hidden Markov Models

**Author:** Samuel Mwania
**Date:** July 2026

---

## 1. Background and Motivation

Human Activity Recognition (HAR) is a growing field that uses sensor data from smartphones and wearable devices to identify physical activities like walking, standing, jumping, and resting. Accurate activity classification has direct applications in personal fitness tracking and rehabilitation monitoring, where patients recovering from injuries need their exercises logged automatically to track compliance.

This project focuses on **Personal Fitness and Rehabilitation Monitoring**. The goal is to build a Hidden Markov Model from scratch that can accurately infer a person's activity state from continuous streams of accelerometer and gyroscope data. The HMM treats the true activity as a hidden state and the sensor-derived features as observable emissions. The Baum-Welch algorithm handles training, and the Viterbi algorithm handles decoding the most likely activity sequence.

---

## 2. Data Collection and Preprocessing

### 2.1 Data Collection Setup

Data was collected using the **Sensor Logger** app on an Android smartphone (CPH2573). The sampling rate was set to **50 Hz** (20 ms sample interval). This rate was chosen because human motion typically occurs below 20 Hz, so by the Nyquist theorem, 50 Hz captures all relevant motion frequencies without aliasing.

**Sampling Rate:**

| Contributor | Device | Sampling Rate |
|-------------|--------|---------------|
| Samuel Mwania | CPH2573 (Android) | 50 Hz |

Four activities were recorded:

| Activity | Description | Files | Duration per file |
|----------|-------------|-------|-------------------|
| Standing | Phone held steady at waist level | 20 | ~6-7 seconds |
| Walking | Consistent walking pace, phone at waist | 20 | ~6-7 seconds |
| Jumping | Continuous vertical jumps | 20 | ~6-7 seconds |
| Still | Phone flat on a table, no movement | 20 | ~6-7 seconds |

**Total:** 80 CSV files across 4 activities, each containing accelerometer (x, y, z) and gyroscope (x, y, z) readings with timestamps. Each activity has over 2 minutes of total recording time, comfortably exceeding the 1 minute 30 second minimum.

### 2.2 Raw Data Visualization

[INSERT FIGURE: raw_sensor_data.png]
*Figure 1: Sample accelerometer and gyroscope signals for each activity. Jumping shows large spikes, walking shows periodic patterns, while standing and still are relatively flat.*

### 2.3 Windowing

Each recording was divided into **1-second non-overlapping windows**. At 50 Hz, each window contains about 50 data points. This window size was chosen because:
- 50 samples are enough to compute stable statistics (mean, variance, RMS)
- One second captures at least one full cycle of walking (~2 Hz) or jumping (~2-3 Hz) through the FFT
- The time resolution is fine enough to detect transitions between activities

### 2.4 Feature Extraction

38 features were extracted per window:

**Time-Domain Features (5 per axis, across 6 axes = 30 features):**

| Feature | Why It Helps |
|---------|--------------|
| Mean | Captures average sensor reading. Gravity causes standing to have a specific mean on the z-axis. |
| Variance | Measures signal spread. Jumping has high variance, still has near-zero variance. |
| Standard Deviation | Same as variance but in original units, easier to interpret. |
| Root Mean Square (RMS) | Captures signal energy regardless of sign. Walking has moderate RMS, jumping has high RMS. |
| Signal Magnitude Area (SMA) | Total body acceleration in one number. Strongly separates dynamic from static activities. |

**Frequency-Domain Features (2 per axis, across 6 axes = 12 features, derived from FFT):**

| Feature | Why It Helps |
|---------|--------------|
| Dominant Frequency | Walking peaks at ~2 Hz, jumping at ~2-3 Hz, still at ~0 Hz. Directly identifies periodic motion. |
| Spectral Energy | Total frequency-domain energy. High for dynamic activities, low for static ones. |

**Cross-Axis Feature (1 feature):**

| Feature | Why It Helps |
|---------|--------------|
| Acc XY Correlation | Walking creates coordinated x-y motion (correlated), standing produces uncorrelated noise. |

### 2.5 Feature Normalization

All features were normalized using **Z-score standardization**, which gives each feature zero mean and unit variance. This prevents features with naturally large values (like spectral energy) from dominating the K-Means clustering used for discretization.

---

## 3. HMM Setup and Implementation Details

### 3.1 Model Components

| Component | Symbol | Description | Setup |
|-----------|--------|-------------|-------|
| Hidden States | Z | True activities being performed | 4 states: Standing, Walking, Jumping, Still |
| Observations | X | Sensor features per time window | 20 discrete symbols from K-Means quantization |
| Transition Matrix | A | Probability of switching between states | 4x4 matrix, initialized with 0.85 diagonal |
| Emission Matrix | B | Probability of observing a symbol in a state | 4x20 matrix, initialized from training labels |
| Initial Probabilities | pi | Probability of starting in each state | Uniform: [0.25, 0.25, 0.25, 0.25] |

The continuous 38-dimensional feature vectors were converted into 20 discrete symbols using K-Means clustering (Vector Quantization).

### 3.2 Algorithm Implementation

The HMM was implemented entirely from scratch using NumPy:

1. **Forward Algorithm** with scaling factors at each time step to prevent numerical underflow
2. **Backward Algorithm** with corresponding scaling
3. **Baum-Welch Algorithm** (Expectation-Maximization) for iterative parameter estimation, with a convergence check that stops when the log-likelihood change drops below 0.001
4. **Viterbi Algorithm** in log-space for decoding the most likely state sequence

### 3.3 Parameter Initialization

- **Transition Matrix (A):** 0.85 on the diagonal, 0.05 off-diagonal. This reflects that people tend to stay in the same activity for extended periods.
- **Emission Matrix (B):** Initialized from the training label distributions with Laplace smoothing to avoid zero probabilities.
- **Initial Distribution (pi):** Uniform at 0.25 per state.

### 3.4 Training Convergence

[INSERT FIGURE: convergence_plot.png]
*Figure 2: Log-likelihood across Baum-Welch iterations. The curve flattens when convergence is reached.*

The data was split 80% training (64 sessions) and 20% testing (16 sessions), with the split done at the session level to prevent data leakage.

---

## 4. Results and Interpretation

### 4.1 Transition Probability Matrix

[INSERT FIGURE: transition_matrix.png]
*Figure 3: Learned transition probability matrix. High diagonal values confirm the model learned that people stay in the same activity.*

### 4.2 Emission Probability Matrix

[INSERT FIGURE: emission_matrix.png]
*Figure 4: Emission probability matrix. Different activities produce distinct emission patterns.*

### 4.3 Confusion Matrix

[INSERT FIGURE: confusion_matrix.png]
*Figure 5: Confusion matrix on unseen test data (16 sessions, 100 windows).*

### 4.4 Performance Metrics

| State (Activity) | Number of Samples | Sensitivity | Specificity | Overall Accuracy |
|-------------------|-------------------|-------------|-------------|------------------|
| Standing | 18 | 100.00% | 96.34% | 97.00% |
| Walking | 32 | 90.62% | 100.00% | 97.00% |
| Jumping | 20 | 100.00% | 100.00% | 100.00% |
| Still | 30 | 100.00% | 100.00% | 100.00% |

**Overall Model Accuracy: 97.00%**

### 4.5 Decoded Activity Sequences

[INSERT FIGURE: decoded_sequences.png]
*Figure 6: Viterbi-decoded predictions vs true labels for test sessions.*

The test data was obtained by holding out 20% of recording sessions that were never seen during training. This strict session-level split tests whether the model generalizes to completely new recordings.

---

## 5. Discussion and Conclusion

### 5.1 Which Activities Were Easiest or Hardest to Distinguish?

"Jumping" and "Still" were the easiest to classify. Jumping produces very high variance and spectral energy due to repeated vertical motion. Still produces near-zero variance since the phone sits on a flat surface with no movement.

"Standing" and "Still" could potentially be confused since both involve minimal movement. However, the gyroscope captures subtle body sway during standing (small balance adjustments), which is absent when the phone rests on a table. The model picked up on this difference.

### 5.2 How Transition Probabilities Reflect Realistic Behavior

The learned transition matrix shows high values along the diagonal, meaning the model learned that people continue the same activity for multiple consecutive seconds. The off-diagonal values are small but not zero, correctly representing that transitions happen but are relatively rare in short recordings.

### 5.3 Effect of Sensor Noise and Sampling Rate

The 50 Hz sampling rate was more than enough for human motion (below 20 Hz). Sensor noise was visible as small fluctuations, but the windowed feature extraction averaged it out by computing statistics over 50-sample windows. Z-score normalization also helped by standardizing all feature scales.

### 5.4 Possible Improvements

1. Longer recordings (30+ seconds each) would provide more training windows per session.
2. Using 50% overlapping windows would double the training data without new recordings.
3. Replacing K-Means discretization with Gaussian Mixture Models (GMM-HMM) would preserve more information from the continuous features.
4. Adding a barometer sensor could detect altitude changes like stair climbing.
5. Extending to more activities (running, sitting, stair climbing) would make the system more practical for rehabilitation monitoring.

---

## 6. Task Allocation

This project was completed individually as per the instructor's updated guidelines.

| Contributor | Tasks Completed | Contribution |
|-------------|-----------------|--------------|
| Samuel Mwania | Data collection, preprocessing, feature extraction, HMM implementation (Baum-Welch and Viterbi from scratch), model evaluation, visualization, and report writing | 100% |
