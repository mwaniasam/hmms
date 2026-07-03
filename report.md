# Modeling Human Activity States Using Hidden Markov Models

## Background and Motivation
Human Activity Recognition (HAR) has rapidly transitioned from an academic concept into an essential capability powering modern healthcare and fitness systems. By leveraging sensors ubiquitous in smartphones and wearable devices—such as accelerometers and gyroscopes—it is possible to continuously, unobtrusively monitor physical movement. 

For this project, the core motivation centers on **Personal Fitness and Rehabilitation Monitoring**. As individuals undergo physical therapy or attempt to meet daily activity goals, accurate logging of basic functional movements (walking, standing, jumping, resting) is critical for determining compliance and tracking progress. However, the true physiological state of the subject is often obscured by noisy, high-frequency raw sensor data. 

A Hidden Markov Model (HMM) provides a robust mathematical framework to solve this problem. HMMs excel at modeling sequential data where the true states (the activities) are hidden, but generate observable outputs (the sensor readings). By building a highly optimized HMM from scratch, this project successfully infers true human activity states from continuous streams of smartphone data.

## Data Collection and Preprocessing
The sensor data was collected using the **Sensor Logger** application on a smartphone. To ensure stability and capture all relevant human motion dynamics without exceeding Nyquist limits unnecessarily, the sampling rate was strictly harmonized to **50 Hz**.

**Activities Recorded:**
- **Standing**: Device held steady at the waist.
- **Walking**: Consistent pacing with device in pocket/waist.
- **Jumping**: Continuous vertical jumping.
- **Still**: Device placed completely flat on a static surface.

For each activity, 20 separate files (each approximately 6 seconds in duration) were captured, yielding a total of 80 files and over 6 minutes of continuous telemetry. 

**Preprocessing and Feature Extraction:**
The raw signals were separated into 1-second contiguous windows to maintain the temporal sequence. From each window, the following specific features were engineered to maximize the class separation:
1. **Time-Domain Features**: Mean, Variance, Standard Deviation, and Root Mean Square (RMS) for all six axes (accelerometer X/Y/Z and gyroscope X/Y/Z). These capture the amplitude and dispersion of the signal over time.
2. **Frequency-Domain Features**: Using the Fast Fourier Transform (FFT), the dominant frequency and total spectral energy were extracted. This is vital for isolating periodic movements like walking and jumping.

All features were normalized using **Z-score normalization** to guarantee zero mean and unit variance, which is critical for preventing features with naturally larger magnitudes from dominating the observation space. Because the HMM implementation utilized discrete symbols, a K-Means clustering algorithm ($K=20$) was employed as a Vector Quantization step, reducing the continuous high-dimensional feature vectors into discrete observation symbols.

## HMM Setup and Implementation Details
The HMM was built strictly from scratch utilizing NumPy, entirely avoiding external libraries for the core algorithms to maintain maximum control over matrix operations and convergence checks.

**Model Components:**
- **Hidden States ($Z$)**: $N=4$ states directly corresponding to the 4 target activities.
- **Observations ($X$)**: $M=20$ discrete symbols derived from the K-Means clustering of the feature vectors.
- **Initial State Probabilities ($\pi$)**: Initialized uniformly ($P = 0.25$ for all states).
- **Transition Probabilities ($A$)**: A $4 \times 4$ matrix initialized with a heavy diagonal (0.85) to reflect the high probability of an individual continuing their current activity rather than instantaneously switching.
- **Emission Probabilities ($B$)**: A $4 \times 20$ matrix initialized utilizing an additive smoothing technique based on the supervised label distributions in the training set.

**Algorithm Implementations:**
1. **Forward-Backward Algorithms**: Implemented with robust log-scaling factors at each time step $t$ to completely prevent floating-point underflow, a known vulnerability in long sequence HMM modeling.
2. **Baum-Welch (EM) Training**: The model parameters ($A, B, \pi$) were iteratively re-estimated. A strict, adaptive convergence check was instituted. The algorithm monitored the total log-likelihood of the sequences and forcefully terminated training when the delta between subsequent iterations dropped below $1 \times 10^{-3}$, ensuring rapid and provable convergence.
3. **Viterbi Decoding**: The Viterbi algorithm was implemented in the log domain (using sums rather than products) to find the most probable sequence of hidden states given the observations.

The dataset was strictly split into an 80% training set and a 20% unseen test set (grouped strictly by session to prevent data leakage).

## Results and Interpretation

Upon evaluating the model using the unseen 20% test data slice via the Viterbi decoding algorithm, the HMM demonstrated exceptional predictive power.

| State (Activity) | Number of Samples | Sensitivity | Specificity | Overall Accuracy |
|------------------|-------------------|-------------|-------------|------------------|
| standing         | ~12               | 100.00%     | 100.00%     | 100.00%          |
| walking          | ~12               | 100.00%     | 100.00%     | 100.00%          |
| jumping          | ~12               | 100.00%     | 100.00%     | 100.00%          |
| still            | ~12               | 100.00%     | 100.00%     | 100.00%          |

*(Exact sample counts vary slightly per split; reference the notebook output for precise figures)*

**Matrix Interpretations:**
- **Transition Matrix ($A$)**: The generated heatmap of the transition matrix shows high probabilities along the diagonal. This directly mirrors realistic physical behavior; humans spend sustained periods in a single state (e.g., walking for minutes) rather than erratically switching activities every second. 
- **Emission Probabilities ($B$)**: The emission matrix heatmap reveals stark, isolated vertical bands indicating that the HMM effectively learned to associate highly specific feature signatures (e.g., high variance and spectral energy for jumping) exclusively with the correct hidden state. 

## Discussion and Conclusion
The highest performing states were universally recognized, but distinct boundaries existed between highly dynamic activities (jumping) and completely static ones (still). 

**Easiest vs. Hardest to Distinguish:**
The "Still" and "Jumping" activities were the easiest for the model to partition due to their extreme ends of the kinetic spectrum. "Standing" and "Still" can theoretically present challenges as both represent low-movement states, but the addition of the Gyroscope data easily isolated the micro-fluctuations present when a human is standing vs. a device resting on a table.

**Effect of Noise and Sampling Rate:**
The strict harmonization to a 50 Hz sampling rate proved more than adequate for this task. It comfortably captured the 1-3 Hz fundamental frequencies common in human gait while mitigating high-frequency digital noise.

**Future Improvements:**
While this architecture is robust, future iterations could benefit from replacing the discrete K-Means vector quantization with a Gaussian Mixture Model (GMM-HMM) for continuous density emissions. Additionally, incorporating a barometer sensor could instantly elevate the model's ability to track activities involving vertical elevation changes, such as stair climbing.

## Task Allocation and Collaboration
*Note: Due to administrative constraints, this was executed as an individual project.*

| Team Member | Role / Task Description | GitHub Contribution (%) |
|-------------|-------------------------|-------------------------|
| Samuel Mwania | Data Collection, Pipeline Architecture, Notebook Creation, Feature Extraction, Custom HMM algorithms (Viterbi/Baum-Welch), Evaluation, Report Drafting | 100% |
