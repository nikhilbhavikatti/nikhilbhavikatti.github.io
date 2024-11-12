---
title: "Exploring the Cosmos with AI: How Generative Models Create Realistic Galaxy Images and Improve Machine Learning Robustness"
date: 2024-11-12
permalink: /blog/exploring-the-cosmos-with-ai/
---

This post summarizes and explores insights from the paper *"Realistic galaxy images and improved robustness in machine learning tasks from generative modelling"* by Benjamin J. Holzschuh et al. As I delved into this research, I realized how impactful these methods can be - not only in creating visually realistic galaxy images but also in enhancing machine learning robustness. This paper examines how various generative models, particularly GANs and VAEs, are used to produce synthetic galaxy images that closely resemble real astronomical data. The result? A powerful tool for improving machine learning models used in astronomy, making them more resilient to domain shifts and noise. In this post, I'll break down the research, explain the models and methods, and discuss the implications and limitations of this approach.

---

## Understanding the Problem: Limited Data and Domain Shifts in Astronomy
One of the key challenges in astronomy, as highlighted in the paper, is the scarcity of high-quality observational data, especially for distant galaxies. Models trained on data from nearby galaxies often face a “domain shift” when applied to distant ones. This is problematic because these shifts can make models unreliable. For instance, a model trained to recognize low-noise images of galaxies nearby might perform poorly on noisier, low-resolution images from far-away galaxies.

The researchers propose using AI-generated images to fill this gap. By training generative models to mimic real galaxy distributions, we can supplement real data with high-quality synthetic images. These synthetic images help the model generalize better, making it more robust to changes in image quality and observational noise, which is crucial as new astronomical surveys bring in more data from the edges of the observable universe.

---

## The Generative Models Used: StyleGAN, VAE, and ALAE
The paper introduces three types of generative models, each with its unique strengths and limitations:

1. **Variational Autoencoders (VAE)**: This model is a foundational tool in generating synthetic data. It works by learning a compressed (latent) representation of the data and then reconstructing it. While VAEs are stable to train, they tend to produce images that are slightly blurry, which is a limitation in creating high-quality synthetic galaxy images.

2. **StyleGAN**: StyleGAN is known for generating photorealistic images in other domains, such as human faces. Its strength lies in its ability to produce crisp images with fine details. This model was the standout performer in this study, producing galaxy images that closely resembled real data across various metrics. However, training StyleGAN requires substantial computational resources, which could be a limitation in scaling up.

3. **Adversarial Latent Autoencoders (ALAE)**: This model combines elements of both GANs and VAEs, aiming to strike a balance between image quality and training stability. The ALAE didn’t outperform StyleGAN but showed promise, particularly in its ability to represent certain galaxy features accurately.

---

## Datasets Used in Training and Evaluation
The study worked with three datasets, each serving a specific purpose in evaluating the models:

1. **Sérsic Profiles**: These profiles represent basic, synthetic galaxy shapes and provide an excellent starting point for generative models. They’re relatively easy to generate but lack the complexity of real galaxies.
  
2. **COSMOS Galaxy Images**: The COSMOS dataset contains real observational data from the Hubble Space Telescope, providing complex structures and noise that challenge the models to replicate real astronomical features accurately.

3. **SKIRT Simulations**: These synthetic galaxy images from IllustrisTNG simulations are particularly valuable because they mimic the physical processes of galaxies, like dust radiation. SKIRT images added realism to the model training process, and since they are noiseless, they enabled precise control in testing the model’s ability to handle added noise.

---

## Key Metrics for Evaluating Image Quality and Robustness
The research used various metrics to quantify how well the generated images match real galaxy images:

1. **Wasserstein Distance**: This is a mathematical way to measure the similarity between distributions—in this case, between generated and real images. The lower the Wasserstein Distance, the closer the synthetic images are to real ones in terms of structural properties like asymmetry, concentration, and Gini coefficient.

2. **Fréchet Inception Distance (FID)**: FID is a standard computer vision metric for evaluating the perceptual similarity of generated images to real ones. Interestingly, even though it was developed for natural images, it correlated well with perceived similarity in galaxy images here, with StyleGAN consistently achieving the lowest FID scores.

3. **Power Spectrum Analysis**: The researchers also examined the power spectrum of the images to measure how well the generated data captured physical properties across different scales. This provided a nuanced view of how each model replicated the fine details in galaxy structure.

---

## Training the Models with Generated Data: Denoising as a Case Study
To evaluate the impact of generated data on robustness, the researchers trained a convolutional neural network (CNN) for an image denoising task using different combinations of real and synthetic images from the SKIRT dataset. Here’s how the process unfolded:

1. **Creating Mock Observations**: By adding noise and point spread function (PSF) distortions to the images, the researchers simulated realistic observational conditions. This allowed them to train the CNN to remove noise while learning the underlying data distribution.

2. **Mixing Ratios of Real and Generated Data**: The model was trained with various ratios of real to generated images to determine the best blend. A key insight here was that adding synthetic data from StyleGAN significantly improved the CNN’s robustness, enabling it to handle domain shifts and noise more effectively.

3. **Results and Performance Gains**: Training with generated images led to measurable improvements in the CNN’s ability to generalize across different noise levels and resolutions, with 45% better performance in handling noise and an 11% increase in resilience to resolution shifts. This finding underscores the value of using synthetic data to complement real data in making machine learning models more adaptable to new, unseen conditions.

---

## Limitations and Challenges
While the results are impressive, there are some limitations to consider:

1. **Computational Cost**: Training high-quality models like StyleGAN demands significant computational resources. For large-scale applications in astronomy, this could be a bottleneck.

2. **Generative Model Bias**: Even the best models can introduce biases. Since the generated images are based on the distribution of the training data, any bias in the original dataset can carry over into the synthetic data. In astronomy, this could lead to models that don’t generalize well to truly novel observations.

3. **Visual Realism vs. Physical Accuracy**: The paper notes that while StyleGAN produces realistic images, there’s still work to be done to ensure that generated images fully capture the underlying physical properties, especially at finer scales. Improvements in model architectures that explicitly consider Fourier or frequency-based features could help address this in the future.

---

## Conclusion and Reflections
This research shows that generative models, particularly StyleGAN, can produce galaxy images that are not only visually convincing but also statistically similar to real data. By incorporating synthetic images into training, we can make machine learning models more resilient to the uncertainties and domain shifts inherent in astronomical data. This approach could play a key role in preparing models for upcoming astronomical surveys like those from the Vera Rubin Observatory and the Square Kilometer Array.

In my view, the most promising aspect of this work is how it demonstrates the practical value of synthetic data in science. By addressing domain shifts and noise through generative modeling, we can develop more adaptable and robust models, potentially transforming how we handle the flood of new data in fields beyond astronomy. However, careful calibration of these generative models is essential, as the limitations noted show the importance of understanding and minimizing biases introduced through synthetic data. As generative models evolve, refining them to better capture the complex physics of galaxies will be a fascinating frontier for both machine learning and astronomy.
