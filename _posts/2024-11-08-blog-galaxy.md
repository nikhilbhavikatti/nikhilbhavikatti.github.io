---
title: "Exploring the Cosmos with AI: How Generative Models Create Realistic Galaxy Images and Improve Machine Learning Robustness"
date: 2024-11-12
permalink: /blog/exploring-the-cosmos-with-ai/
---

This post summarizes and explores insights from the paper *"Realistic galaxy images and improved robustness in machine learning tasks from generative modelling"* by Benjamin J. Holzschuh et al. As I delved into this research, I realized how impactful these methods can be - not only in creating visually realistic galaxy images but also in enhancing machine learning robustness. This paper examines how various generative models, particularly GANs and VAEs, are used to produce synthetic galaxy images that closely resemble real astronomical data. The result? A powerful tool for improving machine learning models used in astronomy, making them more resilient to domain shifts and noise. In this post, I'll break down the research, explain the models and methods, and discuss the implications and limitations of this approach.

---

## The Era of Big Data in Astronomy
Astronomy is undergoing a transformative phase, driven by cutting-edge surveys like Euclid, the Vera Rubin Observatory, and the Square Kilometer Array (SKA). These initiatives promise a deluge of high-quality data, enabling unprecedented studies of cosmic phenomena. For example, the Euclid survey is set to cover a 40-square-degree area of the sky, dwarfing the 2-square-degree COSMOS field while maintaining comparable angular resolution.

This wealth of data opens doors to revolutionary insights into galaxy evolution, dark matter distribution, and cosmic history. However, alongside the opportunities, the sheer scale and complexity of these datasets introduce significant computational and methodological challenges. Machine learning (ML) has emerged as a powerful tool to analyze astronomical data, excelling in tasks like star-galaxy classification, morphological categorization, and the detection of gravitational lens systems.

---

## Understanding the Problem: Limited Data and Domain Shifts in Astronomy
Despite its promise, ML in astronomy faces unique hurdles. The performance of any ML model hinges on the quality and diversity of its training data. However, in the astronomical domain, data is often limited, biased, or unevenly distributed. For instance, while low-redshift galaxies are well-studied and extensively observed, they differ fundamentally from high-redshift galaxies, which are often the focus of advanced cosmological studies.

This mismatch leads to domain shifts - scenarios where the training data distribution does not align with the test data distribution. Such shifts can degrade the performance of ML models, making them less reliable for real-world applications. Addressing these challenges requires innovative solutions, particularly in the form of methods that can mitigate the impact of limited data and enhance model robustness.

---

## What is Generative Modeling?
Generative modeling is a branch of machine learning that aims to create new data samples that resemble the data used to train the model. For instance, in the context of astronomy, generative models can produce synthetic galaxy images that mimic real astronomical observations. These models learn the underlying patterns and structure of the training data and use this knowledge to generate new samples that share the same characteristics.

The primary advantage of generative modeling is its ability to address challenges related to limited or biased datasets. By augmenting existing data, generative models can help machine learning systems perform better, especially when faced with tasks involving domain shifts or out-of-distribution samples.

---

### Variational Autoencoders (VAEs)
VAEs are one of the foundational approaches to generative modeling. They compress data into a low-dimensional latent space and then reconstruct it back into its original form. This compression-decompression process allows VAEs to learn meaningful representations of data.

**Strengths**
* Simplicity: VAEs are relatively easy to train compared to other generative models.
* Latent Space Structure: The structured latent space makes them useful for understanding and manipulating the data they represent.
* Stability: They are less prone to issues like mode collapse, which can affect other models like GANs.

**Weaknesses**
* Overly Smooth Outputs: VAEs often produce blurry images because the reconstruction process minimizes pixel-wise errors, leading to a loss of fine details.
* Lower Visual Quality: They struggle to generate images with complex textures or intricate details, which limits their use in tasks requiring high fidelity.

---

### Generative Adversarial Networks (GANs)
GANs are a more advanced and widely used generative modeling technique. They consist of two neural networks:

1. Generator: Creates synthetic data.
2. Discriminator: Evaluates whether the data is real or generated.

These networks compete with each other - the generator tries to fool the discriminator, while the discriminator works to distinguish real from fake data. This adversarial training pushes the generator to produce highly realistic samples.

**Strengths**
* High-Quality Outputs: GANs excel at generating visually realistic images, making them ideal for applications like galaxy image synthesis.
* Detail Preservation: They can capture complex textures and fine details better than VAEs.

**Weaknesses**
* Training Instability: GANs are notoriously difficult to train, often requiring careful tuning of hyperparameters and loss functions.
* Mode Collapse: GANs can sometimes focus on generating a narrow subset of the data distribution, ignoring other variations in the data.

---

### Adversarial Latent Autoencoders (ALAEs)
ALAEs combine elements of both VAEs and GANs. They use a structured latent space like VAEs but also incorporate adversarial training, similar to GANs. The generator is split into two parts:

1. A mapping network that learns the latent representation.
2. A decoder that generates data from the latent space.

Additionally, ALAEs include a reconstruction loss to ensure that the latent space accurately represents the data.

**Strengths**
* Combines Best of Both Worlds: ALAEs produce high-quality images like GANs while maintaining the structured latent space of VAEs.
* Improved Stability: They are more stable to train compared to traditional GANs.
* Meaningful Latent Representations: The added reconstruction loss ensures the latent space remains interpretable and robust.

**Weaknesses**
* Complexity: ALAEs are more complex to implement and train than VAEs.
* Intermediate Performance: While they strike a balance between VAEs and GANs, they may not always outperform specialized models for specific tasks.

---

## Datasets
The research utilizes three distinct datasets to evaluate the effectiveness of generative models in creating realistic galaxy images. Each dataset has unique characteristics that test the models' ability to generate realistic and diverse outputs while balancing data quality and availability.

---

### Sérsic Profiles
Sérsic profiles are synthetic models commonly used in astronomy to represent the surface brightness of galaxies. These profiles are defined mathematically, allowing unlimited generation of synthetic galaxy images by varying specific parameters like brightness, size, and shape.

**Characteristics**
* Size: 50,000 images, each 256x256 pixels.
* Attributes: Includes parameters such as the Sérsic index, axis ratio, and effective radius.
* Advantages: The dataset is entirely synthetic and free of noise, ensuring complete control over galaxy properties.

**Purpose:**
This dataset is ideal for testing the generative models' ability to reproduce simple and controlled galaxy features.

---

### COSMOS Field Observations
This dataset contains real observations of galaxies from the COSMOS field, captured using the Hubble Space Telescope (HST). It provides a more complex and realistic benchmark compared to the Sérsic profiles.

**Characteristics**
* Size: 20,114 images, cropped to 256x256 pixels.
* Filter: Captured using the F814W filter.
* Resolution: High angular resolution (0.03 arcseconds per pixel).
* Noise: Includes observational noise and point spread function (PSF) effects, which add realism.

**Purpose:**
The COSMOS dataset is used to evaluate the generative models' ability to replicate real astronomical images, including imperfections like noise and PSF distortions.

---

### SKIRT Synthetic Images
The SKIRT dataset is a collection of synthetic high-resolution galaxy images created using the IllustrisTNG simulation and processed with the SKIRT radiative transfer code. These images simulate the effects of dust and light interactions within galaxies, resulting in highly realistic outputs.

**Characteristics**
* Size: 9,564 images with variable dimensions.
* Filters: Includes multiple filters (g, r, i, z) for color information.
* Resolution: Physical pixel size of 0.276 kpc.
* Realism: High-resolution images mimic real galaxy observations while excluding noise and PSF.

**Purpose:**
The SKIRT dataset challenges generative models to handle complex features like realistic radiation and dust effects. It also allows testing on data with varying resolutions

---

## Metrics
To evaluate the performance of generative models, the research uses a combination of physically motivated metrics and computer vision-based metrics. These metrics assess how well the generated images match the original data in terms of both visual quality and physical properties.

---

### Physically Motivated Metrics
These metrics focus on properties relevant to astronomy, ensuring that the generated galaxy images align with real astronomical data.

### Morphological Properties
Morphological properties describe the shape, structure, and brightness distribution of galaxies. The following measurements are used:

* Asymmetry: Measures how symmetric a galaxy's structure is. A high value indicates irregular shapes or disturbances, common in merging galaxies.
* Smoothness: Indicates the clumpiness of the galaxy. Smoother galaxies tend to have less fine structure or noise.
* Concentration: Measures the brightness distribution, with higher values indicating light concentrated toward the center.
* Gini Coefficient: Reflects the distribution of light intensity; higher values mean most of the light is concentrated in a few bright pixels.
* M20 Statistic: Represents the brightness of the brightest 20% of a galaxy's pixels, relative to the total brightness.
* Half-Light Radius: The radius within which half of the galaxy's light is contained.
* Sérsic Index: Describes the steepness of the galaxy's light profile, useful for distinguishing between disk-like and elliptical galaxies.
* Orientation and Ellipticity: Measure the galaxy's alignment and the ratio of its minor to major axis, respectively.

These properties are compared using the Wasserstein distance, which quantifies differences between distributions of these measurements in real and generated datasets.

### Power Spectrum
The 2D power spectrum analyzes the distribution of surface brightness across different spatial scales.

**How it works:** By applying a 2D Fourier Transform, the galaxy image is decomposed into frequencies corresponding to different physical scales.

**Purpose:** This metric provides insight into a galaxy's structural details, such as the balance between fine textures and broader patterns.

The Wasserstein distance is used to compare power spectra between real and generated images.

### Colors and Bulge Statistics
**Colors:** For datasets with multi-band filters, such as SKIRT, the g-i color index measures the difference in brightness between two bands (g and i). This reveals information about a galaxy’s stellar populations and dust content.

* Red Galaxies (Early-Type): Tend to have higher g-i values due to older stars and less active star formation.
* Blue Galaxies (Late-Type): Have lower g-i values, indicating active star formation and younger stellar populations.

**Bulge Statistics (Gini-M20):** The Gini-M20 statistic combines two measures—Gini coefficient and M20—to evaluate the prominence of a galaxy’s central bulge:

* Gini Coefficient: Indicates how concentrated the brightness is.
* M20 Statistic: Represents the spatial distribution of the brightest regions.
* Purpose: Together, these statistics distinguish between galaxies with prominent bulges (early-type) and those without (late-type).

By comparing the g-i colors and bulge statistics, researchers assess how well the generated data captures the diversity of galaxy types, from elliptical to disk-like structures.

### Wasserstein Distance
The Wasserstein distance, also known as Earth Mover's Distance, measures the similarity between two distributions. It is used to compare distributions of:
* Morphological properties
* Power spectrum values
* Colors and bulge statistics
This metric is robust and captures differences beyond simple averages, making it particularly suitable for evaluating complex datasets like galaxies.

---

### Computer Vision-Based Metrics
These metrics evaluate the perceptual quality of generated images, as commonly used in computer vision.

### Fréchet Inception Distance (FID)
FID compares the distributions of feature representations from real and generated images using a pre-trained neural network (InceptionV3). Lower FID scores indicate higher similarity.

**Steps:**
1. Extract features from real and generated images.
2. Calculate the mean and covariance of these features for both datasets.
3. Compute the distance between these distributions.

### Kernel Inception Distance (KID)
KID is similar to FID but uses a kernel-based approach for better reliability with small datasets. Lower KID values indicate higher similarity.

**Advantages:**
* Does not assume the data follows a Gaussian distribution.
* Has an unbiased estimator.

---

## Results
This section presents the performance of the generative models—StyleGAN, ALAEs, and VAEs—across the datasets (Sérsic profiles, COSMOS, and SKIRT) and metrics. Quantitative results are supported by tables, followed by a detailed description.

### Morphological Properties
Performance Table: Wasserstein Distance (Lower is Better)

| **Property**        | **Sérsic Profiles**       |                         |         | **COSMOS Field**         |                         |         | **SKIRT Synthetic Images** |                         |         |
|----------------------|---------------------------|-------------------------|---------|--------------------------|-------------------------|---------|-----------------------------|-------------------------|---------|
|                      | **VAE**                  | **StyleGAN**            | **ALAE**| **VAE**                  | **StyleGAN**            | **ALAE**| **VAE**                     | **StyleGAN**            | **ALAE**|
| **Asymmetry**        | 97.87                    | 57.93                  | 48.94   | 54.78                   | 11.41                  | 33.17   | 95.12                      | 17.09                  | 15.09   |
| **Smoothness**       | 69.46                    | 4.83                   | 14.47   | 57.61                   | 5.06                   | 29.12   | 115.44                     | 4.85                   | 6.86    |
| **Concentration**    | 22.15                    | 5.09                   | 5.36    | 46.69                   | 8.31                   | 58.56   | 45.05                      | 3.92                   | 31.12   |
| **Gini Coefficient** | 21.63                    | 3.05                   | 6.22    | 48.60                   | 3.26                   | 50.63   | 51.15                      | 14.76                  | 36.30   |
| **Half-Light Radius**| 30.10                    | 8.08                   | 2.64    | 66.63                   | 2.75                   | 48.01   | 41.27                      | 5.87                   | 9.60    |
| **Average**          | 32.50                    | 11.89                  | 11.07   | 63.97                   | 6.72                   | 42.32   | 61.14                      | 9.04                   | 17.85   |

#### Observations
* StyleGAN consistently achieved the lowest Wasserstein distances, excelling in complex datasets like COSMOS and SKIRT.
* ALAEs performed well on simpler datasets (e.g., Sérsic profiles) but struggled with higher asymmetry and smoothness values in SKIRT.
* VAEs showed the weakest performance, especially on metrics like Gini coefficient and asymmetry, often producing overly smooth results.

### Power Spectrum Analysis
Performance Table: Average Wasserstein Distance for Power Spectrum (Lower is Better)

| **Dataset**          | **VAE** | **StyleGAN** | **ALAE** |
|-----------------------|---------|--------------|----------|
| **Sérsic Profiles**   | 13.94   | 2.50         | 3.95     |
| **COSMOS Field**      | 24.38   | 5.24         | 8.06     |
| **SKIRT Images**      | 27.61   | 4.83         | 14.00    |

#### Observations
* StyleGAN consistently outperformed other models, particularly in SKIRT and COSMOS datasets, capturing small-scale details and large-scale structures.
* ALAEs performed moderately, matching StyleGAN at larger scales but failing to replicate finer details.
* VAEs struggled across all datasets, producing overly smooth images that lacked high-frequency details.

### Colors and Bulge Statistics
Performance Table: Wasserstein Distance for Colors and Bulge Statistics (Lower is Better)

| **Metric**             | **VAE** | **StyleGAN** | **ALAE** |
|-------------------------|---------|--------------|----------|
| **(g-i) Early Types**   | 12.60   | 1.15         | 3.16     |
| **(g-i) Late Types**    | 12.23   | 1.39         | 2.15     |
| **Gini-M20 Statistic**  | 16.89   | 1.63         | 11.89    |

#### Observations
* StyleGAN closely replicated the bimodal distribution of colors, distinguishing between early-type (red) and late-type (blue) galaxies.
* ALAEs captured some variation in colors but underestimated the fraction of red galaxies and produced biased Gini-M20 distributions.
* VAEs failed to replicate the diversity in both colors and bulge statistics, particularly for galaxies with prominent bulges.

### Computer Vision Metrics
Performance Table: FID and KID (Lower is Better)

| **Metric** | **Dataset**         | **VAE** | **StyleGAN** | **ALAE** |
|------------|---------------------|---------|--------------|----------|
| **FID**    | Sérsic Profiles     | 1088    | 55           | 161      |
|            | COSMOS Field        | 18425   | 145          | 1465     |
|            | SKIRT Images        | 11737   | 776          | 921      |
| **KID**    | Sérsic Profiles     | 0.81    | 0.04         | 0.08     |
|            | COSMOS Field        | 22.28   | 0.08         | 1.19     |
|            | SKIRT Images        | 12.78   | 0.60         | 0.52     |

#### Observations
* StyleGAN achieved the lowest FID and KID scores across all datasets, confirming its ability to produce perceptually realistic and diverse images.
* ALAEs performed reasonably well but were outperformed by StyleGAN in most cases.
* VAEs had the highest scores, reflecting their inability to capture complex textures and structures effectively.

---

## Denoising as a Case Study

### The Problem: Robustness in Machine Learning

Astronomical images are often subjected to real-world distortions, which make it challenging for machine learning models to perform effectively when applied to these images. Two key factors that contribute to this issue are:

- **The Point Spread Function (PSF):** The PSF results from the limitations of telescope optics and causes blurring in the images. This effect can severely distort fine details and make it difficult for models to distinguish between important features.
  
- **Background noise:** Noise can arise from various sources, such as poor observational conditions or limitations in the instruments used to capture the images. This noise can range from random interference to more structured distortions, further complicating image analysis.

Machine learning models that are trained on clean, noise-free data tend to struggle when exposed to noisy or distorted images. This is especially true when there's a domain shift, meaning that the noise levels or image resolution in the test data differ from what the model was exposed to during training.

To improve robustness in these models, one approach suggested by the authors is to incorporate StyleGAN-generated data into the training process. By mixing synthetic data with real-world data, the model can learn to generalize better and handle a wider range of distortions.

---

### Experimental Setup

To test this approach, the authors conducted an experiment focused on image denoising, where the goal was to train a Convolutional Neural Network (CNN) to recover clean galaxy images from noisy, blurred inputs.

- ### Data Preparation:
  - **Original data:** Real, high-resolution, and noiseless galaxy images sourced from the SKIRT dataset.
  - **Generated data:** Synthetic galaxy images created using StyleGAN, trained on the same SKIRT dataset.
  - **Mock observations:** Both real and generated images were degraded by adding the following distortions:
    - A **Gaussian PSF** with a standard deviation of 2.0 pixels, which simulates the blurring effect of telescope optics.
    - **Gaussian noise** with σ = 4.0 e⁻/pixel, mimicking real-world observational distortions.

- ### Training Strategy:
  The authors used a **mixing factor (α)** to control the proportion of real and generated data in the training set:
  - **α = 0:** Only real data was used.
  - **α = 0.5:** A 50-50 mix of real and generated data.
  - **α = 1:** Only generated data was used.

The total dataset size was kept constant to ensure that any improvements in performance were due to data diversity, rather than having more training samples.

- ### Evaluation:
  The model's performance was evaluated based on the following metrics:
  - **Mean Squared Error (MSE):** This metric measured the difference between the denoised output of the model and the original clean image.
  - **Robustness to domain shifts:** The CNN was also tested on images with different resolutions and noise levels to assess its ability to generalize under varying conditions.

---

### Key Results

The authors found several important insights from their experiment:

- **Mixing real and generated data significantly improved robustness:**
  - There was a **45% improvement** in robustness to changes in noise levels.
  - An **11% improvement** in robustness to changes in pixel resolution.
  - The best results were achieved when **80% of the training data** was generated, indicating that generated data complements real data and enhances the model's ability to generalize to unseen conditions.

- **Generated data alone worked surprisingly well:**
  - A model trained solely on StyleGAN-generated data (α = 1.0) performed admirably, with only an **8% higher test error** compared to models trained on a mix of real and generated data. 
  - This demonstrates the high quality of StyleGAN-generated images, which closely mimic the real SKIRT images in terms of morphology and power spectrum properties.

- **Specific improvements were observed under out-of-distribution (OOD) conditions:**
  - In conditions with **higher noise levels**, models trained only on real data performed poorly, while those trained with mixed data maintained strong performance.
  - For images with **variable resolutions**, models trained solely on real data struggled, while mixed data models adapted much better to these changes, demonstrating superior robustness.

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
