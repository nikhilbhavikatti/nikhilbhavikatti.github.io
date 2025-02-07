---
title: "Exploring the Cosmos with AI: How Generative Models Create Realistic Galaxy Images and Improve Machine Learning Robustness"
date: 2025-02-07
permalink: /blog/exploring-the-cosmos-with-ai/
---

<div id="toc-container">
  <h2>Contents</h2>
  <ul id="toc-list"></ul>
</div>

This post summarizes and explores insights from the paper <a href="https://arxiv.org/abs/2203.11956" target="_blank" rel="noopener noreferrer">Realistic galaxy images and improved robustness in machine learning tasks from generative modelling</a> by Benjamin J. Holzschuh et al. As I delved into this research, I realized how impactful these methods can be - not only in creating visually realistic galaxy images but also in enhancing machine learning robustness. This paper examines how various generative models, particularly GANs and VAEs, are used to produce synthetic galaxy images that closely resemble real astronomical data. The result? A powerful tool for improving machine learning models used in astronomy, making them more resilient to domain shifts and noise. In this post, I'll break down the research, explain the models and methods, and discuss the implications and limitations of this approach.

## The Era of Big Data in Astronomy
Astronomy is undergoing a transformative phase, driven by cutting-edge surveys like Euclid, the Vera Rubin Observatory, and the Square Kilometer Array (SKA). These initiatives promise a deluge of high-quality data, enabling unprecedented studies of cosmic phenomena. For example, the Euclid survey is set to cover a 40-square-degree area of the sky, dwarfing the 2-square-degree COSMOS field while maintaining comparable angular resolution.

This wealth of data opens doors to revolutionary insights into galaxy evolution, dark matter distribution, and cosmic history. However, alongside the opportunities, the sheer scale and complexity of these datasets introduce significant computational and methodological challenges. Machine learning (ML) has emerged as a powerful tool to analyze astronomical data, excelling in tasks like star-galaxy classification, morphological categorization, and the detection of gravitational lens systems.

## Understanding the Problem: Limited Data and Domain Shifts in Astronomy
Despite its promise, ML in astronomy faces unique hurdles. The performance of any ML model hinges on the quality and diversity of its training data. However, in the astronomical domain, data is often limited, biased, or unevenly distributed. For instance, while low-redshift galaxies are well-studied and extensively observed, they differ fundamentally from high-redshift galaxies, which are often the focus of advanced cosmological studies.

This mismatch leads to domain shifts - scenarios where the training data distribution does not align with the test data distribution. Such shifts can degrade the performance of ML models, making them less reliable for real-world applications. Addressing these challenges requires innovative solutions, particularly in the form of methods that can mitigate the impact of limited data and enhance model robustness.

## What is Generative Modeling?
Generative modeling is a branch of machine learning that aims to create new data samples that resemble the data used to train the model. For instance, in the context of astronomy, generative models can produce synthetic galaxy images that mimic real astronomical observations. These models learn the underlying patterns and structure of the training data and use this knowledge to generate new samples that share the same characteristics.

The primary advantage of generative modeling is its ability to address challenges related to limited or biased datasets. By augmenting existing data, generative models can help machine learning systems perform better, especially when faced with tasks involving domain shifts or out-of-distribution samples.

### Variational Autoencoders (VAEs)
Variational Autoencoders (VAEs) are a foundational approach to generative modeling, known for their ability to compress data into a low-dimensional latent space and then reconstruct it back into its original form. This compression-decompression process enables VAEs to learn meaningful representations of data, making them relatively simple to train compared to other generative models. Their structured latent space is particularly useful for understanding and manipulating data, and they are less prone to issues like mode collapse, which often plague models like GANs. However, VAEs are not without their limitations. They tend to produce overly smooth or blurry outputs, as the reconstruction process minimizes pixel-wise errors, resulting in a loss of fine details. This limitation makes them less effective at generating images with complex textures or intricate details, ultimately restricting their use in tasks that demand high visual fidelity. Despite these weaknesses, VAEs remain a valuable tool in the generative modeling landscape due to their stability and interpretability.

### Generative Adversarial Networks (GANs)
Generative Adversarial Networks (GANs) represent a more advanced and widely adopted approach to generative modeling, leveraging a unique framework of two competing neural networks: the generator, which creates synthetic data, and the discriminator, which evaluates whether the data is real or generated. This adversarial dynamic drives the generator to produce increasingly realistic samples as it tries to fool the discriminator, while the discriminator refines its ability to distinguish real from fake data. GANs are particularly celebrated for their ability to generate high-quality, visually realistic outputs, making them ideal for tasks like galaxy image synthesis. They excel at preserving intricate details and complex textures, outperforming models like VAEs in this regard. However, GANs come with significant challenges, including training instability, which often necessitates meticulous tuning of hyperparameters and loss functions. Additionally, they are prone to mode collapse, where the generator produces a limited subset of the data distribution, neglecting other variations. Despite these weaknesses, GANs remain a powerful tool for generating highly realistic and detailed data.

### Adversarial Latent Autoencoders (ALAEs)
Adversarial Latent Autoencoders (ALAEs) merge the strengths of both VAEs and GANs, creating a hybrid model that leverages a structured latent space, akin to VAEs, while incorporating adversarial training, similar to GANs. The generator in ALAEs is divided into two components: a mapping network that learns meaningful latent representations and a decoder that generates data from this latent space. Additionally, ALAEs include a reconstruction loss to ensure the latent space accurately reflects the input data. This combination allows ALAEs to produce high-quality images, rivaling GANs, while maintaining the interpretable and structured latent space characteristic of VAEs. They also offer improved training stability compared to traditional GANs, making them less prone to issues like mode collapse. However, ALAEs are more complex to implement and train than VAEs, and while they strike a balance between the two approaches, they may not always surpass specialized models tailored for specific tasks. Despite these trade-offs, ALAEs represent a compelling middle ground, blending the best aspects of VAEs and GANs into a single framework

## Datasets
The research utilizes three distinct datasets to evaluate the effectiveness of generative models in creating realistic galaxy images. Each dataset has unique characteristics that test the models' ability to generate realistic and diverse outputs while balancing data quality and availability.

### Sérsic Profiles
Sérsic profiles are a widely used synthetic modeling tool in astronomy, designed to represent the surface brightness of galaxies. Defined by a mathematical formula, these profiles enable the generation of an unlimited number of synthetic galaxy images by adjusting parameters such as brightness, size, and shape. A typical Sérsic dataset might consist of 50,000 images, each with a resolution of 256x256 pixels, and include attributes like the Sérsic index, axis ratio, and effective radius. One of the key advantages of this dataset is its synthetic nature, which ensures it is free from noise and provides complete control over galaxy properties. This makes Sérsic profiles an ideal resource for testing the capabilities of generative models, as they allow researchers to evaluate how well these models can reproduce simple and controlled galaxy features in a noise-free environment.

### COSMOS Field Observations
The COSMOS dataset offers a more complex and realistic benchmark for generative modeling compared to synthetic Sérsic profiles, as it consists of real observations of galaxies from the COSMOS field, captured using the Hubble Space Telescope (HST). This dataset includes 20,114 images, each cropped to 256x256 pixels and captured using the F814W filter. With a high angular resolution of 0.03 arcseconds per pixel, the COSMOS dataset provides a detailed and authentic representation of galaxy images. Unlike synthetic datasets, it incorporates observational noise and point spread function (PSF) effects, adding a layer of realism that mirrors the challenges of real-world astronomical data. The purpose of this dataset is to evaluate how well generative models can replicate real astronomical images, including their imperfections such as noise and PSF distortions, making it a valuable tool for testing the robustness and fidelity of generative modeling techniques in astronomy.

### SKIRT Synthetic Images
The SKIRT dataset is a collection of synthetic high-resolution galaxy images generated using the IllustrisTNG simulation and processed with the SKIRT radiative transfer code, which simulates the effects of dust and light interactions within galaxies. This results in highly realistic outputs that closely mimic real galaxy observations. The dataset comprises 9,564 images with variable dimensions, captured across multiple filters (g, r, i, z) to provide detailed color information. With a physical pixel size of 0.276 kpc, the SKIRT dataset offers high-resolution images that exclude observational noise and point spread function (PSF) effects, focusing instead on the intricate details of radiation and dust interactions. The purpose of this dataset is to challenge generative models to handle complex features like realistic radiation and dust effects, while also testing their ability to adapt to data with varying resolutions. This makes the SKIRT dataset a valuable benchmark for evaluating the robustness and versatility of generative models in capturing sophisticated astrophysical phenomena.

<figure>
  <img src="{{ site.baseurl }}/assets/images/01_skirt_or.jpg" alt="Original SKIRT images" width="800" height="100">
  <img src="{{ site.baseurl }}/assets/images/02_skirt_gen.jpg" alt="StyleGAN generated SKIRT images" width="800" height="100">
  <figcaption style="text-align: center;">Original (top) and StyleGAN generated (bottom) SKIRT synthetic images</figcaption>
</figure>
<figure>
  <img src="{{ site.baseurl }}/assets/images/03_cosmos_or.jpg" alt="Original COSMOS field observations images" width="800" height="100">
  <img src="{{ site.baseurl }}/assets/images/04_cosmos_gen.jpg" alt="StyleGAN generated COSMOS field observations images" width="800" height="100">
  <figcaption style="text-align: center;">Original (top) and StyleGAN generated (bottom) COSMOS field observations images</figcaption>
</figure>
<figure>
  <img src="{{ site.baseurl }}/assets/images/05_sersic_or.jpg" alt="Original Sersic profile images" width="800" height="100">
  <img src="{{ site.baseurl }}/assets/images/06_sersic_gen.jpg" alt="StyleGAN generated Sersic profile images" width="800" height="100">
  <figcaption style="text-align: center;">Original (top) and StyleGAN generated (bottom) Sersic profile images</figcaption>
</figure>

## Metrics
To evaluate the performance of generative models, the research uses a combination of physically motivated metrics and computer vision-based metrics. These metrics assess how well the generated images match the original data in terms of both visual quality and physical properties.

### Physically Motivated Metrics
These metrics focus on properties relevant to astronomy, ensuring that the generated galaxy images align with real astronomical data.

#### Morphological Properties
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

#### Power Spectrum
The 2D power spectrum is a powerful tool for analyzing the distribution of surface brightness across different spatial scales in galaxy images. It works by applying a 2D Fourier Transform to decompose the image into frequencies that correspond to various physical scales, revealing the balance between fine textures and broader patterns. This metric provides valuable insights into a galaxy's structural details, making it particularly useful for evaluating the quality of generated images. To quantify the similarity between real and generated images, the Wasserstein distance is used to compare their power spectra. This approach ensures that the generative models not only capture the overall appearance of galaxies but also accurately reproduce their underlying structural characteristics across different scales.

#### Colors and Bulge Statistics
- Colors: For datasets with multi-band filters, such as SKIRT, the g-i color index measures the difference in brightness between the g and i bands, providing insights into a galaxy’s stellar populations and dust content. Red galaxies, typically early-type, exhibit higher g-i values due to older stars and less active star formation, while blue galaxies, often late-type, have lower g-i values, indicating younger stellar populations and active star formation.

- Bulge Statistics (Gini-M20): The Gini-M20 statistic combines two measures—the Gini coefficient and the M20 statistic—to evaluate the prominence of a galaxy’s central bulge. The Gini coefficient quantifies how concentrated the brightness is, while the M20 statistic describes the spatial distribution of the brightest regions. Together, these metrics help distinguish between early-type galaxies with prominent bulges and late-type galaxies without them.

By comparing the g-i colors and bulge statistics of real and generated images, researchers can assess how well generative models capture the diversity of galaxy types, from elliptical galaxies with dominant bulges to disk-like galaxies with more diffuse structures. This analysis ensures that the generated data not only looks realistic but also accurately represents the physical and structural properties of galaxies.


#### Wasserstein Distance
The Wasserstein distance, or Earth Mover's Distance, measures the similarity between two distributions, capturing nuanced differences beyond simple averages. It compares distributions of morphological properties, power spectrum values, and color and bulge statistics, making it ideal for evaluating complex datasets like galaxies. This ensures generative models produce realistic outputs that accurately replicate the statistical and physical properties of real galaxies.

### Computer Vision-Based Metrics
These metrics evaluate the perceptual quality of generated images, as commonly used in computer vision.

#### Fréchet Inception Distance (FID)
The Fréchet Inception Distance (FID) is a widely used metric for evaluating the quality of generative models by comparing the distributions of feature representations from real and generated images. These features are extracted using a pre-trained neural network, such as InceptionV3. The process involves three main steps: first, extracting features from both real and generated images; second, calculating the mean and covariance of these features for each dataset; and third, computing the distance between the two distributions. Lower FID scores indicate higher similarity between the datasets, reflecting better performance of the generative model. By focusing on the statistical properties of the feature representations, FID provides a robust measure of how well the generated images match the real data, making it a valuable tool for assessing the fidelity of generative models.

#### Kernel Inception Distance (KID)
KID is a metric similar to FID but employs a kernel-based approach, making it more reliable for small datasets. Like FID, it compares feature representations of real and generated images extracted using a pre-trained network (e.g., InceptionV3), with lower KID values indicating higher similarity. Unlike FID, KID does not assume the data follows a Gaussian distribution and uses an unbiased estimator, which enhances its robustness and accuracy, particularly when working with limited data. These advantages make KID a valuable alternative for evaluating generative models, especially in scenarios where dataset size or distribution assumptions might limit the effectiveness of other metrics.

## Results
This section presents the performance of the generative models—StyleGAN, ALAEs, and VAEs—across the datasets (Sérsic profiles, COSMOS, and SKIRT) and metrics. Quantitative results are supported by tables, followed by a detailed description.

### Morphological Properties
Performance Table: Wasserstein Distance (Lower is Better)

<table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
  <tr style="background-color: #f4f4f9; text-align: center; font-weight: bold;">
    <td style="border: 1px solid #ddd; background-color: #f4f4f9; text-align: center;"><strong>Property</strong></td>
    <td colspan="3" style="border: 1px solid #ddd; background-color: #f4f4f9; text-align: center;"><strong>Sérsic Profiles</strong></td>
    <td colspan="3" style="border: 1px solid #ddd; background-color: #f4f4f9; text-align: center;"><strong>COSMOS Field</strong></td>
    <td colspan="3" style="border: 1px solid #ddd; background-color: #f4f4f9; text-align: center;"><strong>SKIRT Synthetic Images</strong></td>
  </tr>
  <tr style="background-color: #fafafa;">
    <td style="border: 1px solid #ddd; background-color: #fafafa;"></td>
    <td style="border: 1px solid #ddd; text-align: center;"><strong>VAE</strong></td>
    <td style="border: 1px solid #ddd; text-align: center;"><strong>StyleGAN</strong></td>
    <td style="border: 1px solid #ddd; text-align: center;"><strong>ALAE</strong></td>
    <td style="border: 1px solid #ddd; text-align: center;"><strong>VAE</strong></td>
    <td style="border: 1px solid #ddd; text-align: center;"><strong>StyleGAN</strong></td>
    <td style="border: 1px solid #ddd; text-align: center;"><strong>ALAE</strong></td>
    <td style="border: 1px solid #ddd; text-align: center;"><strong>VAE</strong></td>
    <td style="border: 1px solid #ddd; text-align: center;"><strong>StyleGAN</strong></td>
    <td style="border: 1px solid #ddd; text-align: center;"><strong>ALAE</strong></td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; background-color: #fafafa; font-weight: bold; text-align: left;">Asymmetry</td>
    <td style="border: 1px solid #ddd; text-align: center;">97.87</td>
    <td style="border: 1px solid #ddd; text-align: center;">57.93</td>
    <td style="border: 1px solid #ddd; text-align: center;">48.94</td>
    <td style="border: 1px solid #ddd; text-align: center;">54.78</td>
    <td style="border: 1px solid #ddd; text-align: center;">11.41</td>
    <td style="border: 1px solid #ddd; text-align: center;">33.17</td>
    <td style="border: 1px solid #ddd; text-align: center;">95.12</td>
    <td style="border: 1px solid #ddd; text-align: center;">17.09</td>
    <td style="border: 1px solid #ddd; text-align: center;">15.09</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; background-color: #fafafa; font-weight: bold; text-align: left;">Smoothness</td>
    <td style="border: 1px solid #ddd; text-align: center;">69.46</td>
    <td style="border: 1px solid #ddd; text-align: center;">4.83</td>
    <td style="border: 1px solid #ddd; text-align: center;">14.47</td>
    <td style="border: 1px solid #ddd; text-align: center;">57.61</td>
    <td style="border: 1px solid #ddd; text-align: center;">5.06</td>
    <td style="border: 1px solid #ddd; text-align: center;">29.12</td>
    <td style="border: 1px solid #ddd; text-align: center;">115.44</td>
    <td style="border: 1px solid #ddd; text-align: center;">4.85</td>
    <td style="border: 1px solid #ddd; text-align: center;">6.86</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; background-color: #fafafa; font-weight: bold; text-align: left;">Concentration</td>
    <td style="border: 1px solid #ddd; text-align: center;">22.15</td>
    <td style="border: 1px solid #ddd; text-align: center;">5.09</td>
    <td style="border: 1px solid #ddd; text-align: center;">5.36</td>
    <td style="border: 1px solid #ddd; text-align: center;">46.69</td>
    <td style="border: 1px solid #ddd; text-align: center;">8.31</td>
    <td style="border: 1px solid #ddd; text-align: center;">58.56</td>
    <td style="border: 1px solid #ddd; text-align: center;">45.05</td>
    <td style="border: 1px solid #ddd; text-align: center;">3.92</td>
    <td style="border: 1px solid #ddd; text-align: center;">31.12</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; background-color: #fafafa; font-weight: bold; text-align: left;">Gini Coefficient</td>
    <td style="border: 1px solid #ddd; text-align: center;">21.63</td>
    <td style="border: 1px solid #ddd; text-align: center;">3.05</td>
    <td style="border: 1px solid #ddd; text-align: center;">6.22</td>
    <td style="border: 1px solid #ddd; text-align: center;">48.60</td>
    <td style="border: 1px solid #ddd; text-align: center;">3.26</td>
    <td style="border: 1px solid #ddd; text-align: center;">50.63</td>
    <td style="border: 1px solid #ddd; text-align: center;">51.15</td>
    <td style="border: 1px solid #ddd; text-align: center;">14.76</td>
    <td style="border: 1px solid #ddd; text-align: center;">36.30</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; background-color: #fafafa; font-weight: bold; text-align: left;">Half-Light Radius</td>
    <td style="border: 1px solid #ddd; text-align: center;">30.10</td>
    <td style="border: 1px solid #ddd; text-align: center;">8.08</td>
    <td style="border: 1px solid #ddd; text-align: center;">2.64</td>
    <td style="border: 1px solid #ddd; text-align: center;">66.63</td>
    <td style="border: 1px solid #ddd; text-align: center;">2.75</td>
    <td style="border: 1px solid #ddd; text-align: center;">48.01</td>
    <td style="border: 1px solid #ddd; text-align: center;">41.27</td>
    <td style="border: 1px solid #ddd; text-align: center;">5.87</td>
    <td style="border: 1px solid #ddd; text-align: center;">9.60</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; background-color: #fafafa; font-weight: bold; text-align: left;">Average</td>
    <td style="border: 1px solid #ddd; text-align: center;">32.50</td>
    <td style="border: 1px solid #ddd; text-align: center;">11.89</td>
    <td style="border: 1px solid #ddd; text-align: center;">11.07</td>
    <td style="border: 1px solid #ddd; text-align: center;">63.97</td>
    <td style="border: 1px solid #ddd; text-align: center;">6.72</td>
    <td style="border: 1px solid #ddd; text-align: center;">42.32</td>
    <td style="border: 1px solid #ddd; text-align: center;">61.14</td>
    <td style="border: 1px solid #ddd; text-align: center;">9.04</td>
    <td style="border: 1px solid #ddd; text-align: center;">17.85</td>
  </tr>
</table>

<figure>
  <img src="{{ site.baseurl }}/assets/images/07_histograms_morph.png" alt="">
  <figcaption>
  Histograms showing selected optical morphological measurements for the SKIRT dataset and the generated datasets
  </figcaption>
</figure>

#### Observations
* StyleGAN consistently achieved the lowest Wasserstein distances, excelling in complex datasets like COSMOS and SKIRT.
* ALAEs performed well on simpler datasets (e.g., Sérsic profiles) but struggled with higher asymmetry and smoothness values in SKIRT.
* VAEs showed the weakest performance, especially on metrics like Gini coefficient and asymmetry, often producing overly smooth results.

### Power Spectrum Analysis
Performance Table: Average Wasserstein Distance for Power Spectrum (Lower is Better)

<table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
  <tr style="background-color: #f4f4f9; text-align: center; font-weight: bold;">
    <td style="border: 1px solid #ddd; background-color: #f4f4f9; text-align: center; width: 25%;"><strong>Dataset</strong></td>
    <td style="border: 1px solid #ddd; text-align: center; width: 25%;"><strong>VAE</strong></td>
    <td style="border: 1px solid #ddd; text-align: center; width: 25%;"><strong>StyleGAN</strong></td>
    <td style="border: 1px solid #ddd; text-align: center; width: 25%;"><strong>ALAE</strong></td>
  </tr>
  <tr style="background-color: #fafafa;">
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left; font-weight: bold;">Sérsic Profiles</td>
    <td style="border: 1px solid #ddd; text-align: center;">13.94</td>
    <td style="border: 1px solid #ddd; text-align: center;">2.50</td>
    <td style="border: 1px solid #ddd; text-align: center;">3.95</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left; font-weight: bold;">COSMOS Field</td>
    <td style="border: 1px solid #ddd; text-align: center;">24.38</td>
    <td style="border: 1px solid #ddd; text-align: center;">5.24</td>
    <td style="border: 1px solid #ddd; text-align: center;">8.06</td>
  </tr>
  <tr style="background-color: #fafafa;">
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left; font-weight: bold;">SKIRT Images</td>
    <td style="border: 1px solid #ddd; text-align: center;">27.61</td>
    <td style="border: 1px solid #ddd; text-align: center;">4.83</td>
    <td style="border: 1px solid #ddd; text-align: center;">14.00</td>
  </tr>
</table>

<figure>
  <img src="{{ site.baseurl }}/assets/images/08_contour_2dpower_spectrum.png" alt="">
  <figcaption>
  Contour plots of the average shifted 2D power spectrum of the \( r \)-band of the raw network outputs (VAE (i), ALAE (ii), and StyleGAN  (iii)) and the resized \( 256 \times 256 \) images of the SKIRT dataset (iv)
  </figcaption>
</figure>

#### Observations
* StyleGAN consistently outperformed other models, particularly in SKIRT and COSMOS datasets, capturing small-scale details and large-scale structures.
* ALAEs performed moderately, matching StyleGAN at larger scales but failing to replicate finer details.
* VAEs struggled across all datasets, producing overly smooth images that lacked high-frequency details.

### Colors and Bulge Statistics
Performance Table: Wasserstein Distance for Colors and Bulge Statistics (Lower is Better)

<table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
  <tr style="background-color: #f4f4f9; text-align: center; font-weight: bold;">
    <td style="border: 1px solid #ddd; background-color: #f4f4f9; text-align: center; width: 25%;"><strong>Metric</strong></td>
    <td style="border: 1px solid #ddd; text-align: center; width: 25%;"><strong>VAE</strong></td>
    <td style="border: 1px solid #ddd; text-align: center; width: 25%;"><strong>StyleGAN</strong></td>
    <td style="border: 1px solid #ddd; text-align: center; width: 25%;"><strong>ALAE</strong></td>
  </tr>
  <tr style="background-color: #fafafa;">
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left; font-weight: bold;">(g-i) Early Types</td>
    <td style="border: 1px solid #ddd; text-align: center;">12.60</td>
    <td style="border: 1px solid #ddd; text-align: center;">1.15</td>
    <td style="border: 1px solid #ddd; text-align: center;">3.16</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left; font-weight: bold;">(g-i) Late Types</td>
    <td style="border: 1px solid #ddd; text-align: center;">12.23</td>
    <td style="border: 1px solid #ddd; text-align: center;">1.39</td>
    <td style="border: 1px solid #ddd; text-align: center;">2.15</td>
  </tr>
  <tr style="background-color: #fafafa;">
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left; font-weight: bold;">Gini-M20 Statistic</td>
    <td style="border: 1px solid #ddd; text-align: center;">16.89</td>
    <td style="border: 1px solid #ddd; text-align: center;">1.63</td>
    <td style="border: 1px solid #ddd; text-align: center;">11.89</td>
  </tr>
</table>

<figure>
  <img src="{{ site.baseurl }}/assets/images/09_dist_bulge_colour.png" alt="">
  <figcaption>
  The Gini-M20 bulge statistic \( F(G, M_{20}) \) vs \( (g - i)_{\text{SDSS}} \) color of the generated data and the SKIRT galaxies. The bottom panels show contour plots of the galaxy distribution, while the histograms in the top panels show the marginal distribution of late- and early-type galaxies. A galaxy is classified as early-type if \( F(G, M_{20}) \geq 0 \) and late-type otherwise.
  </figcaption>
</figure>

#### Observations
* StyleGAN closely replicated the bimodal distribution of colors, distinguishing between early-type (red) and late-type (blue) galaxies.
* ALAEs captured some variation in colors but underestimated the fraction of red galaxies and produced biased Gini-M20 distributions.
* VAEs failed to replicate the diversity in both colors and bulge statistics, particularly for galaxies with prominent bulges.

### Computer Vision Metrics
Performance Table: FID and KID (Lower is Better)

<table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
  <tr style="background-color: #f4f4f9; text-align: center; font-weight: bold;">
    <td style="border: 1px solid #ddd; background-color: #f4f4f9; text-align: center;"><strong>Metric</strong></td>
    <td style="border: 1px solid #ddd; text-align: center; width: 25%;"><strong>Dataset</strong></td>
    <td style="border: 1px solid #ddd; text-align: center; width: 25%;"><strong>VAE</strong></td>
    <td style="border: 1px solid #ddd; text-align: center; width: 25%;"><strong>StyleGAN</strong></td>
    <td style="border: 1px solid #ddd; text-align: center; width: 25%;"><strong>ALAE</strong></td>
  </tr>
  <tr style="background-color: #fafafa;">
    <td rowspan="3" style="border: 1px solid #ddd; background-color: #fafafa; text-align: left; font-weight: bold;">FID</td>
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left;">Sérsic Profiles</td>
    <td style="border: 1px solid #ddd; text-align: center;">1088</td>
    <td style="border: 1px solid #ddd; text-align: center;">55</td>
    <td style="border: 1px solid #ddd; text-align: center;">161</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left;">COSMOS Field</td>
    <td style="border: 1px solid #ddd; text-align: center;">18425</td>
    <td style="border: 1px solid #ddd; text-align: center;">145</td>
    <td style="border: 1px solid #ddd; text-align: center;">1465</td>
  </tr>
  <tr style="background-color: #fafafa;">
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left;">SKIRT Images</td>
    <td style="border: 1px solid #ddd; text-align: center;">11737</td>
    <td style="border: 1px solid #ddd; text-align: center;">776</td>
    <td style="border: 1px solid #ddd; text-align: center;">921</td>
  </tr>
  <tr style="background-color: #fafafa;">
    <td rowspan="3" style="border: 1px solid #ddd; background-color: #fafafa; text-align: left; font-weight: bold;">KID</td>
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left;">Sérsic Profiles</td>
    <td style="border: 1px solid #ddd; text-align: center;">0.81</td>
    <td style="border: 1px solid #ddd; text-align: center;">0.04</td>
    <td style="border: 1px solid #ddd; text-align: center;">0.08</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left;">COSMOS Field</td>
    <td style="border: 1px solid #ddd; text-align: center;">22.28</td>
    <td style="border: 1px solid #ddd; text-align: center;">0.08</td>
    <td style="border: 1px solid #ddd; text-align: center;">1.19</td>
  </tr>
  <tr style="background-color: #fafafa;">
    <td style="border: 1px solid #ddd; background-color: #fafafa; text-align: left;">SKIRT Images</td>
    <td style="border: 1px solid #ddd; text-align: center;">12.78</td>
    <td style="border: 1px solid #ddd; text-align: center;">0.60</td>
    <td style="border: 1px solid #ddd; text-align: center;">0.52</td>
  </tr>
</table>

#### Observations
* StyleGAN achieved the lowest FID and KID scores across all datasets, confirming its ability to produce perceptually realistic and diverse images.
* ALAEs performed reasonably well but were outperformed by StyleGAN in most cases.
* VAEs had the highest scores, reflecting their inability to capture complex textures and structures effectively.

## Denoising as a Case Study

### The Problem: Robustness in Machine Learning

Astronomical images are often subjected to real-world distortions, which make it challenging for machine learning models to perform effectively when applied to these images. Two key factors that contribute to this issue are:

- The Point Spread Function (PSF): The PSF results from the limitations of telescope optics and causes blurring in the images. This effect can severely distort fine details and make it difficult for models to distinguish between important features.
  
- Background noise: Noise can arise from various sources, such as poor observational conditions or limitations in the instruments used to capture the images. This noise can range from random interference to more structured distortions, further complicating image analysis.

Machine learning models that are trained on clean, noise-free data tend to struggle when exposed to noisy or distorted images. This is especially true when there's a domain shift, meaning that the noise levels or image resolution in the test data differ from what the model was exposed to during training.

To improve robustness in these models, one approach suggested by the authors is to incorporate StyleGAN-generated data into the training process. By mixing synthetic data with real-world data, the model can learn to generalize better and handle a wider range of distortions.

### Experimental Setup

<figure>
  <img src="{{ site.baseurl }}/assets/images/10_experiment_setup.png" alt="">
  <figcaption>
  Experiment setup for evaluating the influence of mixing source and generated data on the denoising model. The real data is split into a testing set, validation set, and generated set. The generative models are trained on the training and validation sets. When training the denoising models, we randomly draw a sample from the training set with probability \(1 - \alpha\) and from the generated set with probability \(\alpha\). The best denoising model is selected based on its performance on the validation set. The model is then evaluated on the testing set with multiple augmentations to simulate a domain shift.
</figcaption>


</figure>

To test this approach, the authors conducted an experiment focused on image denoising, where the goal was to train a Convolutional Neural Network (CNN) to recover clean galaxy images from noisy, blurred inputs.

#### Data Preparation:
- Original data: Real, high-resolution, and noiseless galaxy images sourced from the SKIRT dataset.
- Generated data: Synthetic galaxy images created using StyleGAN, trained on the same SKIRT dataset.
- Mock observations: Both real and generated images were degraded by adding the following distortions:
  - A Gaussian PSF with a standard deviation of 2.0 pixels, which simulates the blurring effect of telescope optics.
  - Gaussian noise with σ = 4.0 e⁻/pixel, mimicking real-world observational distortions.

#### Training Strategy:
The authors used a mixing factor (α) to control the proportion of real and generated data in the training set:
- α = 0: Only real data was used.
- α = 0.5: A 50-50 mix of real and generated data.
- α = 1: Only generated data was used.

The total dataset size was kept constant to ensure that any improvements in performance were due to data diversity, rather than having more training samples.

#### Evaluation:

<figure>
  <img src="{{ site.baseurl }}/assets/images/11_experiment_result.png" alt="">
  <figcaption>
  Mean squared error of the final models trained with different mixing factors \(\alpha\) on the testing set and on the testing set with various augmentations.
  </figcaption>
</figure>

The model's performance was evaluated based on the following metrics:
- Mean Squared Error (MSE): This metric measured the difference between the denoised output of the model and the original clean image.
- Robustness to domain shifts: The CNN was also tested on images with different resolutions and noise levels to assess its ability to generalize under varying conditions.

### Key Results

The authors found several important insights from their experiment:

- Mixing real and generated data significantly improved robustness:
  - There was a 45% improvement in robustness to changes in noise levels.
  - An 11% improvement in robustness to changes in pixel resolution.
  - The best results were achieved when 80% of the training data was generated, indicating that generated data complements real data and enhances the model's ability to generalize to unseen conditions.

- Generated data alone worked surprisingly well:
  - A model trained solely on StyleGAN-generated data (α = 1.0) performed admirably, with only an 8% higher test error compared to models trained on a mix of real and generated data. 
  - This demonstrates the high quality of StyleGAN-generated images, which closely mimic the real SKIRT images in terms of morphology and power spectrum properties.

- Specific improvements were observed under out-of-distribution (OOD) conditions:
  - In conditions with higher noise levels, models trained only on real data performed poorly, while those trained with mixed data maintained strong performance.
  - For images with variable resolutions, models trained solely on real data struggled, while mixed data models adapted much better to these changes, demonstrating superior robustness.

## Limitations and Challenges
While the results are impressive, there are some limitations to consider:

1. Computational Cost: Training high-quality models like StyleGAN demands significant computational resources. For large-scale applications in astronomy, this could be a bottleneck.

2. Generative Model Bias: Even the best models can introduce biases. Since the generated images are based on the distribution of the training data, any bias in the original dataset can carry over into the synthetic data. In astronomy, this could lead to models that don’t generalize well to truly novel observations.

3. Visual Realism vs. Physical Accuracy: The paper notes that while StyleGAN produces realistic images, there’s still work to be done to ensure that generated images fully capture the underlying physical properties, especially at finer scales. Improvements in model architectures that explicitly consider Fourier or frequency-based features could help address this in the future.

## Conclusion and Reflections
This research shows that generative models, particularly StyleGAN, can produce galaxy images that are not only visually convincing but also statistically similar to real data. By incorporating synthetic images into training, we can make machine learning models more resilient to the uncertainties and domain shifts inherent in astronomical data. This approach could play a key role in preparing models for upcoming astronomical surveys like those from the Vera Rubin Observatory and the Square Kilometer Array.

In my view, the most promising aspect of this work is how it demonstrates the practical value of synthetic data in science. By addressing domain shifts and noise through generative modeling, we can develop more adaptable and robust models, potentially transforming how we handle the flood of new data in fields beyond astronomy. However, careful calibration of these generative models is essential, as the limitations noted show the importance of understanding and minimizing biases introduced through synthetic data. As generative models evolve, refining them to better capture the complex physics of galaxies will be a fascinating frontier for both machine learning and astronomy.
