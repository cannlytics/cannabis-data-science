# Consumer Recommendations

Parse a producer's corpus of CoAs, create a standardized datafile, then use the data, augmented with data about consumer's prior purchases to create product recommendations for each consumer.

## Introduction

This project aims to revolutionize the way consumers choose cannabis strains. By leveraging cutting-edge data science techniques and tools, we harness the power of consumer purchase history and chemical profiles of cannabis strains to recommend the most fitting products to individual consumers.

## Literature Review

Current cannabis recommendation systems are largely based on subjective reviews and broad classifications. However, with the advancements in data science, there's potential to make these recommendations more personal and precise. Our approach builds on foundational concepts in machine learning, customer analytics, and cannabis chemistry.

## Methodology

1. **Data Collection**: Harnessing powerful web scraping tools like Selenium and BeautifulSoup, we collect comprehensive datasets from Raw Garden's lab results. Further, to enrich our data, we utilize PDF scraping tools like PDFPlumber and OCR (Tesseract).

2. **Data Preprocessing**: Standardizing and cleaning the data is crucial. We employ cryptography (SHA256) for generating unique IDs and plan to use NLP for data standardization.

3. **Modeling**: Using the k-nearest neighbor model, we predict products with similar chemical profiles to a consumer's history. The algorithm calculates average product concentrations by consumer, finds k-nearest products, and iterates when a purchase is made. Further extensions include the use of moving averages and creating optimal blends.

## Data

- [Aggregated Cannabis Laboratory Test Results](https://huggingface.co/datasets/cannlytics/aggregated-cannabis-test-results)

## Results

The model's results provide a list of recommended cannabis products tailored to a user's historical preference. It's essential to understand that the recommendations are based on chemical profiles and previous purchases.

## Discussion

To use the model, input the user's historical purchase data, and run the algorithm. The result will be a list of products with chemical profiles similar to the user's preferred strains. This methodology allows for a personalized approach to cannabis recommendations, enhancing user satisfaction.

Scenario:

Imagine a user who historically prefers strains high in linalool. Our model will sift through countless product profiles and recommend products with a high concentration of linalool or similar terpenes.

Future work:

1. Incorporate sentiment analysis from user reviews to refine recommendations.
2. Explore other machine learning models for enhanced accuracy.
3. Integration with online cannabis marketplaces for real-time recommendations.

## References

*Data Sources*

- [Raw Garden Lab Results](https://rawgarden.farm/lab-results/)
- [Strain Reviews](https://cannlytics.page.link/reported-effects)
