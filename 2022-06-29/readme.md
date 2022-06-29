# Psychology API

The idea is to input a review(s) for a product, then get back measures for:

- [x] Polarity;
- [x] Intensity;
- [ ] Emotion;
- [ ] Aspects;
- [ ] Personality;
- [ ] Depression;
- [ ] Engagement;
- [ ] Well-being;

## Polarity Classification

Given review(s), a classification of `positive`, `negative`, or `neutral` is assigned to each review.

- http://sentic.net/senticnet-6.pdf
- http://sentic.net/sentic-patterns.pdf

## Intensity Ranking

Given review(s), a floating-point number between 0 (lowest intensity) and 1 (highest intensity) is output.

- http://sentic.net/predicting-intensities-of-emotions-and-sentiments.pdf

## Emotion Recognition with Sentiment Analysis

Given the review(s), the algorithm extracts emotion labels from text and outputs is a list of emotion labels.

- http://sentic.net/hourglass-model-revisited.pdf

## Aspect Extraction

Given review(s), the algorithm outputs a list of aspects with corresponding polarity values.

- http://sentic.net/sentic-gcn.pdf

## Personality Prediction

Given review(s), the algorithm outputs a measure for 5 personality traits (OCEAN).

- http://sentic.net/predicting-personality-with-psycholinguistic-and-language-model-features.pdf

## Depression Categorization

Given review(s), the algorithm outputs a depression score between zero (no depression detected) and 100 (high depression detected).

- http://sentic.net/suicidal-ideation-and-mental-disorder-detection.pdf

## Engagement Measure

Given review(s), the algorithm outputs an engagement score between -100 (high disengagement) and 100 (high engagement. This analysis can provide actionable insights into how users view and use a specific service or product.

- http://sentic.net/predicting-video-engagement.pdf

## Well-being assessment

Given review(s), the algorithm  outputs a well-being score between -100 (high stress) and 100 (high well-being).

- http://sentic.net/mentalbert.pdf
