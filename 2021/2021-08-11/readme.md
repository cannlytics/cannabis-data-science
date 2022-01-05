# Cannabis Data Science – Meetup 8/11/2021

Agenda

1. Welcome and introductions (10 minutes)
2. Presentation (15 minutes)
3. Data Exploration (15 minutes)
4. Questions and Answers (15 minutes)
5. Future work (5 minutes)

## Introduction

Anecdotally, some purchase managers at retail establishments take moisture content into consideration when purchasing cannabis. The moisture content of a product affects it's moisture corrected cannabinoid concentration. So, purchase managers can calculate moisture corrected concentrations to compare products with various moisture contents. The question is, does utilizing a moisture correction factor enable a purchase manager to better predict sales than if a purchaser tried to predict sales on cannabinoid concentration alone.

## Hypothesis

Null hypothesis: There is no statistically significant relationship between moisture corrected THC concentrations and total sales for a product.

Null hypothesis: The correlation between THC concentrations and total sales for a product is not significantly different than the correlation between moisture adjusted THC concentrations and totals sales of a given product.

## Methodology

The outline of the research methodology is as follows.

1. Calculate sales per lab result.
2. Calculate moisture corrected THC concentration for each sample.
3. Estimate a regressions of sales per lab result on the lab result's moisture corrected THC concentration.

## Resources

- [Converting Wet Corn Weight to Dry Corn Weight](https://www.agry.purdue.edu/ext/corn/news/timeless/WaterShrink.html)
- [Washington State Traceability Data 2018 - 2020](https://lcb.app.box.com/s/fnku9nr22dhx04f6o646xv6ad6fswfy9?page=1)
