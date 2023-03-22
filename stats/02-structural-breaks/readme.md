# Structural Breaks

In an exclusive session, we explore data from the Massachusetts Cannabis Control Commission and the Federal Reserve Economic Data (FRED) to test if there have been structural breaks in consumer preferences and the aggregate cannabis production function (in MA). We use the Chow Test to deal with heterogeneity. We aggregate daily production data into monthly and quarterly averages, calculate sales differences, and get supplemental data from FRED. Then, we plot weekly and monthly data and test for structural breaks. This is another step in the classical work on cannabis statistics that you won't want to miss.

## Objectives
    
1. Test if there has been a structural break in consumer preferences. Perhaps from a structural change in sales per capita or sales as a percent of gross domestic income.
2. Test if there has been a structural break in the production function.

## Data Sources

*MA Cannabis Control Commission*

- [Retail Sales by Date and Product Type](https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/xwf2-j7g9)
- [Approved Massachusetts Licensees](https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/hmwt-yiqy)
- [Average Monthly Price per Ounce for Adult-Use Cannabis](https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/rqtv-uenj)
- [Plant Activity and Volume](https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu)
- [Weekly sales by product type](https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/87rp-xn9v)

*Fed Fred*

- [MA Population](https://fred.stlouisfed.org/series/MAPOP)
- [MA Median Income](https://fred.stlouisfed.org/series/MEHOINUSMAA646N)
- [MA Income per Capita](https://fred.stlouisfed.org/series/MAPCPI)

## References
    
- [The Chow Test — Dealing with Heterogeneity in Python](https://medium.com/@remycanario17/the-chow-test-dealing-with-heterogeneity-in-python-1b9057f0f07a)
- [Tests for structural breaks in time-series data](https://www.stata.com/features/overview/structural-breaks/)
- [Difference-in-Difference Estimation](https://www.publichealth.columbia.edu/research/population-health-methods/difference-difference-estimation)
