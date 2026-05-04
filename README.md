# **DATA-543-CA-Energy**
**Term Project for UNC DATA 543 Spring 2026**

This project, developed by Ahmed Hamed, Alex Lapina, and Evan Schmid, aims to use a combination of pricing data, power demand, and climatic variables to predict future price conditions at NP15, a major power hub for Northern California. To do this, we obtained daily time series from 2010-Present data from a variety of sources, including NOAA, NEIC, CDEC, CAISO, and the Bloomberg Terminal. Due to the use of lagged datasets and the incomplete nature of the 2026 data, the full dataset provided to a series of machine learning models ranged from 1/9/2010 to 12/31/2025. Through the standard use of an 80/20 chronological test/train split, this meant that our validation time series began in 2022, allowing it to assess performance against a range of moderate-to-severe price spikes as well as standard volatility over a 3.5 year time period. 

We assessed model performance for a 1D hybrid-CNN model similar to the ones covered during lecture, as well as for a Random Forest model, which was briefly discussed during lectures on classification. We found that each model obtained reasonably strong overall performance, but lacked the ability to correctly predict the magnitude of the price spike when compared to validation data. Our model can reasonably assess that a spike should occur, but lacks the required resolution to support any specific financial risk mitigation strategies. In addition, we assessed this performance across three predictive time horizons: 1-day ahead, 2-days ahead, and 7-days ahead. As expected, model performance decreased significantly as the predictive horizon increased. However, we were able to gather valuable information through feature importance about what datasets had the most power for future price predictions. 

To view and run our code, either the Jupyter Notebook File final_model.ipynb or the Python file final_model.py can be run after cloning this repository. The Jupyter Notebook is recommended for being able to step-through the process. Non-standard required python packages include pyTorch, Scikit-Learn, and xgboost. 

After running, these files will produce full model comparisons and the visuals contained within our final presentation. Project sources outside of Financial Data from the Bloomberg Terminal can be found below. Full Raw Data files are located in the folder 'Local Data' to facilitate replication of our results. 

https://www.ncei.noaa.gov/access/past-weather/
https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/4/hdd/1/0/2000-2026
https://psl.noaa.gov/data/correlation/
https://www.caiso.com/library/



