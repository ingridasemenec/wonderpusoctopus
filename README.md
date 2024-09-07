# Wonderpus Octopus
# Modeling the relationship between biogeochemical layers and chlorophyll density  

Prediction of chlorophyll concentration in the ocean aids in assessing ecosystem health, advising fisheries management, and enhancing our understanding of Earth's biogeochemical processes. By employing deep learning methods, we will develop models to analyze the relationship between chlorophyll concentration and biogeochemical features using the [Copernicus Marine Dataset](https://www.copernicus.eu/en).

## Authors
- [Francesca Balestrieri](https://github.com/fbalestrieri)
- [Deniz Olgu Devecioglu](https://github.com/heineborell)
- [Nadir Hajouji](https://github.com/nhajouji)
- [Saswat Mishra](https://github.com/sswtmshr)
- [Kshitiz Parihar](https://github.com/kparihar13)
- [Ingrida Semenec](https://github.com/ingridasemenec)

## Overview:
Chlorophyll concentration in the ocean reflects the interaction of factors like nutrient availability and sunlight, influencing oceanic biomass productivity. High chlorophyll levels suggest nutrient-rich waters where phytoplankton, the ocean's primary producers, thrive, which in turn impacts fishery population dynamics and ecosystem health. Additionaly, from chlorophyll patterns we can gain insight about climate change, ocean currents and mixing processes. A predictive and/or explanatory analysis using machine and deep learning models will be helpful in addressing these environmental challenges.

The project goals are:
- What environmental, biogeochemical, climatic features affect chlorophyll levels?
- Determine if the available data can predict impacts on fisheries, enabling policymakers to better prepare and respond. (we can take this out, not sure if we are gonna do this for now!)

## Stakeholders:
NOAA, Fish and Wildlife Department, local fisheries, coastal communities.
## KPI:
Predicting/Forecasting chlorophyll levels accurately.

## Dataset
The data is collected by [Copernicus Marine Dataset](https://www.copernicus.eu/en).
Copernicus Marine Datasets included:

- Dataset 1: [Global Ocean Color](https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L4_MY_009_104/description) (satellite observations)
  - Variable of interest: Chlorophyll 

- Dataset 2: [Global Ocean Biochemistry Hindcast](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_BGC_001_029/description) (simulated using PISCES model)
  - Variables of interest: Concentrations for dissolved oxygen (o2), nitrate (no3), phosphate (po4), silicate (Si), and iron (Fe)

- Dataset 3: [Global Ocean OSTIA](https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011/description) (using in-situ & satellite data)
  - Variable of interest: Sea Surface Temperature (analyzed_sst)

Dataset 4: [Multi Observation Global Ocean](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013/description) (using in-situ & satellite data)
- Variables of interest: Sea Surface Salinity (sos) and Sea Surface Density (dos)

Dataset 5: [Global Ocean Surface Carbon](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_BIO_CARBON_SURFACE_REP_015_008/description) (From model based on in-situ data)
 - Variables of interest: Total alkalinity (talk); Surface partial pressure of CO2 (spco2); Sea water pH (ph); Calcite saturation state (omega_ca); Aragonite saturation state (omega_ar); Surface downward flux of CO2 (fgco2)

## Approach
Select Ocean Regions: 
We are selecting shallower sea regions for our analysis. Shallow areas offer better light penetration for the productivity of phytoplankton growth, there’s better nutrient mixing between the seabed and the water, as well as the runoff nutrients from the land. Winds and tides in these areas offer physical nutrient mixing, influencing the phytoplankton blooms. All these reasons influence higher biodiversity and fishery feeding and spawning areas. Modeling chlorophyll density in these areas is vital for understanding fishery management, health, and productivity.

The selected region of interest: North Sea

Select time frame: Sep 1997 - Dec 2021

Research questions:

- What are the primary biogeochemical and physical factors influencing chlorophyll density in various shallow sea regions?
- How can these factors be quantitatively integrated into a robust predictive model for chlorophyll density?

Here, we outline a summary of our employed strategies:
- Collect and combine the datasets extracting the chosen base features for the model training.
- Perform exploratory data analysis, use XGBoost to get the first look at the data and feature importance using SHAP.
- Train deep learning models from the insights gained from EDA and XGBoost.

Methods:
- [**XGBoost**](https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/XGBoost.ipynb): eXtreme Gradient Boosting trained to predict chlorophyll concentrations using latitude, longitude, year, month, and biogeochmical features as input.
- [**CNN**](https://github.com/ingridasemenec/wonderpusoctopus/blob/main/CNN/CNN.ipynb): Convolutional Neural Network trained to predict the chlorophyll concentrations across both space and time.
- [**ConvLSTM**](https://github.com/ingridasemenec/wonderpusoctopus/blob/main/ConvLSTM/ConvLSTM.ipynb): Convolutional Long Short-Term Memory model trained to predict chlorophyll concentrations across both space and time.

## [XGBoost](https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/XGBoost.ipynb)
The flowchart below shows the modeling framework used for training an XGBoost regressor to predict chlorophyll concentrations. 

![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/modeling_framework.png" width=80%>

- The datasets described above were accessed using [Copernicus Mariner Toolbox API](https://help.marine.copernicus.eu/en/articles/7949409-copernicus-marine-toolbox-introduction) for the North Sea region (with latitude range [50, 62] and longitude range [-6, 12]), for the period from 1997-01-01 to 2023-01-01 (we remark that the overall overlapping period of all the datasets was from 1997 to 2021). Since we were only interested in the surface sea level, we retrieved data with depth range [0, 0.5].
- Data preprocessing involved
  - matching the resolution between datasets (final resolution = 0.25° × 0.25°, please refer to the Jupyter notebook `merged_datasets_EDA_and_XGBoost_049depth_data_cleaning.ipynb` in the XGBoost folder for more information on the merging process)
  - dealing with the outliers from the chlorophyll density values, whose distribution had (min, max, mean, stdev) = (0.075, 46.301, 0.976, 1.364), by removing all the values above 6.73 (corresponding to the 99th percentile of the chlorophyll values)
- Final dataset had 421801 data points with 19 features, which was further divided into training, validation and test sets.
- Before model training, correlated features were removes, i.e. only one feature among features with correlation > 0.8 was kept. From the correlation heatmap below, feature sets with correlation > 0.8 are
  1. [ph, spco2]
  2. [omega_ar, omega_ca]
  3. [sos, dos, talk, tco2]
  4. [no3, po4]
  5. [o2, analysed_sst]

![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/correlation_heatmap.png" width=80%>

- After removing the correlated features, the final feature set included 12 features, namely [’latitude’, ‘longitude’, ‘year, 'month', 'fgco2', 'omega_ca', 'ph', 'fe', 'no3', 'si', 'o2', ’sos’].
- The hyperparamter for the XGBoost model were tuned using a bayesian optimization framework using [hyperopt library](http://hyperopt.github.io/hyperopt/) where the objective function was to minimize 5-fold CV RMSE on the training set. The range of hyperparameter values specifying the feature space were
```
xgbr_param_space = {'max_depth': hp.choice('max_depth', range(3,9)),
                    'learning_rate': hp.uniform('learning_rate',0.01,0.5),
                    'subsample': hp.uniform('subsample',0.5,1.0),
                    'n_estimators': hp.choice('n_estimators', range(100,1000)),
                    'reg_lambda': hp.uniform('reg_lambda',0,5), #L2 regularization
                    'reg_alpha': hp.uniform('reg_alpha',0,5), #L1 regularization
                    'min_child_weight': hp.uniform('min_child_weight',0,10),
                    'gamma': hp.uniform('gamma',0,0.5),
                    'colsample_bytree': hp.uniform('colsample_bytree',0.5,1.0)}
```

- Hyperparameter set corresponding to minimum 5-fold CV RMSE after 50  iterations is selected for final model training. Comparing the performance of XGBoost regressor on the validation to the baseline model (where chlorophyll prediction = mean(chlorophyll values in the training set)), we can see singnificant improvement in both RMSE and MAPE.

![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/model_performance.png" width=35%>

The trained XGBoost regressor had 0.32 RMSE and 0.17 MAPE on the test set. Next, we used [SHAP](https://shap.readthedocs.io/en/latest/index.html) (SHapley Additive exPlanations) for interpreting the XGBoost regressor model. From the graph below, we can see the pH levels and Fe concentrations are the top two features in prediciton of chlorophyll concentrations. Lower pH can reduce the availability of carbonate ions, which are crucial for the growth of phytoplankton. The North Sea is subject to various anthropogenic pressures, including pollution and carbon dioxide emissions, which can lead to changes in pH. Phytoplankton require iron for photosynthesis. In high-nutrient, low-chlorophyll (HNLC) regions, iron is often the limiting factor that controls phytoplankton growth. While the North Sea is not considered an HNLC region, iron can still play a significant role, particularly in shallow waters where it might be more readily available due to sediment resuspension.

![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/shap_summary_bar.png" width=80%>

## [CNN](https://github.com/ingridasemenec/wonderpusoctopus/blob/main/CNN/CNN.ipynb)

The Convolutional Neural Network (CNN) model captures spatial dependencies in chlorophyll concentration data. This model is designed to process image-like data of the ocean area where spatial relationships between pixels can provide critical information for chlorophyll predictions. The CNN model is trained using normalized datasets and is evaluated based on the  RMSE (Root Mean Square Error) per pixel.

In addition to normalizing training data to have zero mean and unit standard deviation, we also process the NaN values as follows:
- First flatten each feature tensor, find and save the positions of NaNs and drop them out. (NaNs correspond to land coordinates so they are same across the dataset).
- Train the model on the data without NaNs.
- After prediction we inserted the NaNs back and get final plot.

![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/CNN/CNN_transformation.PNG" width=80%>

The model demonstrates promising results by identifying spatial features and patterns, although there is still room for improvement, especially in areas with higher chlorophyll concentrations close to the shore.

We compared multiple CNN architectures, including models using batch normalization, dropout, and deeper residual networks (ResNet-based), to evaluate their effectiveness in predicting chlorophyll concentrations. The RMSE values for both training and validation sets are displayed below. We found the NetResDeep model to perform the best among the three models.

![]()<img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/CNN/CNN_model_comp.png" width=80%>

The figure shows the RMSE per pixel for the NetResDeep model, indicating areas where the model performed well and where it struggled.
![]()<img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/CNN/CNN_RMSE.png" width=80%>

Finally, the following plot shows the Chorophyll prediction for the NetResDeep model.
![]()<img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/CNN/CNN_chlor.gif" width=80%>

## [ConvLSTM](https://github.com/ingridasemenec/wonderpusoctopus/blob/main/ConvLSTM/ConvLSTM.ipynb)
The model captures key temporal patterns, as shown in the seasonality plot, but struggles with higher concentration areas, as indicated by the RMSE heatmap. The model tends to underpredict the concentration at the shoreline areas. However as we can see from the animated timesteps of prediction vs actual, we succeed in capturing the basic spacial structure as well as seasonal dependence. This demonstrates that this model is a viable option for this application if given more data and time.

![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/ConvLSTM/chlorophyll_animation_map.gif" width=80%>
![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/ConvLSTM/seasonality.png" width=60%>
![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/ConvLSTM/rmseheatmap.png" width=60%>
