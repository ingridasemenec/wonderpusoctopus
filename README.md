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
- The data is collected by [Copernicus Marine Dataset](https://www.copernicus.eu/en).
Copernicus Marine Datasets included:

Dataset 1: [Global Ocean Color](https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L4_MY_009_104/description) (satellite observations)
Chlorophyll 

Dataset 2: [Global Ocean Biochemistry Hindcast](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_BGC_001_029/description) (simulated)
O2; NO3; PO4; Si; Fe

Dataset 3: [Global Ocean OSTIA](https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011/description) (using in-situ & satellite data)
Sea Surface Temperature (analyzed_sst)

Dataset 4: [Multi Observation Global Ocean](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013/description) (using in-situ & satellite data)
Sea Surface Salinity (sos) and Sea Surface Density (dos)

Dataset 5: [Global Ocean Surface Carbon](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_BIO_CARBON_SURFACE_REP_015_008/description) (From model based on in-situ data)
Total alkalinity (talk); Surface partial pressure of CO2 (spco2); Sea water pH (ph); Calcite saturation state (omega_ca); Aragonite saturation state (omega_ar); Surface downward flux of CO2 (fgco2)

- Data processing:
  - ...

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
- **XGBoost**: eXtreme Gradient Boosting trained to predict chlorophyll concentrations using latitude, longitude, year, month, and biogeochmical features as input.
- **CNN**:
- **ConvLSTM**: Convolutional Long Short-Term Memory model trained to predict chlorophyll concentrations across both space and time.

## XGBoost
The flowchart below shows the modeling framework used for training an XGBoost regressor to predict chlorophyll concentrations. Briefly, the datasets described above were accessed using [Copernicus Mariner Toolbox API](https://help.marine.copernicus.eu/en/articles/7949409-copernicus-marine-toolbox-introduction). Data preprocessing was done to match the resolution between datasets (final resolution = 0.25° × 0.25°) and only chlorophyll values within 99 percentile were kept. Final dataset had 421801 data points with 19 features, which was further divided into training, validation and test sets. Before model training, correlated features were removes, i.e. only one feature among features with correlation > 0.8 was kept. XGBoost hypereparameters were tuned using a bayesian hyperparameter optimization framework. 

![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/modeling_framework.png" width=80%>

From the correlation heatmap below, feature sets with correlation > 0.8 are
1. [ph, spco2]
2. [omega_ar, omega_ca]
3. [sos, dos, talk, tco2]
4. [no3, po4]
5. [o2, analysed_sst]
6. 
![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/correlation_heatmap.png" width=80%>

After removing the correlated features, the final feature set included [’latitude’, ‘longitude’, ‘year, 'month', 'fgco2', 'omega_ca', 'ph', 'fe', 'no3', 'si', 'o2', ’sos’]. The hyperparamter for the XGBoost model were trained via bayesian hyperparameter optimization using [hyperopt library](http://hyperopt.github.io/hyperopt/) where the objective function was to minimize 5-fold CV RMSE on the training set. The range of hyperparameter values specifying the feature space were
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
![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/hyperparameter_tuning.png" width=80%>

The scatter plot above shows the 5-fold CV RMSE values on the trianing set for the 50 hyperparameter sets tried during tuning. Hyperparameter set corresponding to minimum 
5-fold CV RMSE after 50  iterations is selected for final model training. Comparing the performance of XGBoost regressor on the validation to the baseline model (where chlorophyll prediction = mean(chlorophyll values in the training set)), we can see singnificant improvement in both RMSE and MAPE.

![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/model_performance.png" width=30%>

The trianed XGBoost regressor had 0.32 RMSE and 0.17 MAPE on the test set. Next, we used [SHAP](https://shap.readthedocs.io/en/latest/index.html) (SHapley Additive exPlanations) for interpreting the XGBoost regressor model. From the graph below, we can see the pH levels and Fe concentrations are the top two features in prediciton of chlorophyll concentrations. Lower pH can reduce the availability of carbonate ions, which are crucial for the growth of phytoplankton. The North Sea is subject to various anthropogenic pressures, including pollution and carbon dioxide emissions, which can lead to changes in pH. Phytoplankton require iron for photosynthesis. In high-nutrient, low-chlorophyll (HNLC) regions, iron is often the limiting factor that controls phytoplankton growth. While the North Sea is not considered an HNLC region, iron can still play a significant role, particularly in shallow waters where it might be more readily available due to sediment resuspension.

![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/XGBoost/shap_summary_bar.png">

## ConvLSTM
The model captures key temporal patterns, as shown in the seasonality plot, but struggles with higher concentration areas, as indicated by the RMSE heatmap. The model tends to underpredict the concentration at the shoreline areas. However as we can see from the animated timesteps of prediction vs actual, we succeed in capturing the basic spacial structure as well as seasonal dependence. This demonstrates that this model is a viable option for this application if given more data and time.

![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/ConvLSTM/chlorophyll_animation_map.gif" width=80%>
![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/ConvLSTM/seasonality.png" width=60%>
![]() <img src="https://github.com/ingridasemenec/wonderpusoctopus/blob/main/ConvLSTM/rmseheatmap.png" width=60%>
