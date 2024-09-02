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

Dataset 1: Global Ocean Color (satellite observations)
 Chlorophyll 

Dataset 2: Global Ocean Biochemistry (simulated)
O2; NO3; PO4; Si; Fe

Dataset 3: Global Ocean OSTIA (using in-situ & satellite data)
Sea Surface Temperature 

Dataset 4: Multi Observation Global Ocean (using in-situ & satellite data)
Sea Surface Salinity and Sea Surface Density

Dataset 5: Global Ocean Surface Carbon (From model based on in-situ data)
Dissolved inorganic carbon; Total alkalinity; Surface partial pressure of CO2
              Sea water pH; Calcite saturation state; Aragonite saturation state; Surface 
              downward flux of CO2
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

- **Random Forests**:
- **XgBoost**:
- **CNN**:
- **ConvLSTM**: Convolutional Long Short-Term Memory model trained to predict chlorophyll concentrations across both space and time.
  
## ConvLSTM
The model captures key temporal patterns, as shown in the seasonality plot, but struggles with higher concentration areas, as indicated by the RMSE heatmap. The model tends to underpredict the concentration at the shoreline areas. However as we can see from the animated timesteps of prediction vs actual, we succeed in capturing the basic spacial structure as well as seasonal dependence. This demonstrates that this model is a viable option for this application if given more data and time.

![]() <img src="https://github.com/ingridasemenec/wunderpusoctopus/tree/main/ConvLSTM/chlorophyll_animation_map.gif" width=70%>
![]() <img src="https://github.com/ingridasemenec/wunderpusoctopus/tree/main/ConvLSTM/seasonality.png" width=70%>
