# Canada ML Labour Visualizations

Two small ML projects using Canadian public data:

1) **Retraining/Reskilling Success Clusters**
   - Clusters Canadian education outcomes (employment/earnings/etc.) into interpretable “program outcome types”
   - Outputs: PCA cluster scatter + cluster profile plots + exemplar CSV for plain-language interpretation

2) **Job Market Tightness Predictor**
   - Builds a derived "tight vs loose" label from vacancy rate and unemployment rate
   - Trains a time-respecting classifier and visualizes performance + a Canada-level tightness time series

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
