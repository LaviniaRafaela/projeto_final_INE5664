import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import NNRaw.neural_network as NN


def stratify_cl(x):

    if x <= 3:
        return 0

    elif x <= 60:
        return 1

    elif x <= 300:
        return 2

    elif x <= 1500:
        return 3

    else:
        return 4

df = pd.read_excel("dataset_cl_completo.xlsx")

X_cols = [
    "distance_to_coast",
    "elevation",
    "merra2_sssmass",
    "era5_wind_speed",
    "era5_time_of_wind",
    "roughness_mean"
]

y_col = "cl"


df["salinity_class"] = df[y_col].apply(
    stratify_cl
)

#===
train_projects = [
    "international_micat_1998",
    "international_isocorrag_2010",
    "chile_vera_2012",
    "spain_canary_isles_2019",
    "international_cole_2003",
    "brazil_nutec_2008",
    "brazil_sica_2006",
    "brazil_portella_2011",
    "brazil_vitali_2013",
    "brazil_brambilla_2011",
    "brazil_pontes_2011"
]

test_projects = [
    "brazil_portella_2012",
    "brazil_vitali_2013",
    "brazil_brambilla_2011",
    "brazil_pontes_2011"
]

df_train = df[
    df["project_name"].isin(train_projects)
].copy()

df_test = df[
    df["project_name"].isin(test_projects)
].copy()

X_train = df_train[X_cols].to_numpy()
y_train = df_train[y_col].to_numpy().reshape(-1,1)

X_test = df_test[X_cols]
y_test = df_test[y_col]

stratify_train = df_train["salinity_class"]

#======
rede = NN.RedeNeural(perda="cross_entropy_categorica")
rede.adicionar_camada(6, 8, "relu", inicializacao="aleatorio")
rede.adicionar_camada(8,5,ativacao="softmax",inicializacao="aleatorio")


print(rede.resumo())


rede.treinar(X_train, y_train, epocas=100, taxa_aprendizado=0.01)

'''
y_pred = rede.prever(X_test)
acc = acuracia(y_pred, y_val, multiclasse=True)
print(f"\n✓ Acurácia na validação: {acc * 100:.1f}%")
'''