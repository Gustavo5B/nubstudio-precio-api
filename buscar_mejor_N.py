"""
Búsqueda del número óptimo de características (N) para el pipeline de selección.
Prueba N de 5 a 30, corre los 3 métodos de selección + votación + CV de 7 pliegues,
y reporta las métricas para Regresión Lineal y MLP.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

# ── Conexión y carga de datos ──────────────────────────────────────────────
DB_URL = (
    'postgresql+psycopg2://neondb_owner:npg_2T5BNyqMzgpw'
    '@ep-sweet-lake-ad4v82aj-pooler.c-2.us-east-1.aws.neon.tech'
    '/neondb?sslmode=require'
)
engine = create_engine(DB_URL)

query = '''
SELECT
    o.precio_base,
    c.nombre  AS categoria,
    t.nombre  AS tecnica,
    m.nombre  AS material,
    o.anio_creacion,
    COALESCE(o.dimensiones_alto,  0)::float AS alto_cm,
    COALESCE(o.dimensiones_ancho, 0)::float AS ancho_cm,
    COALESCE(o.peso_kg,           0)::float AS peso_kg,
    COALESCE(o.con_certificado,  false)::int AS con_certificado,
    COALESCE(o.permite_marco,    false)::int AS permite_marco,
    COALESCE(o.disponible_envio, false)::int AS disponible_envio,
    COALESCE(o.es_original,      true)::int  AS es_original
FROM obras o
JOIN categorias c ON c.id_categoria = o.id_categoria
JOIN tecnicas   t ON t.id_tecnica   = o.id_tecnica
JOIN materiales m ON m.id_material  = o.id_material
WHERE o.eliminada = false
  AND o.precio_base > 0
  AND o.precio_base IS NOT NULL
ORDER BY o.id_obra
'''

print("Cargando datos...")
df = pd.read_sql(query, engine)

# Área como feature derivada
df['area_cm2'] = df['alto_cm'] * df['ancho_cm']

# Outlier removal (P5-P95)
p5  = df['precio_base'].quantile(0.05)
p95 = df['precio_base'].quantile(0.95)
df_clean = df[(df['precio_base'] >= p5) & (df['precio_base'] <= p95)].copy()

# OHE
df_encoding = pd.get_dummies(df_clean,
                             columns=['categoria', 'tecnica', 'material'],
                             drop_first=True, dtype=int)

y = df_encoding['precio_base'].values
X = df_encoding.drop(columns=['precio_base', 'area_cm2'])

print(f"Dataset: {len(y)} observaciones, {X.shape[1]} variables tras OHE\n")

# ── Escalar para selección ─────────────────────────────────────────────────
X_scaled_sel = StandardScaler().fit_transform(X)

# Correlación con y (calculada una vez, no depende de N)
corr_abs = pd.Series(np.corrcoef(X_scaled_sel.T, y)[-1, :-1], index=X.columns).abs()

# Random Forest (calculado una vez)
print("Entrenando Random Forest para importancia...")
bosque = RandomForestRegressor(n_estimators=100, random_state=42)
bosque.fit(X_scaled_sel, y)
importancia = pd.Series(bosque.feature_importances_, index=X.columns)

# ── Búsqueda de N ──────────────────────────────────────────────────────────
valores_N = list(range(5, 31))   # 5 a 30

resultados = []

print("Probando diferentes valores de N...\n")
print(f"{'N':>3}  {'Feats':>5}  {'LR R2_CV':>9}  {'LR MAE_CV':>10}  {'MLP R2_CV':>10}  {'MLP MAE_CV':>11}  {'Votos>=2':>8}")
print("-" * 75)

for N in valores_N:
    # Top N por correlación
    top_corr = corr_abs.sort_values(ascending=False).head(N).index.tolist()

    # Top N por RFE
    selector_rfe = RFE(estimator=LinearRegression(), n_features_to_select=N)
    selector_rfe.fit(X_scaled_sel, y)
    top_rfe = X.columns[selector_rfe.support_].tolist()

    # Top N por Random Forest
    top_rf = importancia.sort_values(ascending=False).head(N).index.tolist()

    # Votación
    todas = sorted(set(top_corr + top_rfe + top_rf))
    votos = {v: (1 if v in top_corr else 0) +
                (1 if v in top_rfe  else 0) +
                (1 if v in top_rf   else 0) for v in todas}

    features_finales = [v for v, s in votos.items() if s >= 2]
    n_feats = len(features_finales)

    if n_feats == 0:
        print(f"{N:>3}  {'—':>5}  {'—':>9}  {'—':>10}  {'—':>10}  {'—':>11}  {'0':>8}")
        continue

    # Pipeline de CV
    entrada = X[features_finales]
    escalador = StandardScaler()
    X_cv = escalador.fit_transform(entrada)

    # Linear Regression CV
    cv_lr = cross_validate(LinearRegression(), X_cv, y, cv=7,
                           scoring=['r2', 'neg_mean_absolute_error'])
    lr_r2  = cv_lr['test_r2'].mean()
    lr_mae = (-cv_lr['test_neg_mean_absolute_error']).mean()

    # MLP CV
    cv_mlp = cross_validate(
        MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500,
                     activation='relu', solver='adam', random_state=42),
        X_cv, y, cv=7, scoring=['r2', 'neg_mean_absolute_error'])
    mlp_r2  = cv_mlp['test_r2'].mean()
    mlp_mae = (-cv_mlp['test_neg_mean_absolute_error']).mean()

    resultados.append({
        'N': N, 'features': n_feats,
        'lr_r2': lr_r2, 'lr_mae': lr_mae,
        'mlp_r2': mlp_r2, 'mlp_mae': mlp_mae,
        'features_finales': features_finales
    })

    print(f"{N:>3}  {n_feats:>5}  {lr_r2:>9.4f}  {lr_mae:>10.2f}  {mlp_r2:>10.4f}  {mlp_mae:>11.2f}  {n_feats:>8}")

# ── Resultado final ────────────────────────────────────────────────────────
df_res = pd.DataFrame(resultados)

mejor_lr  = df_res.loc[df_res['lr_r2'].idxmax()]
mejor_mlp = df_res.loc[df_res['mlp_r2'].idxmax()]

print("\n" + "="*75)
print(f"\nMEJOR N para Regresión Lineal  → N={int(mejor_lr['N'])}  |  R2_CV={mejor_lr['lr_r2']:.4f}  |  MAE_CV=${mejor_lr['lr_mae']:.2f}")
print(f"  Características seleccionadas ({int(mejor_lr['features'])}):")
for f in mejor_lr['features_finales']:
    print(f"    • {f}")

print(f"\nMEJOR N para Red Neuronal (MLP) → N={int(mejor_mlp['N'])}  |  R2_CV={mejor_mlp['mlp_r2']:.4f}  |  MAE_CV=${mejor_mlp['mlp_mae']:.2f}")
print(f"  Características seleccionadas ({int(mejor_mlp['features'])}):")
for f in mejor_mlp['features_finales']:
    print(f"    • {f}")

# Guardar tabla completa
df_res_print = df_res[['N','features','lr_r2','lr_mae','mlp_r2','mlp_mae']].copy()
df_res_print.columns = ['N','Feats_selec','LR_R2_CV','LR_MAE_CV','MLP_R2_CV','MLP_MAE_CV']
df_res_print = df_res_print.round({'LR_R2_CV':4,'LR_MAE_CV':2,'MLP_R2_CV':4,'MLP_MAE_CV':2})
print("\n\n=== TABLA COMPLETA ===")
print(df_res_print.to_string(index=False))
