import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


file = '...path...'
df = pd.read_excel(file)

columns_to_drop = ['max_curvature', 'fft_std', 'acc_magnitude_max', 'acc_magnitude_rms']
df.drop(columns=columns_to_drop, inplace=True)

df_pivot = df.pivot_table(index=['condition', 'window_number'], columns='axis', 
                          values=['mean_curvature', 'std_first_derivative', 'fft_mean'])

df_pivot.columns = [f'{feat}_{axis}' for feat, axis in df_pivot.columns]
df_pivot.reset_index(inplace=True)

values = df.groupby(['condition', 'window_number'])[['acc_magnitude_mean', 'acc_magnitude_std', 'acc_magnitude_energy']].first().reset_index()
df_final = pd.merge(df_pivot, values, on=['condition', 'window_number'])

desired_order = [
    'mean_curvature_x', 'std_first_derivative_x', 'fft_mean_x',
    'mean_curvature_y', 'std_first_derivative_y', 'fft_mean_y',
    'mean_curvature_z', 'std_first_derivative_z', 'fft_mean_z',
    'acc_magnitude_mean', 'acc_magnitude_std', 'acc_magnitude_energy'
]

ordered_columns = ['condition', 'window_number'] + desired_order
df_final = df_final[ordered_columns]

output = '...path...'
df_final.to_excel(output, index=False)

X = df_final.drop(columns=['condition', 'window_number'])
y = df_final['condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



model_path = '/Users/mariakoryakina/Desktop/random_forest_model_wolfeiq.joblib'
joblib.dump(rf, model_path)
print(f'Model saved to {model_path}')
