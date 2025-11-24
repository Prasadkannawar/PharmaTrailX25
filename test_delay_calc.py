import pandas as pd
import numpy as np

df = pd.read_csv('PharmaTrailX_ClinicalTrialMaster2.csv')
df['Visit_Date'] = pd.to_datetime(df['Visit_Date'])

# Test the exact logic from simple_test.py
patient_delay_data = []
patient_groups = df.groupby(['Patient_ID'])

for i, (patient_id, patient_data) in enumerate(patient_groups):
    if i >= 20:  # Test first 20 patients
        break
        
    avg_adherence = patient_data['Medication_Adherence(%)'].mean()
    adr_rate = patient_data['ADR_Reported'].mean()
    efficacy_trend = patient_data['Efficacy_Score'].diff().mean()
    visit_count = len(patient_data)
    expected_visits = patient_data['Week'].max()
    visit_compliance = visit_count / max(expected_visits, 1)
    
    seed_val = sum(ord(c) for c in str(patient_id)) % 1000
    np.random.seed(seed_val)
    
    base_prob = np.random.uniform(0.1, 0.6)
    adherence_factor = max(0, (80 - avg_adherence) / 100) * 0.2
    adr_factor = adr_rate * 0.15
    efficacy_factor = max(0, -efficacy_trend / 30) * 0.1 if not np.isnan(efficacy_trend) else 0
    good_adherence_bonus = max(0, (avg_adherence - 85) / 100) * 0.15
    
    delay_prob = base_prob + adherence_factor + adr_factor + efficacy_factor - good_adherence_bonus
    delay_prob = min(max(delay_prob, 0.05), 0.95)
    
    is_delayed = delay_prob > 0.5
    
    patient_delay_data.append({
        'Patient_ID': patient_id,
        'delay_probability': delay_prob,
        'is_delayed': is_delayed,
        'delay_days': int(delay_prob * 45) if is_delayed else 0
    })
    
    print(f'{patient_id}: prob={delay_prob:.3f}, delayed={is_delayed}')

print(f'\nCreated {len(patient_delay_data)} patient records')
delay_df = pd.DataFrame(patient_delay_data)
print(f'Delay distribution: {delay_df["is_delayed"].value_counts().to_dict()}')

# Test merge
sample_df = df[df['Patient_ID'].isin(delay_df['Patient_ID'].tolist())]
print(f'Sample df shape: {sample_df.shape}')
merged = sample_df.merge(delay_df, on='Patient_ID', how='left')
print(f'Merged shape: {merged.shape}')
print(f'Merged delay distribution: {merged["is_delayed"].value_counts().to_dict()}')
print(f'NaN count: {merged["is_delayed"].isna().sum()}')
