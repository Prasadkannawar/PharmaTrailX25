import pandas as pd
import numpy as np

df = pd.read_csv('PharmaTrailX_ClinicalTrialMaster2.csv')
df['Visit_Date'] = pd.to_datetime(df['Visit_Date'])

# Test a few patients manually
test_patients = ['P00001', 'P00002', 'P00003', 'P00004', 'P00005']
results = []

for patient_id in test_patients:
    patient_data = df[df['Patient_ID'] == patient_id]
    
    avg_adherence = patient_data['Medication_Adherence(%)'].mean()
    adr_rate = patient_data['ADR_Reported'].mean()
    efficacy_trend = patient_data['Efficacy_Score'].diff().mean()
    visit_count = len(patient_data)
    expected_visits = patient_data['Week'].max()
    visit_compliance = visit_count / max(expected_visits, 1)
    
    seed_val = sum(ord(c) for c in str(patient_id)) % 1000
    np.random.seed(seed_val)
    
    base_prob = 0.2 + np.random.uniform(0, 0.3)
    adherence_factor = max(0, (75 - avg_adherence) / 100) * 0.3
    adr_factor = min(adr_rate * 0.4, 0.3)
    efficacy_factor = max(0, -efficacy_trend / 20) * 0.2 if not np.isnan(efficacy_trend) else 0
    compliance_factor = max(0, (1 - visit_compliance)) * 0.2
    
    delay_prob = base_prob + adherence_factor + adr_factor + efficacy_factor + compliance_factor
    delay_prob = min(max(delay_prob, 0.05), 0.95)
    
    threshold = 0.45 + np.random.uniform(-0.1, 0.1)
    is_delayed = delay_prob > threshold
    
    results.append({
        'patient': patient_id,
        'adherence': avg_adherence,
        'adr_rate': adr_rate,
        'delay_prob': delay_prob,
        'threshold': threshold,
        'is_delayed': is_delayed
    })

for r in results:
    print(f'{r["patient"]}: adherence={r["adherence"]:.1f}, delay_prob={r["delay_prob"]:.3f}, threshold={r["threshold"]:.3f}, delayed={r["is_delayed"]}')

# Check overall adherence distribution
print(f'\nOverall adherence stats:')
print(f'Mean: {df["Medication_Adherence(%)"].mean():.1f}')
print(f'Min: {df["Medication_Adherence(%)"].min():.1f}')
print(f'Max: {df["Medication_Adherence(%)"].max():.1f}')

# Test with lower threshold to get more balance
print(f'\nTesting with lower threshold (0.3):')
delayed_count = 0
total_count = 0

for patient_id in df['Patient_ID'].unique()[:100]:  # Test first 100 patients
    patient_data = df[df['Patient_ID'] == patient_id]
    
    avg_adherence = patient_data['Medication_Adherence(%)'].mean()
    adr_rate = patient_data['ADR_Reported'].mean()
    
    seed_val = sum(ord(c) for c in str(patient_id)) % 1000
    np.random.seed(seed_val)
    
    base_prob = 0.1 + np.random.uniform(0, 0.4)  # 0.1 to 0.5
    delay_prob = base_prob + adr_rate * 0.2
    delay_prob = min(max(delay_prob, 0.05), 0.95)
    
    is_delayed = delay_prob > 0.3  # Lower threshold
    if is_delayed:
        delayed_count += 1
    total_count += 1

print(f'With threshold 0.3: {delayed_count}/{total_count} delayed ({delayed_count/total_count:.2%})')
