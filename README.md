# March-Machine-learning-Mania-2026

Pipeline para predecir probabilidades de victoria en el torneo NCAA 2026 (masculino y femenino) usando ingeniería de características históricas + regresión logística.

## Ejecutar

```bash
python train_ncaa_prob_model.py
```

Esto genera:

- `outputs/submission_stage2.csv`: probabilidades para cada enfrentamiento del `SampleSubmissionStage2`.
- `outputs/validation_report.txt`: validación histórica por temporada (log loss).
