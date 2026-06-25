# Sistema Portátil para Estudos e Aplicações de Manutenção Preditiva em Equipamentos Industriais

This repository contains the data acquisition code, machine learning models, and vibration datasets developed for this thesis.

## Repository Structure

| Folder | Description |
|---|---|
| `Data_rig1/` | Vibration data collected from test rig 1 |
| `Data_rig2/` | Vibration data collected from test rig 2 |
| `ML/` | Machine learning algorithms used for fault classification |
| `TwinCAT_MMS/` | TwinCAT PLC code for data acquisition and the monitoring system implementation |
| `TwinCAT_ShuttleMovement/` | TwinCAT PLC code for lateral shuttle motion control |

## Data Labels

Each data folder is named according to the condition under which it was recorded:

| Label | Description |
|---|---|
| `Normal` | Normal condition, no added mass, static setup, 1000 RPM |
| `Mass1` | Imbalance fault (mass 1), static setup, 1000 RPM |
| `Mass2` | Imbalance fault (mass 2), static setup, 1000 RPM |
| `Mass3` | Imbalance fault (mass 3), static setup, 1000 RPM |
| `Mass4` | Imbalance fault (mass 4), static setup, 1000 RPM |
| `Moving_Normal` | Normal condition, no added mass, lateral movement active, 1000 RPM |
| `Moving_Mass1` | Imbalance fault (mass 1), lateral movement active, 1000 RPM |
| `Moving_Mass2` | Imbalance fault (mass 2), lateral movement active, 1000 RPM |
| `Moving_Mass3` | Imbalance fault (mass 3), lateral movement active, 1000 RPM |
| `1.2x_Normal` | Normal condition, no added mass, static setup, 1200 RPM |
| `1.2x_Mass1` | Imbalance fault (mass 1), static setup, 1200 RPM |
| `1.2x_Mass2` | Imbalance fault (mass 2), static setup, 1200 RPM |
| `1.2x_Mass3` | Imbalance fault (mass 3), static setup, ~1200 RPM |
| `1.2x_Moving_Normal` | Normal condition, no added mass, lateral movement active, ~1200 RPM |
| `1.2x_Moving_Mass1` | Imbalance fault (mass 1), lateral movement active, ~1200 RPM |
| `1.2x_Moving_Mass2` | Imbalance fault (mass 2), lateral movement active, ~1200 RPM |
| `1.2x_Moving_Mass3` | Imbalance fault (mass 3), lateral movement active, ~1200 RPM |
