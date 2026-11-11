# Things to know about the data

The workshops use the [Synthetic Vehicle Fleet Telemetry](https://huggingface.co/datasets/renumics/synthetic-vehicle-fleet-telemetry)
dataset (Renumics, CC BY-SA 4.0). Like real measurement data, it has quirks. They hold for anyone reading
the silver layer.

**Principle: the stored data is never modified.** Silver keeps every measured value exactly as logged.
Each quirk is *flagged* (in `impulse_silver_dq_findings` where it can be checked automatically) and
*handled in the TSAL signal definition* of the analysis that uses the signal.

| Signal | Quirk | Handling in the labs |
|---|---|---|
| `PT.BattHV.Temp` | labelled `degC`, logged in **Kelvin** (≈ 293–335) | `batt_temp - 273.15` |
| `Vhcl.PoI.GCS.Lat` / `.Long` | labelled degrees, logged in **radians** | `× 180/π` (mountain drives: Engadin / Bernina Pass; lap sessions: Hockenheimring) |
| `Vhcl.v` | m/s with implausible spikes > 2,000 m/s; trip **95** (BMW_5) is corrupted throughout, trips 37 and 51 after the first seconds | drop values > 100 m/s: `v.where(v < 100)` (benchmark DQ-01) |
| `PT.BCU.BattHV.SOC` — *state of charge*, how full the HV battery is (% of usable capacity) | **0** for a few samples while the BMS initialises (22 trips); unit `none` but holds **percent** | ignore SOC ≤ 0: `soc.where(soc > 0)` |
| `Env.Temperature` | labelled `degC`, but some trips log Kelvin (range −10 … 299) | **not used** for analysis |
| `PT.PwrSupply.HV1.Pwr_aux` | −(charger power) while plugged in; in 16 trips the charger keeps feeding ~90 kW while the car drives | not used; charging is defined as current < −10 A **at standstill** |
| fuel | no fuel-flow signal for combustion / hybrid vehicles | energy analyses cover the electric drive only (EVs / hybrids) |
| `PT.Motor.*`, `PT.BattHV.*` | only present in the 63 electrified trips | expected: combustion cars have no e-motor; the `drive_unit_speed` alias falls back to the engine |
| signal timestamps | time-normalized: **every trip starts at 2026-03-20 12:00:00** | order trips by `container_metrics.trip_start_dt` (the original trip start from the metadata) |
| `Car.Gen.vx_1` | independent longitudinal velocity; agrees with `Vhcl.v` where `Vhcl.v` is valid | corroborating sensor for DQ checks |

Where these show up:

* `workshop_1_intro/01_intro_to_impulse_tsal` — the first signal definitions handle speed and SOC.
* `workshop_2_deep_dive/01_bronze_to_silver_data_engineering` — step 0 finds the quirks, step 2 flags them.
* `workshop_2_deep_dive/02a_tsal_deep_dive` — Kelvin → °C, radians → degrees and the alias fallback.
