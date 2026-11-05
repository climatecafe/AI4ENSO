# ENSO indices (`ai4enso.indices`)

All functions take numpy arrays: fields shaped (..., lat, lon) with 1-D `lat` and `lon`
(0–360 or −180–180), and monthly time series with an array of calendar months (1–12).
Area means are weighted by cos(latitude) and ignore NaNs, so land can be masked.

## Index definitions

| Index | Definition | Function | Reference |
|---|---|---|---|
| Niño 1+2 | SST anomaly, 10°S–0°, 90°W–80°W | `nino_index(..., "nino12")` | |
| Niño 3 | 5°S–5°N, 150°W–90°W | `nino_index(..., "nino3")` | |
| Niño 3.4 | 5°S–5°N, 170°W–120°W | `nino_index(..., "nino34")` | |
| Niño 4 | 5°S–5°N, 160°E–150°W | `nino_index(..., "nino4")` | |
| ONI | 3-month centred running mean of Niño 3.4 anomalies | `oni` | NOAA CPC |
| Relative Niño 3.4 | Niño 3.4 minus the 20°S–20°N mean anomaly, rescaled to the Niño 3.4 std | `relative_nino34` | van Oldenborgh et al. (2021) |
| TNI | standardized Niño 1+2 minus standardized Niño 4 (5-month smoothing) | `tni` | Trenberth & Stepaniak (2001) |
| EMI | A − 0.5 B − 0.5 C (El Niño Modoki boxes) | `emi` | Ashok et al. (2007) |
| E and C | (PC1 ∓ PC2)/√2 from EOFs of tropical Pacific SST | `e_c_indices` | Takahashi et al. (2011) |
| SOI | standardized Tahiti minus standardized Darwin SLP anomaly | `soi` | |
| D20 | depth of the 20 °C isotherm | `isotherm_depth` | |
| WWV | volume above D20, 5°S–5°N, 120°E–80°W | `warm_water_volume` | Meinen & McPhaden (2000) |

EMI boxes: A = 10°S–10°N, 165°E–140°W; B = 15°S–5°N, 110°W–70°W; C = 10°S–20°N,
125°E–145°E.

Notes:

- **ONI base period.** NOAA computes ONI anomalies against centred 30-year base periods
  updated every 5 years. Compute anomalies that way (`monthly_anomalies` with a `base` mask
  for each window) for an exact match.
- **Relative Niño 3.4** removes the uniform tropical warming trend, which otherwise biases
  event counts in a warming climate. In the synthetic example, the ONI trend is 0.92 °C per
  century and the relative ONI trend is −0.10 °C per century.
- **E and C indices.** PCs are normalized to unit variance before the 45° rotation, and
  the signs are fixed so PC1 correlates positively with Niño 3.4 and EOF2 is positive over
  Niño 4. Check against Takahashi et al. (2011) if exact amplitudes matter. Remove trends
  (or the tropical-mean anomaly) first, otherwise an EOF can pick up the warming trend.
- **Warm water volume** counts grid cells whose centres fall in the box as whole cells,
  so it depends slightly on the grid near the box edges.

## Events and multi-year events

- `classify_events(oni)`: El Niño (La Niña) when ONI ≥ +0.5 (≤ −0.5) °C for at least five
  consecutive overlapping seasons (NOAA convention). It returns monthly labels and a list
  of events with start, end and peak.
- `winter_means(index, months)`: DJF means, labelled by December.
- `multiyear_events(djf)`: an event starts in a winter beyond `first_threshold` and
  continues while the following winters stay beyond `next_threshold` with the same sign.
  The defaults (0.5 and 0.4, in standard-deviation units) follow Bai et al. (2026).
- `multiyear_summary(events)`: counts of single- and multi-year El Niño and La Niña events.

## Example

```python
from ai4enso import indices as I

anom = I.monthly_anomalies(sst, months)            # sst: (time, lat, lon)
n34 = I.nino_index(anom, lat, lon, "nino34")
oni = I.oni(I.relative_nino34(anom, lat, lon))
labels, events = I.classify_events(oni)
djf, _ = I.winter_means(oni, months)
print(I.multiyear_summary(I.multiyear_events(djf / djf.std())))
```

`examples/01_enso_indices.py` runs these on a synthetic SST field. For real data, load an
SST product such as ERSSTv5 with xarray and pass `.values`.

## References

- Ashok, K., Behera, S. K., Rao, S. A., Weng, H., & Yamagata, T. (2007). El Niño Modoki and its possible teleconnection. *JGR*, 112, C11007.
- Bai, K., Jin, Y., Zhang, W., Lin, X., & Kirtman, B. (2026). Atmospheric noise forcing increases multiyear El Niño events. *GRL*, 53(14), e2026GL123553.
- Meinen, C. S., & McPhaden, M. J. (2000). Observations of warm water volume changes in the equatorial Pacific and their relationship to El Niño and La Niña. *J. Climate*, 13, 3551–3559.
- Takahashi, K., Montecinos, A., Goubanova, K., & Dewitte, B. (2011). ENSO regimes: Reinterpreting the canonical and Modoki El Niño. *GRL*, 38, L10704.
- Trenberth, K. E., & Stepaniak, D. P. (2001). Indices of El Niño evolution. *J. Climate*, 14, 1697–1701.
- van Oldenborgh, G. J., Hendon, H., Stockdale, T., L'Heureux, M., Coughlan de Perez, E., Singh, R., & van Aalst, M. (2021). Defining El Niño indices in a warming climate. *Environ. Res. Lett.*, 16, 044003.
