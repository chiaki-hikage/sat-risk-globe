# 3D Globe + Moving Satellites + Risk Layer (Streamlit)

3D地球儀上で衛星（TLE）を動かしつつ、通信影響のダミーリスクレイヤー（色）を時間とともに変動表示する Streamlit アプリです。

- ✅ **TLE はローカルファイル読み込み**（CelesTrak 403 / IPブロック回避）
- ✅ **国境線 / 海岸線はローカル GeoJSON 読み込み**（GeoPandas不要、外部障害に強い）
- ✅ Play/Stop と時間ジャンプに対応

---

## Demo / Features

- 地球儀（ドラッグで回転、スクロールでズーム）
- 衛星点群（ISS/Stations, GNSS, Weather, Starlinkサンプル）
- 国境線・海岸線（GeoJSON）
- リスクレイヤー（ダミー）
  - 高緯度帯の時間変動 + 衛星直下点周辺のスポット

---

## Repository structure

```text
.
├── app.py
├── requirements.txt
└── data/
    ├── tle/
    │   ├── stations.tle
    │   ├── gnss.tle
    │   ├── weather.tle
    │   └── starlink_sample.tle
    └── map/
        ├── 110m_borders.geojson
        ├── 110m_coastline.geojson
        ├── 50m_borders.geojson        (optional)
        ├── 50m_coastline.geojson      (optional)
        ├── 10m_borders.geojson        (optional)
        └── 10m_coastline.geojson      (optional)
```
---

## run locally

```ruby
pip install -r requirements.txt
streamlit run app.py
```

