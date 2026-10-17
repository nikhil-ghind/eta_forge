-- 008: hourly weather observations for the residual model's weather features.
--
-- Stations rather than a gridded product: NOAA ISD / ASOS publish hourly METAR-derived
-- observations for JFK, LGA, EWR and Central Park, which is enough coverage for a city
-- and is what the training corpus can be joined against historically.
--
-- The join is as-of (latest observation at or before the departure) with a staleness
-- bound, done in Python by eta_forge/features/weather.py; `eta_weather_asof` below is
-- the same rule in SQL for ad-hoc analysis and backfills.

CREATE TABLE IF NOT EXISTS weather_station (
    station_id text PRIMARY KEY,
    name       text,
    location   geometry(Point, 4326) NOT NULL
);

CREATE TABLE IF NOT EXISTS weather_observation (
    station_id         text        NOT NULL REFERENCES weather_station(station_id) ON DELETE CASCADE,
    observed_at        timestamptz NOT NULL,
    temperature_c      double precision,
    precipitation_mm_h double precision NOT NULL DEFAULT 0,
    wind_kph           double precision NOT NULL DEFAULT 0,
    condition          text        NOT NULL DEFAULT 'clear',
    PRIMARY KEY (station_id, observed_at),
    CONSTRAINT weather_precip_ck CHECK (precipitation_mm_h >= 0),
    CONSTRAINT weather_wind_ck   CHECK (wind_kph >= 0),
    CONSTRAINT weather_condition_ck CHECK (condition IN
        ('clear','cloudy','fog','rain','heavy_rain','snow','storm'))
);

-- As-of lookups walk backwards in time from a departure: (station, time DESC).
CREATE INDEX IF NOT EXISTS weather_observation_asof_ix
    ON weather_observation (station_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS weather_station_geom_gix
    ON weather_station USING GIST (location);

INSERT INTO weather_station (station_id, name, location) VALUES
    ('USW00094728', 'Central Park',  ST_SetSRID(ST_MakePoint(-73.9692, 40.7789), 4326)),
    ('USW00014732', 'LaGuardia',     ST_SetSRID(ST_MakePoint(-73.8803, 40.7794), 4326)),
    ('USW00094789', 'JFK',           ST_SetSRID(ST_MakePoint(-73.7639, 40.6386), 4326)),
    ('USW00014734', 'Newark',        ST_SetSRID(ST_MakePoint(-74.1688, 40.6827), 4326))
ON CONFLICT (station_id) DO NOTHING;

-- Nearest station by KNN, then its latest observation not after p_at and not older
-- than the staleness bound. Returns no row (not a row of NULLs) when weather is
-- missing, so callers can distinguish "no data" from "a dry, calm 0 degree hour".
CREATE OR REPLACE FUNCTION eta_weather_asof(
    p_lat double precision,
    p_lon double precision,
    p_at  timestamptz,
    p_max_staleness interval DEFAULT interval '90 minutes'
) RETURNS TABLE (
    station_id text,
    observed_at timestamptz,
    temperature_c double precision,
    precipitation_mm_h double precision,
    wind_kph double precision,
    condition text
)
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    WITH nearest AS (
        SELECT s.station_id
        FROM weather_station s
        ORDER BY s.location <-> ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326)
        LIMIT 1
    )
    SELECT o.station_id, o.observed_at, o.temperature_c, o.precipitation_mm_h,
           o.wind_kph, o.condition
    FROM weather_observation o
    JOIN nearest n USING (station_id)
    WHERE o.observed_at <= p_at
      AND o.observed_at >  p_at - p_max_staleness
    ORDER BY o.observed_at DESC
    LIMIT 1
$$;
