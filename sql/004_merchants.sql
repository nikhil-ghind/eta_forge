-- 004: merchants and order history for the prep-time half of the decomposition.
--
-- ETA = prep + travel + handoff. Prep is not a routing problem at all: it is a
-- per-merchant queueing problem, and it is frequently the largest and most variable
-- term in a delivery ETA. It gets its own table, its own history, and its own model.

CREATE TABLE IF NOT EXISTS merchant (
    merchant_id       text PRIMARY KEY,
    name              text,
    cuisine_category  text NOT NULL DEFAULT 'other',
    location          geometry(Point, 4326),
    -- Static parking/access difficulty at the pickup door, 0 (kerbside) .. 1 (tower).
    access_difficulty double precision NOT NULL DEFAULT 0.3,
    onboarded_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT merchant_access_ck CHECK (access_difficulty BETWEEN 0 AND 1)
);

ALTER TABLE merchant
    ADD COLUMN IF NOT EXISTS h3_r8 h3index
        GENERATED ALWAYS AS (h3_lat_lng_to_cell(location, 8)) STORED;

CREATE TABLE IF NOT EXISTS merchant_order (
    order_id     text PRIMARY KEY,
    merchant_id  text NOT NULL REFERENCES merchant(merchant_id) ON DELETE CASCADE,
    trip_id      text REFERENCES trip(trip_id) ON DELETE SET NULL,
    placed_at    timestamptz NOT NULL,
    ready_at     timestamptz,
    picked_up_at timestamptz,
    delivered_at timestamptz,
    item_count   integer NOT NULL DEFAULT 1,
    order_value_usd double precision,
    -- Merchant queue depth at placement time: the load signal that separates "this
    -- kitchen is slow" from "this kitchen is slammed right now".
    open_orders_at_placement integer,
    CONSTRAINT merchant_order_items_ck CHECK (item_count >= 0)
);

-- Observed prep duration, derived rather than reported: ready_at - placed_at when the
-- merchant confirms readiness, otherwise picked_up_at - placed_at, which overstates
-- prep by the courier's waiting time and is therefore flagged.
CREATE OR REPLACE VIEW merchant_prep_observation AS
SELECT
    o.order_id,
    o.merchant_id,
    m.cuisine_category,
    o.placed_at,
    o.item_count,
    o.order_value_usd,
    o.open_orders_at_placement,
    COALESCE(
        EXTRACT(EPOCH FROM (o.ready_at    - o.placed_at)),
        EXTRACT(EPOCH FROM (o.picked_up_at - o.placed_at))
    ) AS prep_s,
    (o.ready_at IS NULL) AS prep_is_upper_bound
FROM merchant_order o
JOIN merchant m USING (merchant_id)
WHERE COALESCE(o.ready_at, o.picked_up_at) IS NOT NULL
  AND COALESCE(
        EXTRACT(EPOCH FROM (o.ready_at    - o.placed_at)),
        EXTRACT(EPOCH FROM (o.picked_up_at - o.placed_at))
      ) BETWEEN 30 AND 7200;

-- Handoff dwell: the last-metre cost of finding a door, a lobby, a lift. Modelled per
-- destination H3 cell because it is a property of the built environment, not of the
-- courier.
CREATE TABLE IF NOT EXISTS handoff_dwell_observation (
    order_id        text PRIMARY KEY REFERENCES merchant_order(order_id) ON DELETE CASCADE,
    destination_h3_r8 h3index NOT NULL,
    arrived_at      timestamptz NOT NULL,
    dwell_s         double precision NOT NULL,
    CONSTRAINT handoff_dwell_ck CHECK (dwell_s >= 0 AND dwell_s < 3600)
);
