-- 001: extensions.
--
-- postgis     — geometry, GiST indexes, ST_* geodesic functions
-- pgrouting   — pgr_dijkstra and friends over the road graph
-- h3 / h3_postgis — hexagonal cell indexing for origin/destination features
-- btree_gist  — lets a GiST index carry a scalar alongside a geometry, used by the
--               segment candidate search in the map matcher

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgrouting;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS h3;
CREATE EXTENSION IF NOT EXISTS h3_postgis CASCADE;

-- Everything in this schema is metric-reasoned but stored in EPSG:4326, so distance
-- work goes through geography casts rather than a projected SRID. That keeps the
-- graph usable across the whole region without picking a state-plane zone, at the
-- cost of a slightly more expensive distance computation.
DO $$
BEGIN
    IF current_setting('server_version_num')::int < 160000 THEN
        RAISE WARNING 'ETA Forge targets PostgreSQL 16+; running on %',
            current_setting('server_version');
    END IF;
END
$$;
