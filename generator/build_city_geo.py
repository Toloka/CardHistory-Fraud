"""City coordinates and UTC offsets for every merchant city in IBM's TabFormer data, for the travel times and time zones of trips.

Writes city_geo.json (next to this file): {"City|ST": [lat, lon, utc_offset_hours, how]} for every (city, state / country) pair.
  - US cities: the `zipcodes` package (median of the city's ZIP centroids, ZIPs at 0, 0 skipped; the city's most common time zone);
  - cities abroad: the `geonamescache` package, data from GeoNames (geonames.org, CC BY 4.0): cities of 15,000+ people, same name
    in the same country, the most populous one;
  - fallbacks: a US city not found -> its state's ZIP centroid and most common time zone; a city abroad not found -> the country's
    capital (else its most populous city in geonamescache).
  - utc_offset_hours is the zone's STANDARD offset (the smaller of its January and July offsets), so daylight saving is ignored.
The generator (build.py) only reads the JSON, so building the data needs neither package. city_geo.json is included in the
release; this script shows how it was made.
python generator/build_city_geo.py --std <work>/data/raw/tabformer_std.parquet [--out generator/city_geo.json]
"""
import argparse, collections, datetime as dt, json, os, unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

ALIASES = {"The Bahamas": "BS", "Myanmar (Burma)": "MM", "Cote d'Ivoire": "CI", "Macedonia": "MK", "Swaziland": "SZ",
           "Vatican City": "VA", "Cabo Verde": "CV", "South Korea": "KR", "North Korea": "KP", "Russia": "RU", "Czech Republic": "CZ",
           "United Kingdom": "GB", "United Arab Emirates": "AE", "Kosovo": "XK", "Taiwan": "TW", "Hong Kong": "HK", "Macau": "MO",
           "Republic of the Congo": "CG", "Democratic Republic of the Congo": "CD", "Congo": "CG", "Laos": "LA", "Vietnam": "VN",
           "Syria": "SY", "Iran": "IR", "Moldova": "MD", "Bolivia": "BO", "Venezuela": "VE", "Tanzania": "TZ", "Brunei": "BN",
           "East Timor": "TL", "Timor-Leste": "TL", "Micronesia": "FM", "Palestine": "PS", "St. Lucia": "LC", "Saint Lucia": "LC",
           "St. Kitts and Nevis": "KN", "Saint Kitts and Nevis": "KN", "St. Vincent and the Grenadines": "VC",
           "Saint Vincent and the Grenadines": "VC", "The Gambia": "GM", "Gambia": "GM", "Sao Tome and Principe": "ST",
           "Turks and Caicos Islands": "TC", "British Virgin Islands": "VG", "U.S. Virgin Islands": "VI", "Netherlands Antilles": "CW",
           "Curacao": "CW", "Guinea-Bissau": "GW", "Equatorial Guinea": "GQ", "Central African Republic": "CF", "South Sudan": "SS",
           "Netherlands": "NL", "East Timor (Timor-Leste)": "TL"}
# fallback time zone of a US state or territory when no ZIP code of it carries one
STATE_TZ = {"AL": "America/Chicago", "AK": "America/Anchorage", "AZ": "America/Phoenix", "AR": "America/Chicago", "CA": "America/Los_Angeles",
            "CO": "America/Denver", "CT": "America/New_York", "DE": "America/New_York", "DC": "America/New_York", "FL": "America/New_York",
            "GA": "America/New_York", "HI": "Pacific/Honolulu", "ID": "America/Boise", "IL": "America/Chicago", "IN": "America/Indiana/Indianapolis",
            "IA": "America/Chicago", "KS": "America/Chicago", "KY": "America/New_York", "LA": "America/Chicago", "ME": "America/New_York",
            "MD": "America/New_York", "MA": "America/New_York", "MI": "America/Detroit", "MN": "America/Chicago", "MS": "America/Chicago",
            "MO": "America/Chicago", "MT": "America/Denver", "NE": "America/Chicago", "NV": "America/Los_Angeles", "NH": "America/New_York",
            "NJ": "America/New_York", "NM": "America/Denver", "NY": "America/New_York", "NC": "America/New_York", "ND": "America/Chicago",
            "OH": "America/New_York", "OK": "America/Chicago", "OR": "America/Los_Angeles", "PA": "America/New_York", "RI": "America/New_York",
            "SC": "America/New_York", "SD": "America/Chicago", "TN": "America/Chicago", "TX": "America/Chicago", "UT": "America/Denver",
            "VT": "America/New_York", "VA": "America/New_York", "WA": "America/Los_Angeles", "WV": "America/New_York", "WI": "America/Chicago",
            "WY": "America/Denver", "PR": "America/Puerto_Rico", "VI": "America/St_Thomas", "GU": "Pacific/Guam", "AS": "Pacific/Pago_Pago",
            "MP": "Pacific/Saipan", "AA": "America/New_York", "AE": "Europe/Berlin", "AP": "Pacific/Guam"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper().strip()
    return " ".join(s.replace("-", " ").replace(".", "").split())


def std_offset(tz):
    z = ZoneInfo(tz)
    jan = z.utcoffset(dt.datetime(2019, 1, 15, 12)).total_seconds() / 3600
    jul = z.utcoffset(dt.datetime(2019, 7, 15, 12)).total_seconds() / 3600
    return min(jan, jul)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--std", default="data/raw/tabformer_std.parquet")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "city_geo.json"))
    a = ap.parse_args()
    import geonamescache, zipcodes
    dec = json.load(open(a.std.replace(".parquet", "_decode.json")))
    pairs = pl.read_parquet(a.std, columns=["city", "state"]).unique()
    city_d, state_d = dec["Merchant City"], dec["Merchant State"]

    # US: city -> ZIP centroids and time zones
    # a city's own ZIP codes first; ZIP codes that only list it as an acceptable name are used when it has none of its own
    prim_pts, alt_pts = collections.defaultdict(list), collections.defaultdict(list)
    us_tz, st_pts, st_tz = collections.defaultdict(collections.Counter), collections.defaultdict(list), collections.defaultdict(collections.Counter)
    for z in zipcodes.list_all():
        if not z["lat"] or not z["long"] or float(z["lat"]) == 0 or float(z["long"]) == 0: continue   # some PO-box ZIPs carry 0, 0
        p = (float(z["lat"]), float(z["long"])); tz = z.get("timezone") or ""
        prim_pts[(norm(z["city"]), z["state"])].append(p)
        if tz: us_tz[(norm(z["city"]), z["state"])][tz] += 1
        for c in z.get("acceptable_cities") or []:
            alt_pts[(norm(c), z["state"])].append(p)
            if tz and (norm(c), z["state"]) not in prim_pts: us_tz[(norm(c), z["state"])][tz] += 1
        st_pts[z["state"]].append(p)
        if tz: st_tz[z["state"]][tz] += 1
    us_pts = dict(alt_pts); us_pts.update(prim_pts)

    # abroad: geonamescache
    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()
    by_name = {norm(v["name"]): k for k, v in countries.items()}
    cities = list(gc.get_cities().values())
    gn = collections.defaultdict(list)
    for c in cities:
        names = [c["name"]] + [x for x in (c.get("alternatenames") or []) if x and len(x) < 40]
        for nm in set(norm(x) for x in names):
            gn[(nm, c["countrycode"])].append(c)
    biggest = {}
    for c in cities:
        cc = c["countrycode"]
        if cc not in biggest or c["population"] > biggest[cc]["population"]: biggest[cc] = c
    capital = {}
    for cc, v in countries.items():
        cands = gn.get((norm(v.get("capital", "")), cc))
        if cands: capital[cc] = max(cands, key=lambda x: x["population"])

    out, how = {}, collections.Counter()
    for ci, si in pairs.iter_rows():
        if ci is None or si is None or ci < 0 or si < 0: continue
        city, st = city_d[ci], state_d[si]
        if city.upper() == "ONLINE" or st in ("NA", ""): continue
        key = f"{city}|{st}"
        if len(st) == 2:
            pts = us_pts.get((norm(city), st))
            if pts:
                tz = (us_tz[(norm(city), st)].most_common(1)[0][0] if us_tz[(norm(city), st)] else
                      st_tz[st].most_common(1)[0][0] if st_tz[st] else STATE_TZ.get(st, "America/Chicago"))
                lat, lon = float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts])); h = "us_city"
            elif st in st_pts:
                pts = st_pts[st]; tz = st_tz[st].most_common(1)[0][0] if st_tz[st] else STATE_TZ.get(st, "America/Chicago")
                lat, lon = float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts])); h = "us_state"
            else:
                how["missing"] += 1; continue
        else:
            cc = ALIASES.get(st) or by_name.get(norm(st))
            if not cc:
                how["missing_country"] += 1; print("no country code:", st); continue
            cands = gn.get((norm(city), cc))
            if cands:
                c = max(cands, key=lambda x: x["population"]); h = "abroad_city"
            elif cc in capital:
                c = capital[cc]; h = "abroad_capital"
            elif cc in biggest:
                c = biggest[cc]; h = "abroad_biggest"
            else:
                how["missing_country"] += 1; print("no city for country:", st, cc); continue
            lat, lon, tz = float(c["latitude"]), float(c["longitude"]), c["timezone"]
        out[key] = [round(lat, 4), round(lon, 4), std_offset(tz), h]; how[h] += 1
    json.dump(out, open(a.out, "w"), indent=0, sort_keys=True)
    print(f"{len(out):,} cities written to {a.out}:", dict(how))


if __name__ == "__main__":
    main()
