import os
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

HEADERS = {
    "DB-Client-Id": os.environ["DB_CLIENT_ID"],
    "DB-Api-Key": os.environ["DB_API_KEY"],
    "accept": "application/xml"
}
BASE_URL = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"
DATABASE_URL = os.environ["DATABASE_URL"]
TARGET_STATIONS = ["8000142", "8000080", "8000266", "8000085"]


def parse_db_time(ts: str):
    return datetime.strptime(ts, "%y%m%d%H%M") if ts else None


def log_pipeline_run(cur, eva_no, endpoint, status, latency_ms, count=0, error=None):
    cur.execute(
        """
        INSERT INTO ops_pipeline_logs (eva_no, endpoint, http_status, latency_ms, records_upserted, error_message)
        VALUES (%s, %s, %s, %s, %s, %s)""",
        (eva_no, endpoint, status, latency_ms, count, error)
    )


def fetch_and_merge_station(cur, eva_no: str) -> list:
    now = datetime.now()
    date_str = now.strftime("%y%m%d")
    hour_str = now.strftime("%H")

    # 1. Fetch /plan baseline
    t0 = time.time()
    plan_resp = requests.get(f"{BASE_URL}/plan/{eva_no}/{date_str}/{hour_str}", headers=HEADERS, timeout=15)
    plan_latency = int((time.time() - t0) * 1000)
    plan_resp.raise_for_status()
    log_pipeline_run(cur, eva_no, "/plan", plan_resp.status_code, plan_latency)

    stops = {}
    for s in ET.fromstring(plan_resp.text).findall("s"):
        stop_id = s.attrib.get("id")
        tl = s.find("tl")
        dp = s.find("dp")
        if dp is not None and tl is not None:
            cat = tl.attrib.get("c", "")
            line = dp.attrib.get("l") or tl.attrib.get("n", "")
            planned_dt = parse_db_time(dp.attrib.get("pt"))
            planned_pp = dp.attrib.get("pp")
            stops[stop_id] = {
                "stop_id": stop_id,
                "eva_no": eva_no,
                "train_name": f"{cat} {line}".strip(),
                "planned_dep": planned_dt,
                "actual_dep": planned_dt,
                "planned_platform": planned_pp,
                "actual_platform": planned_pp,
                "delay_min": 0,
                "is_cancelled": False
            }

    # 2. Fetch /fchg live changes
    t1 = time.time()
    fchg_resp = requests.get(f"{BASE_URL}/fchg/{eva_no}", headers=HEADERS, timeout=15)
    fchg_latency = int((time.time() - t1) * 1000)
    fchg_resp.raise_for_status()

    for s in ET.fromstring(fchg_resp.text).findall("s"):
        stop_id = s.attrib.get("id")
        if stop_id in stops:
            dp = s.find("dp")
            if dp is not None:
                if dp.attrib.get("cs") == "c":
                    stops[stop_id]["is_cancelled"] = True
                if dp.attrib.get("cp"):
                    stops[stop_id]["actual_platform"] = dp.attrib.get("cp")
                if dp.attrib.get("ct"):
                    actual_dt = parse_db_time(dp.attrib.get("ct"))
                    stops[stop_id]["actual_dep"] = actual_dt
                    stops[stop_id]["delay_min"] = int((actual_dt - stops[stop_id]["planned_dep"]).total_seconds() / 60)

    rows = [
        (
            v["stop_id"], v["eva_no"], v["train_name"], v["planned_dep"],
            v["actual_dep"], v["planned_platform"], v["actual_platform"],
            v["delay_min"], v["is_cancelled"]
        )
        for v in stops.values()
    ]
    log_pipeline_run(cur, eva_no, "/fchg", fchg_resp.status_code, fchg_latency, len(rows))
    return rows


def run_ingestion():
    conn = psycopg2.connect(DATABASE_URL)
    upsert_sql = """
        INSERT INTO fact_train_stops (
            stop_id, eva_no, train_name, planned_dep, actual_dep,
            planned_platform, actual_platform, delay_min, is_cancelled
        )
        VALUES %s
        ON CONFLICT (stop_id) DO UPDATE SET
            actual_dep = EXCLUDED.actual_dep,
            actual_platform = EXCLUDED.actual_platform,
            delay_min = EXCLUDED.delay_min,
            is_cancelled = EXCLUDED.is_cancelled,
            last_updated = NOW();
    """
    with conn:
        with conn.cursor() as cur:
            for eva_no in TARGET_STATIONS:
                try:
                    rows = fetch_and_merge_station(cur, eva_no)
                    if rows:
                        execute_values(cur, upsert_sql, rows)
                        print(f"[{eva_no}] Upserted {len(rows)} stops.")
                except Exception as exc:
                    log_pipeline_run(cur, eva_no, "ERROR", 500, 0, 0, str(exc)[:250])
                    print(f"[{eva_no}] Failed: {exc}")
    conn.close()


if __name__ == "__main__":
    run_ingestion()