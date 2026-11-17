from datetime import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import requests

CLIENT_ID = "8879f6316c91d25a29cce75caebb9007"
API_KEY = "8ce8d4984b31a9b0e6bd399c7c65c656"
EVA_NO = "8000266"  # Hagen Hbf (Dortmund Hbf: 8000080, Wuppertal Hbf: 8000266)

HEADERS = {
    "DB-Client-Id": CLIENT_ID,
    "DB-Api-Key": API_KEY,
    "accept": "application/xml",
}
BASE_URL = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"


def parse_db_time(ts: str):
    # DB format is YYMMDDHHMM (Europe/Berlin local time)
    return datetime.strptime(ts, "%y%m%d%H%M") if ts else None


def get_live_station_board(eva_no: str) -> pd.DataFrame:
    now = datetime.now()
    date_str = now.strftime("%y%m%d")
    hour_str = now.strftime("%H")

    # 1. Fetch static planned schedule for the current hour
    plan_resp = requests.get(
        f"{BASE_URL}/plan/{eva_no}/{date_str}/{hour_str}",
        headers=HEADERS,
        timeout=15,
    )
    plan_resp.raise_for_status()
    plan_root = ET.fromstring(plan_resp.text)

    stops = {}
    for s in plan_root.findall("s"):
        stop_id = s.attrib.get("id")
        tl = s.find("tl")  # train label metadata
        dp = s.find("dp")  # departure info
        if dp is not None and tl is not None:
            category = tl.attrib.get("c", "")
            line = dp.attrib.get("l") or tl.attrib.get("n", "")
            stops[stop_id] = {
                "stop_id": stop_id,
                "train": f"{category} {line}".strip(),
                "planned_dep": parse_db_time(dp.attrib.get("pt")),
                "planned_platform": dp.attrib.get("pp"),
                "actual_dep": parse_db_time(dp.attrib.get("pt")),
                "delay_min": 0,
                "cancelled": False,
            }

    # 2. Fetch live changes (/fchg) and merge on stop_id
    fchg_resp = requests.get(
        f"{BASE_URL}/fchg/{eva_no}", headers=HEADERS, timeout=15
    )
    fchg_resp.raise_for_status()
    fchg_root = ET.fromstring(fchg_resp.text)

    for s in fchg_root.findall("s"):
        stop_id = s.attrib.get("id")
        if stop_id in stops:
            dp = s.find("dp")
            if dp is not None:
                if dp.attrib.get("cs") == "c":
                    stops[stop_id]["cancelled"] = True
                if dp.attrib.get("ct"):
                    actual = parse_db_time(dp.attrib.get("ct"))
                    stops[stop_id]["actual_dep"] = actual
                    diff = (
                        actual - stops[stop_id]["planned_dep"]
                    ).total_seconds() / 60
                    stops[stop_id]["delay_min"] = int(diff)

    df = pd.DataFrame(stops.values())
    return df.sort_values("planned_dep") if not df.empty else df


if __name__ == "__main__":
    board = get_live_station_board(EVA_NO)
    print(board.to_string(index=False))
    board.to_csv("hagen_live_sample.csv", index=False)
    print("\nSaved sample to hagen_live_sample.csv")