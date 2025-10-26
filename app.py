import os
import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pymongo import MongoClient

# Load environment variables
load_dotenv()

PG_SCHEMA = os.getenv("PG_SCHEMA", "public")

def qualify(sql: str) -> str:
    """Replace schema placeholder {S} with actual schema name"""
    return sql.replace("{S}.", f"{PG_SCHEMA}.")

# CONFIG: Postgres and Mongo Queries
CONFIG = {
    "postgres": {
    "enabled": True,
    "uri": os.getenv("PG_URI", "postgresql+psycopg2://postgres:password@localhost:5432/smart_guardian"),
    "queries": {
        "Patient: Real-time Vital Signs (Last 2 Hours)": {
            "sql": """
                SELECT hr.heart_rate, hr.systolic_bp, hr.diastolic_bp, hr.temperature, hr.record_time
                FROM {S}.health_record hr
                JOIN {S}.health_monitor_device hmd ON hr.device_id = hmd.device_id
                WHERE hmd.user_id = :user_id
                  AND hr.record_time >= NOW() - INTERVAL '2 hours'
                ORDER BY hr.record_time DESC;
            """,
            "chart": {"type": "table"},
            "tags": ["patient"],
            "params": ["user_id"]
        },
        "Patient: 7-Day Heart Rate Trend (Daily Avg)": {
            "sql": """
                SELECT DATE(hr.record_time) AS record_date,
                       AVG(hr.heart_rate) AS avg_heart_rate
                FROM {S}.health_record hr
                JOIN {S}.health_monitor_device hmd ON hr.device_id = hmd.device_id
                WHERE hmd.user_id = :user_id
                  AND hr.record_time >= NOW() - INTERVAL '7 days'
                GROUP BY record_date
                ORDER BY record_date;
            """,
            "chart": {"type": "line", "x": "record_date", "y": "avg_heart_rate"},
            "tags": ["patient"],
            "params": ["user_id"]
        },
        "Patient: Bound Device Status (Battery/Network)": {
            "sql": """
                SELECT hmd.device_type, hmd.battery_level, hmd.network_status, hmd.activation_time
                FROM {S}.health_monitor_device hmd
                WHERE hmd.user_id = :user_id
                  AND hmd.device_id = :device_id;
            """,
            "chart": {"type": "table"},
            "tags": ["patient"],
            "params": ["user_id", "device_id"]
        },
        "Caregiver: My Assigned Patients (table)": {
            "sql": """
                SELECT u.user_id, u.user_name, u.user_age, u.medical_history, u.user_phone
                FROM {S}.users u
                JOIN {S}.user_caregiver_mapping ucm ON u.user_id = ucm.user_id
                WHERE ucm.caregiver_id = :caregiver_id
                ORDER BY u.user_name;
            """,
            "chart": {"type": "table"},
            "tags": ["caregiver"],
            "params": ["caregiver_id"]
        },
        "Caregiver: Unread Notifications (Last 24h)": {
            "sql": """
                SELECT n.notification_id, n.notification_type, n.content, n.send_time, u.user_name
                FROM {S}.notification n
                JOIN {S}.users u ON n.user_id = u.user_id
                WHERE n.caregiver_id = :caregiver_id
                  AND n.notification_id = :notification_id
                  AND n.read_status = 'unread'
                  AND n.send_time >= NOW() - INTERVAL '24 hours'
                ORDER BY n.send_time DESC;
            """,
            "chart": {"type": "table"},
            "tags": ["caregiver"],
            "params": ["caregiver_id", "notification_id"]
        },
        "Caregiver: Patient Device Status Changes (Last 7 Days)": {
            "sql": """
                SELECT dsl.log_id, dsl.device_id, dsl.old_status, dsl.new_status, dsl.status_change_time, u.user_name
                FROM {S}.device_status_log dsl
                JOIN {S}.health_monitor_device hmd ON dsl.device_id = hmd.device_id
                JOIN {S}.users u ON hmd.user_id = u.user_id
                JOIN {S}.user_caregiver_mapping ucm ON u.user_id = ucm.user_id
                WHERE ucm.caregiver_id = :caregiver_id
                  AND ucm.mapping_id = :mapping_id
                  AND dsl.status_change_time >= NOW() - INTERVAL '7 days'
                ORDER BY dsl.status_change_time DESC;
            """,
            "chart": {"type": "table"},
            "tags": ["caregiver"],
            "params": ["caregiver_id", "mapping_id"]
        },
        "Doctor: My Patients' Health Records (Last 30 Days)": {
            "sql": """
                SELECT u.user_name, hr.heart_rate, hr.systolic_bp, hr.diastolic_bp, hr.record_time
                FROM {S}.health_record hr
                JOIN {S}.users u ON hr.user_id = u.user_id
                WHERE u.doctor_id = :doctor_id
                  AND hr.record_id = :record_id
                  AND hr.record_time >= NOW() - INTERVAL '30 days'
                ORDER BY hr.record_time DESC;
            """,
            "chart": {"type": "table"},
            "tags": ["doctor"],
            "params": ["doctor_id", "record_id"]
        },
        "Doctor: Patient Medical History & Doctor Info": {
            "sql": """
                SELECT u.user_name, u.medical_history, d.doctor_name, d.doctor_specialty, d.hospital_affiliation
                FROM {S}.users u
                JOIN {S}.doctor d ON u.doctor_id = d.doctor_id
                WHERE u.user_id = :user_id
                  AND d.doctor_id = :doctor_id;
            """,
            "chart": {"type": "table"},
            "tags": ["doctor"],
            "params": ["user_id", "doctor_id"]
        },
        "Doctor: High Blood Pressure Patients (Systolic > 140)": {
            "sql": """
                SELECT u.user_name, AVG(hr.systolic_bp) AS avg_systolic_bp
                FROM {S}.health_record hr
                JOIN {S}.users u ON hr.user_id = u.user_id
                WHERE u.doctor_id = :doctor_id
                  AND hr.record_time >= NOW() - INTERVAL '7 days'
                GROUP BY u.user_name
                HAVING AVG(hr.systolic_bp) > 140
                ORDER BY avg_systolic_bp DESC;
            """,
            "chart": {"type": "bar", "x": "user_name", "y": "avg_systolic_bp"},
            "tags": ["doctor"],
            "params": ["doctor_id"]
        },
        "Technician: Devices with Low Battery (<20%)": {
            "sql": """
                SELECT hmd.device_id, hmd.device_type, hmd.battery_level, u.user_name, u.user_phone
                FROM {S}.health_monitor_device hmd
                JOIN {S}.users u ON hmd.user_id = u.user_id
                WHERE hmd.device_id = :device_id
                  AND hmd.battery_level < 20
                ORDER BY hmd.battery_level ASC;
            """,
            "chart": {"type": "bar", "x": "device_id", "y": "battery_level"},
            "tags": ["technician"],
            "params": ["device_id"]
        },
        "Technician: Device Status Change Logs (Last 24h)": {
            "sql": """
                SELECT dsl.device_id, dsl.old_status, dsl.new_status, dsl.status_change_time
                FROM {S}.device_status_log dsl
                WHERE dsl.device_id = :device_id
                  AND dsl.status_change_time >= NOW() - INTERVAL '24 hours'
                ORDER BY dsl.status_change_time DESC;
            """,
            "chart": {"type": "table"},
            "tags": ["technician"],
            "params": ["device_id"]
        }
    }
},
    "mongo": {
        "enabled": True,
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "db_name": os.getenv("MONGO_DB", "Smart_Health_Monitoring"),  # Matches image data's DB context
        "queries": {
            # ---------------------- 1. Hourly Avg Heart Rate (Matches Image 2 Data) ----------------------
            "TS: Minutely avg heart rate (Patient, All Time)": {
                "collection": "sensor_readings_ts",  # Matches time-series collection
                "aggregate": [
                    {"$match": {
                        "patient_id": ":patient_id",  # Use "patient_id" (image: "PAT_8316") instead of "meta.resident_id"
                        "data_type": "heart_rate",    # Exact match with image 2's "data_type: heart_rate"
                        "reading_value": {"$exists": True}  # Ensure reading exists
                    }},
                    {"$project": {
                        "minute": {"$dateTrunc": {"date": "$timestamp", "unit": "minute"}},  # Use "timestamp" (image's time field)
                        "hr": "$reading_value"  # Directly map to image 2's "reading_value" (e.g., 89.2)
                    }},
                    {"$group": {
                        "_id": "$minute", 
                        "avg_hr": {"$avg": "$hr"},  # Calculate hourly average
                        "data_count": {"$count": {}}  # Count of readings per hour (for reliability check)
                    }},
                    {"$sort": {"_id": 1}}  # Sort by time ascending
                ],
                "chart": {"type": "line", "x": "_id", "y": "avg_hr"},  # Line chart for trend
                "tags": ["patient", "doctor", "caregiver"],
                "params": ["patient_id"]  # Requires patient_id (e.g., "PAT_8316")
            },

            # ---------------------- 2. Device Battery Status (Matches Image 1 Data) ----------------------
            "TS: Device Battery Status (All Time)": {
                "collection": "device_status_ts",  # Matches device status collection
                "aggregate": [
                    {"$match": {
                        "device_id": ":device_id",    # Exact match with image 1's "device_id: DEVICE_004"
                        "battery_level": {"$exists": True},  # Ensure battery data exists
                        "network_status": {"$exists": True}  # Optional: Retain network info for context
                    }},
                    {"$project": {
                        "day": {"$dateTrunc": {"date": "$timestamp", "unit": "day"}},  # Daily battery trend
                        "device_id": "$device_id",  # Retain device ID for multi-device comparison
                        "battery_level": "$battery_level",  # Directly map to image 1's "battery_level: 79"
                        "network_status": "$network_status"  # Optional: Add network status for reference
                    }},
                    {"$group": {
                        "_id": {"day": "$day", "device_id": "$device_id"},  # Group by day + device
                        "avg_battery": {"$avg": "$battery_level"},  # Daily average battery
                        "last_network_status": {"$last": "$network_status"}  # Latest network status that day
                    }},
                    {"$sort": {"_id.day": 1, "_id.device_id": 1}}  # Sort by time + device
                ],
                "chart": {"type": "line", "x": "_id.day", "y": "avg_battery", "color": "_id.device_id"},  # Color by device
                "tags": ["technician", "caregiver"],
                "params": ["device_id"]  # Requires device_id (e.g., "DEVICE_004")
            },

            # ---------------------- 3. Low SpO2 Alerts (Matches Image 2's SpO2 Data) ----------------------
            "Telemetry: Low SpO2 Alerts (All Time)": {
                "collection": "sensor_readings_ts",
                "aggregate": [
                    {"$match": {
                        "data_type": "blood_oxygen",  # Exact match with image 2's "data_type: blood_oxygen"
                        "reading_value": {"$lt": 92},  # SpO2 < 92 = low (clinical threshold)
                        "patient_id": {"$exists": True}
                    }},
                    {"$group": {
                        "_id": "$patient_id",  # Group alerts by patient (e.g., "PAT_8316")
                        "alert_count": {"$count": {}},  # Total low SpO2 occurrences
                        "latest_alert_time": {"$max": "$timestamp"}  # Most recent alert time
                    }},
                    {"$project": {
                        "patient_id": "$_id",
                        "alert_count": 1,
                        "latest_alert_time": 1,
                        "_id": 0  # Hide default _id for readability
                    }},
                    {"$sort": {"alert_count": -1}}  # Prioritize patients with most alerts
                ],
                "chart": {"type": "bar", "x": "patient_id", "y": "alert_count"},  # Bar chart for alert comparison
                "tags": ["doctor", "caregiver"],
                "params": []  # No params (shows all patients with low SpO2)
            },

            # ---------------------- 4. Device Network Status Distribution (Matches Image 1) ----------------------
            "Telemetry: Device Network Status (All Time)": {
                "collection": "device_status_ts",
                "aggregate": [
                    {"$match": {
                        "network_status": {"$in": ["online", "offline", "intermittent"]},  # Match image 1's "network_status: online"
                        "device_id": {"$exists": True}
                    }},
                    {"$group": {
                        "_id": "$network_status",  # Group by network status (online/offline)
                        "device_count": {"$count": {}},  # Number of devices in this status
                        "affected_devices": {"$addToSet": "$device_id"}  # List of devices (for troubleshooting)
                    }},
                    {"$project": {
                        "network_status": "$_id",
                        "device_count": 1,
                        "affected_devices": {"$slice": ["$affected_devices", 5]},  # Show top 5 devices
                        "_id": 0
                    }},
                    {"$sort": {"device_count": -1}}
                ],
                "chart": {"type": "pie", "names": "network_status", "values": "device_count"},  # Pie chart for distribution
                "tags": ["technician", "caregiver"],
                "params": []
            },

            # ---------------------- 5. Sensor Reading Details (Raw Data for Verification) ----------------------
            "Telemetry: Raw Sensor Readings (Patient)": {
                "collection": "sensor_readings_ts",
                "aggregate": [
                    {"$match": {
                        "patient_id": ":patient_id",
                        "data_type": {"$in": ["heart_rate", "blood_oxygen"]}  # Show both HR and SpO2 (image 2's data types)
                    }},
                    {"$project": {
                        "timestamp": 1,
                        "patient_id": 1,
                        "sensor_id": 1,  # Match image 2's "sensor_id: HR_SENSOR_047"
                        "data_type": 1,
                        "reading_value": 1,
                        "unit": 1,  # Match image 2's "unit: bpm" / "unit: %"
                        "quality_score": 1  # Match image 2's "quality_score: 0.98" (for data reliability)
                    }},
                    {"$sort": {"timestamp": -1}},  # Show latest readings first
                    {"$limit": 100}  # Limit to 100 rows for performance
                ],
                "chart": {"type": "table"},  # Table for raw data verification
                "tags": ["patient", "doctor"],
                "params": ["patient_id"]
            }
        }
    }
}

# ---------------------- 3. Dashboard Core Logic ----------------------
st.set_page_config(page_title="SmartGuardian Health Dashboard", layout="wide")
st.title("SmartGuardian | Wearable Health Monitoring System (Postgres + MongoDB)")

def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for (k, v), c in zip(metrics.items(), cols):
        c.metric(k, v)

@st.cache_resource
def get_pg_engine(uri: str):
    return create_engine(uri, pool_pre_ping=True, future=True)

@st.cache_data(ttl=60)
def run_pg_query(_engine, sql: str, params: dict | None = None):
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

@st.cache_resource
def get_mongo_client(uri: str):
    return MongoClient(uri)

def mongo_overview(client: MongoClient, db_name: str):
    info = client.server_info()
    db = client[db_name]
    colls = db.list_collection_names()
    total_docs = sum(db[c].estimated_document_count() for c in colls) if colls else 0
    stats = db.command("dbstats")
    return {
        "MongoDB Database": db_name,
        "Collections": f"{len(colls):,}",
        "Total Docs (Est.)": f"{total_docs:,}",
        "Storage Size": f"{round(stats.get('storageSize',0)/1024/1024,1)} MB",
        "MongoDB Version": info.get("version", "unknown")
    }

@st.cache_data(ttl=60)
def run_mongo_aggregate(_client, db_name: str, coll: str, stages: list, params: dict | None = None):
    # Replace MongoDB aggregation placeholders with prefixed IDs
    if params:
        import json
        stages_str = json.dumps(stages)
        for k, v in params.items():
            stages_str = stages_str.replace(f":{k}", v)
        stages = json.loads(stages_str)
    db = _client[db_name]
    print(coll)
    print(stages)
    print(params)
    docs = list(db[coll].aggregate(stages, allowDiskUse=True))
    return pd.json_normalize(docs) if docs else pd.DataFrame()

def render_chart(df: pd.DataFrame, spec: dict):
    if df.empty:
        st.info("No rows.")
        return
    ctype = spec.get("type", "table")
    # light datetime parsing for x axes
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
    
    if ctype == "table":
        st.dataframe(df, use_container_width=True)
    elif ctype == "line":
        st.plotly_chart(px.line(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "bar":
        st.plotly_chart(px.bar(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "pie":
        st.plotly_chart(px.pie(df, names=spec["names"], values=spec["values"]), use_container_width=True)
    elif ctype == "heatmap":
        pivot = pd.pivot_table(df, index=spec["rows"], columns=spec["cols"], values=spec["values"], aggfunc="mean")
        st.plotly_chart(px.imshow(pivot, aspect="auto", origin="upper",
                                  labels=dict(x=spec["cols"], y=spec["rows"], color=spec["values"])),
                        use_container_width=True)
    elif ctype == "treemap":
        st.plotly_chart(px.treemap(df, path=spec["path"], values=spec["values"]), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# The following block of code is for the dashboard sidebar, where you can pick your users, provide parameters, etc.
with st.sidebar:
    st.header("Database Connections")
    pg_uri = st.text_input("Postgres URI", CONFIG["postgres"]["uri"])
    mongo_uri = st.text_input("MongoDB URI", CONFIG["mongo"]["uri"])
    mongo_db = st.text_input("MongoDB Database Name", CONFIG["mongo"]["db_name"])
    st.divider()

    auto_run = st.checkbox("Auto-run on selection change", value=False, key="auto_run_global")
    st.header("Role & Prefix IDs")
    role = st.selectbox("User Role", ["patient", "caregiver", "doctor", "technician", "all"], index=4)
    
    # ---------------------- Prefix ID Input (Match ERD Rules) ----------------------
    # User ID: U + integer (e.g., U101)
    user_id_num = st.number_input("User ID Number (for Patient)", min_value=1, value=1, step=1)
    formatted_user_num = f"{user_id_num:03d}" 
    user_id = f"U{formatted_user_num}"  # Format: U101
    
    # Caregiver ID: C + integer (e.g., C201)
    caregiver_id_num = st.number_input("Caregiver ID Number (for Caregiver)", min_value=1, value=1, step=1)
    formatted_caregiver_num = f"{caregiver_id_num:03d}"
    caregiver_id = f"C{formatted_caregiver_num}"  # Format: C201
    
    # Doctor ID: Doc + integer (e.g., Doc301)
    doctor_id_num = st.number_input("Doctor ID Number (for Doctor)", min_value=1, value=1, step=1)
    formatted_doctor_num = f"{doctor_id_num:03d}" 
    doctor_id = f"Doc{formatted_doctor_num}"
    
    # Notification ID: D + integer (e.g., D1001)
    notification_id_num = st.number_input("Notification ID Number (for Caregiver)", min_value=1, value=1, step=1)
    notification_doctor_num = f"{notification_id_num:03d}"
    notification_id = f"D{notification_doctor_num}"  # Format: D1001
    
    # Mapping ID: M + integer (e.g., M301)
    mapping_id_num = st.number_input("Mapping ID Number (for Caregiver)", min_value=1, value=1, step=1)
    mapping_doctor_num = f"{mapping_id_num:03d}"
    mapping_id = f"M{mapping_doctor_num}"  # Format: M301
    
    # Record ID: R + integer (e.g., R401)
    record_id_num = st.number_input("Record ID Number (for Doctor)", min_value=1, value=100, step=1)
    record_doctor_num = f"{record_id_num:03d}"
    record_id = f"R{record_doctor_num}"  # Format: R401
    
    # Device ID: D + integer (e.g., D001)
    device_id_num = st.number_input("Device ID Number (for All Roles)", min_value=1, value=1, step=1)
    formatted_device_num = f"{device_id_num:03d}" 
    device_id = f"D{formatted_device_num}"  # Format: D001 (3-digit padding for consistency)
    
    # Treatment Date (for Doctor)
    treatment_date = st.date_input("Treatment Date (for Doctor)", value=dt.date.today())
    
    # Patient ID
    patient_id = f"PAT_{formatted_user_num}"

    # Prefix ID Context (Pass to Queries)
    PARAMS_CTX = {
        "user_id": user_id,
        "caregiver_id": caregiver_id,
        "doctor_id": doctor_id,
        "notification_id": notification_id,
        "mapping_id": mapping_id,
        "record_id": record_id,
        "device_id": device_id,
        "treatment_date": str(treatment_date),
        "patient_id": patient_id
    }

#Postgres part of the dashboard
st.subheader("📊 Postgres: Structured Data (Patients/Devices/Alerts)")
try:

    eng = get_pg_engine(pg_uri)

    with st.expander("Run Postgres Query", expanded=True):
        # The following will filter queries by role
        def filter_queries_by_role(qdict: dict, role: str) -> dict:
            def ok(tags):
                t = [s.lower() for s in (tags or ["all"])]
                return "all" in t or role.lower() in t
            return {name: q for name, q in qdict.items() if ok(q.get("tags"))}
        
        pg_all = CONFIG["postgres"]["queries"]
        pg_q = filter_queries_by_role(pg_all, role)

        names = list(pg_q.keys()) or ["(no queries for this role)"]
        sel = st.selectbox("Choose a saved query", names, key="pg_sel")

        if sel in pg_q:
            q = pg_q[sel]
            sql = qualify(q["sql"])   
            st.code(sql, language="sql")

            run  = auto_run or st.button("▶ Run Postgres", key="pg_run")
            if run:
                wanted = q.get("params", [])
                params = {k: PARAMS_CTX[k] for k in wanted}
                df = run_pg_query(eng, sql, params=params)
                render_chart(df, q["chart"])
        else:
            st.info("No Postgres queries tagged for this role.")
except Exception as e:
    st.error(f"Postgres error: {e}")

# Mongo panel
if CONFIG["mongo"]["enabled"]:
    st.subheader("🍃 MongoDB")
    try:
        mongo_client = get_mongo_client(mongo_uri)   
        metric_row(mongo_overview(mongo_client, mongo_db))

        with st.expander("Run Mongo aggregation", expanded=True):
            mongo_query_names = list(CONFIG["mongo"]["queries"].keys())
            selm = st.selectbox("Choose a saved aggregation", mongo_query_names, key="mongo_sel")
            q = CONFIG["mongo"]["queries"][selm]
            st.write(f"**Collection:** `{q['collection']}`")
            st.code(str(q["aggregate"]), language="python")
            runm = auto_run or st.button("▶ Run Mongo", key="mongo_run")
            if runm:
                wanted = q.get("params", [])
                params = {k: PARAMS_CTX[k] for k in wanted}
                dfm = run_mongo_aggregate(mongo_client, mongo_db, q["collection"], q["aggregate"], params=params)
                render_chart(dfm, q["chart"])
    except Exception as e:
        st.error(f"Mongo error: {e}")