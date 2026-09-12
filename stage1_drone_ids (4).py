#!/usr/bin/env python3
"""
PUSHPAK Grand Challenge 2026
Grand Challenge 3 - Security of Drones
Objective 2 - Drone Intrusion Detection System
Stage 1 Proof-of-Concept

This is a SINGLE-FILE implementation of the Stage 1 PoC.

What it demonstrates:
- Drone telemetry model
- Feature extraction
- Normal-flight simulation
- Controlled test-data scenarios
- Rule-based detection:
    * GPS spoofing
    * MAVLink anomaly
    * command anomaly
    * telemetry manipulation
    * DoS/rate anomaly
- Firmware SHA-256 integrity verification
- Hash-chained event logging
- Benchmarking
- JSON report generation
- Basic self-tests

This code is for controlled simulation / validation.
It does not transmit attack traffic or control a real drone.

Python:
    3.10+

Run:
    python stage1_drone_ids.py
    python stage1_drone_ids.py --duration 5
    python stage1_drone_ids.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


# ============================================================
# 1. CONFIGURATION
# ============================================================

THRESHOLDS = {
    # Proposed PoC values.
    # These are NOT official competition thresholds.
    "gps_speed_difference_mps": 18.0,
    "message_rate_high": 40.0,
    "command_rate_high": 12.0,
    "sensor_altitude_disagreement_m": 30.0,
}


# ============================================================
# 2. DATA MODELS
# ============================================================

@dataclass
class DroneTelemetry:
    timestamp: float

    latitude: float
    longitude: float

    gps_speed: float
    ground_speed: float

    altitude: float
    heading: float

    satellites: int
    hdop: float

    roll: float
    pitch: float

    battery_voltage: float
    flight_mode: str

    message_type: str
    message_rate: float
    command_count: int

    source: str = "stage1_simulator"

    scenario: str = "NORMAL"
    expected_attack: str = "NONE"


@dataclass
class FeatureVector:
    speed_difference: float
    position_change_m: float
    heading_change_deg: float
    altitude_change_m: float

    gps_quality_score: float

    message_rate: float
    command_rate: float

    sensor_speed_disagreement: float
    sensor_altitude_disagreement: float


@dataclass
class SecurityAlert:
    timestamp: float
    attack_type: str
    category: str
    severity: str
    confidence: float
    source: str
    evidence: Dict
    processing_latency_ms: float = 0.0


@dataclass
class ScenarioResult:
    scenario: str
    expected_attack: str
    detected_attack: str
    passed: bool
    processing_latency_ms: float


# ============================================================
# 3. GEOMETRY / TELEMETRY UTILITIES
# ============================================================

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate great-circle distance in metres."""

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlambda / 2.0) ** 2
    )

    a = max(0.0, min(1.0, a))

    return (
        2.0
        * EARTH_RADIUS_M
        * math.asin(math.sqrt(a))
    )


def angle_difference_deg(
    a: float,
    b: float,
) -> float:
    """Calculate smallest absolute angular difference."""

    return abs(
        (a - b + 180.0) % 360.0 - 180.0
    )


# ============================================================
# 4. NORMAL-FLIGHT SIMULATOR
# ============================================================

class NormalFlightSimulator:
    """
    Generate deterministic, controlled normal-flight telemetry.

    The simulator does NOT contain attack logic.
    """

    def __init__(
        self,
        duration_s: float = 5.0,
        sample_period_s: float = 0.1,
        seed: int = 42,
        start_lat: float = 19.1330,
        start_lon: float = 72.9150,
    ) -> None:

        if duration_s <= 0:
            raise ValueError(
                "duration_s must be greater than zero."
            )

        if sample_period_s <= 0:
            raise ValueError(
                "sample_period_s must be greater than zero."
            )

        self.duration_s = duration_s
        self.sample_period_s = sample_period_s
        self.rng = random.Random(seed)

        self.start_lat = start_lat
        self.start_lon = start_lon

    def generate(self) -> List[DroneTelemetry]:

        count = int(
            round(
                self.duration_s
                / self.sample_period_s
            )
        )

        lat = self.start_lat
        lon = self.start_lon

        records: List[DroneTelemetry] = []

        for i in range(count):

            t = (
                i
                * self.sample_period_s
            )

            ground_speed = (
                8.0
                + 0.45
                * math.sin(t / 3.5)
                + self.rng.uniform(
                    -0.08,
                    0.08,
                )
            )

            gps_speed = (
                ground_speed
                + self.rng.uniform(
                    -0.12,
                    0.12,
                )
            )

            heading = (
                35.0
                + 4.0
                * math.sin(t / 5.0)
                + self.rng.uniform(
                    -0.3,
                    0.3,
                )
            )

            altitude = (
                40.0
                + math.sin(t / 4.5)
                + self.rng.uniform(
                    -0.05,
                    0.05,
                )
            )

            north_m = (
                ground_speed
                * self.sample_period_s
                * math.cos(
                    math.radians(
                        heading
                    )
                )
            )

            east_m = (
                ground_speed
                * self.sample_period_s
                * math.sin(
                    math.radians(
                        heading
                    )
                )
            )

            lat += (
                north_m
                / 111_320.0
            )

            lon += (
                east_m
                /
                (
                    111_320.0
                    * max(
                        math.cos(
                            math.radians(
                                lat
                            )
                        ),
                        0.2,
                    )
                )
            )

            record = DroneTelemetry(
                timestamp=t,

                latitude=lat,
                longitude=lon,

                gps_speed=max(
                    gps_speed,
                    0.0,
                ),

                ground_speed=max(
                    ground_speed,
                    0.0,
                ),

                altitude=altitude,

                heading=heading % 360.0,

                satellites=12 + self.rng.choice(
                    [-1, 0, 0, 0, 1]
                ),

                hdop=max(
                    0.55,
                    0.90
                    + self.rng.uniform(
                        -0.08,
                        0.08,
                    ),
                ),

                roll=(
                    1.5
                    * math.sin(t / 2.5)
                ),

                pitch=(
                    1.2
                    * math.sin(t / 3.0)
                ),

                battery_voltage=(
                    16.4
                    - 0.005 * t
                ),

                flight_mode="GUIDED",

                message_type=(
                    "GLOBAL_POSITION_INT"
                ),

                message_rate=(
                    10.0
                    + self.rng.uniform(
                        -0.3,
                        0.3,
                    )
                ),

                command_count=i // 10,
            )

            records.append(record)

        return records


# ============================================================
# 5. CONTROLLED TEST SCENARIOS
# ============================================================

def base_flight(
    seed: int = 42,
) -> List[DroneTelemetry]:

    return NormalFlightSimulator(
        duration_s=5.0,
        sample_period_s=0.1,
        seed=seed,
    ).generate()


def normal_scenario(
    seed: int = 42,
) -> List[DroneTelemetry]:

    return base_flight(seed)


def gps_spoofing_scenario(
    seed: int = 42,
) -> List[DroneTelemetry]:

    data = base_flight(seed)

    start = max(
        1,
        len(data) // 2,
    )

    for i in range(
        start,
        len(data),
    ):

        item = data[i]

        data[i] = DroneTelemetry(
            **{
                **asdict(item),
                "gps_speed": 45.0,
                "scenario": "GPS_SPOOFING",
                "expected_attack": "GPS_SPOOFING",
            }
        )

    return data


def mavlink_anomaly_scenario(
    seed: int = 42,
) -> List[DroneTelemetry]:

    data = base_flight(seed)

    start = max(
        1,
        len(data) // 2,
    )

    for i in range(
        start,
        len(data),
    ):

        item = data[i]

        data[i] = DroneTelemetry(
            **{
                **asdict(item),
                "message_rate": 70.0,
                "scenario": "MAVLINK_ANOMALY",
                "expected_attack": "MAVLINK_ANOMALY",
            }
        )

    return data


def command_anomaly_scenario(
    seed: int = 42,
) -> List[DroneTelemetry]:

    data = base_flight(seed)

    start = max(
        1,
        len(data) // 2,
    )

    baseline = data[
        start - 1
    ].command_count

    for i in range(
        start,
        len(data),
    ):

        item = data[i]

        controlled_count = (
            baseline
            + 50 * (
                i - start + 1
            )
        )

        data[i] = DroneTelemetry(
            **{
                **asdict(item),
                "command_count": controlled_count,
                "scenario": "COMMAND_ANOMALY",
                "expected_attack": "COMMAND_ANOMALY",
            }
        )

    return data


def telemetry_manipulation_scenario(
    seed: int = 42,
) -> List[DroneTelemetry]:

    data = base_flight(seed)

    start = max(
        1,
        len(data) // 2,
    )

    for i in range(
        start,
        len(data),
    ):

        item = data[i]

        data[i] = DroneTelemetry(
            **{
                **asdict(item),
                "gps_speed": 35.0,
                "ground_speed": 8.0,
                "scenario": (
                    "TELEMETRY_MANIPULATION"
                ),
                "expected_attack": (
                    "TELEMETRY_MANIPULATION"
                ),
            }
        )

    return data


def dos_scenario(
    seed: int = 42,
) -> List[DroneTelemetry]:

    data = base_flight(seed)

    start = max(
        1,
        len(data) // 2,
    )

    for i in range(
        start,
        len(data),
    ):

        item = data[i]

        data[i] = DroneTelemetry(
            **{
                **asdict(item),
                "message_rate": 100.0,
                "scenario": "DOS_ANOMALY",
                "expected_attack": "DOS_ANOMALY",
            }
        )

    return data


SCENARIOS: Dict[
    str,
    Callable[
        [int],
        List[DroneTelemetry],
    ],
] = {

    "NORMAL": normal_scenario,

    "GPS_SPOOFING":
        gps_spoofing_scenario,

    "MAVLINK_ANOMALY":
        mavlink_anomaly_scenario,

    "COMMAND_ANOMALY":
        command_anomaly_scenario,

    "TELEMETRY_MANIPULATION":
        telemetry_manipulation_scenario,

    "DOS_ANOMALY":
        dos_scenario,
}


# ============================================================
# 6. FEATURE ENGINE
# ============================================================

class FeatureEngine:

    def __init__(self) -> None:

        self.previous: Optional[
            DroneTelemetry
        ] = None

    def reset(self) -> None:

        self.previous = None

    def extract(
        self,
        current: DroneTelemetry,
    ) -> FeatureVector:

        if self.previous is None:

            result = FeatureVector(

                speed_difference=abs(
                    current.gps_speed
                    - current.ground_speed
                ),

                position_change_m=0.0,

                heading_change_deg=0.0,

                altitude_change_m=0.0,

                gps_quality_score=(
                    self.gps_quality(
                        current
                    )
                ),

                message_rate=(
                    current.message_rate
                ),

                command_rate=0.0,

                sensor_speed_disagreement=abs(
                    current.gps_speed
                    - current.ground_speed
                ),

                sensor_altitude_disagreement=0.0,
            )

            self.previous = current

            return result

        dt = max(
            current.timestamp
            - self.previous.timestamp,
            1e-6,
        )

        command_delta = max(
            0,
            current.command_count
            - self.previous.command_count,
        )

        speed_difference = abs(
            current.gps_speed
            - current.ground_speed
        )

        altitude_change = abs(
            current.altitude
            - self.previous.altitude
        )

        position_change = haversine_m(
            self.previous.latitude,
            self.previous.longitude,
            current.latitude,
            current.longitude,
        )

        heading_change = (
            angle_difference_deg(
                current.heading,
                self.previous.heading,
            )
        )

        result = FeatureVector(

            speed_difference=(
                speed_difference
            ),

            position_change_m=(
                position_change
            ),

            heading_change_deg=(
                heading_change
            ),

            altitude_change_m=(
                altitude_change
            ),

            gps_quality_score=(
                self.gps_quality(
                    current
                )
            ),

            message_rate=(
                current.message_rate
            ),

            command_rate=(
                command_delta / dt
            ),

            sensor_speed_disagreement=(
                speed_difference
            ),

            sensor_altitude_disagreement=(
                altitude_change
            ),
        )

        self.previous = current

        return result

    @staticmethod
    def gps_quality(
        telemetry: DroneTelemetry,
    ) -> float:

        satellite_score = max(
            0.0,
            min(
                1.0,
                (
                    telemetry.satellites
                    - 4
                )
                / 8.0,
            ),
        )

        hdop_score = max(
            0.0,
            min(
                1.0,
                2.0
                / max(
                    telemetry.hdop,
                    0.1,
                ),
            ),
        )

        return (
            satellite_score
            * hdop_score
        )


# ============================================================
# 7. DETECTORS
# ============================================================

class GPSDetector:

    def detect(
        self,
        telemetry: DroneTelemetry,
        features: FeatureVector,
    ) -> List[SecurityAlert]:

        threshold = THRESHOLDS[
            "gps_speed_difference_mps"
        ]

        if (
            features.speed_difference
            > threshold
        ):

            confidence = min(
                0.99,
                0.70
                + (
                    features.speed_difference
                    - threshold
                )
                / 100.0,
            )

            return [
                SecurityAlert(

                    timestamp=(
                        telemetry.timestamp
                    ),

                    attack_type=(
                        "GPS_SPOOFING"
                    ),

                    category="navigation",

                    severity="HIGH",

                    confidence=confidence,

                    source="GPSDetector",

                    evidence={
                        "gps_speed":
                            telemetry.gps_speed,

                        "ground_speed":
                            telemetry.ground_speed,

                        "speed_difference":
                            features.speed_difference,

                        "satellites":
                            telemetry.satellites,

                        "hdop":
                            telemetry.hdop,
                    },
                )
            ]

        return []


class MAVLinkDetector:

    def detect(
        self,
        telemetry: DroneTelemetry,
        features: FeatureVector,
    ) -> List[SecurityAlert]:

        threshold = THRESHOLDS[
            "message_rate_high"
        ]

        if (
            features.message_rate
            > threshold
        ):

            confidence = min(
                0.99,
                0.75
                + (
                    features.message_rate
                    - threshold
                )
                / 200.0,
            )

            return [
                SecurityAlert(

                    timestamp=(
                        telemetry.timestamp
                    ),

                    attack_type=(
                        "MAVLINK_ANOMALY"
                    ),

                    category="communication",

                    severity="HIGH",

                    confidence=confidence,

                    source="MAVLinkDetector",

                    evidence={
                        "message_rate":
                            features.message_rate,

                        "message_type":
                            telemetry.message_type,
                    },
                )
            ]

        return []


class CommandDetector:

    def detect(
        self,
        telemetry: DroneTelemetry,
        features: FeatureVector,
    ) -> List[SecurityAlert]:

        threshold = THRESHOLDS[
            "command_rate_high"
        ]

        if (
            features.command_rate
            > threshold
        ):

            confidence = min(
                0.99,
                0.75
                + (
                    features.command_rate
                    - threshold
                )
                / 100.0,
            )

            return [
                SecurityAlert(

                    timestamp=(
                        telemetry.timestamp
                    ),

                    attack_type=(
                        "COMMAND_ANOMALY"
                    ),

                    category="control",

                    severity="HIGH",

                    confidence=confidence,

                    source="CommandDetector",

                    evidence={
                        "command_rate":
                            features.command_rate,

                        "flight_mode":
                            telemetry.flight_mode,
                    },
                )
            ]

        return []


class TelemetryConsistencyDetector:

    def detect(
        self,
        telemetry: DroneTelemetry,
        features: FeatureVector,
    ) -> List[SecurityAlert]:

        speed_limit = THRESHOLDS[
            "gps_speed_difference_mps"
        ]

        altitude_limit = THRESHOLDS[
            "sensor_altitude_disagreement_m"
        ]

        speed_bad = (
            features.sensor_speed_disagreement
            > speed_limit
        )

        altitude_bad = (
            features.sensor_altitude_disagreement
            > altitude_limit
        )

        if speed_bad or altitude_bad:

            confidence = (
                0.94
                if speed_bad
                and altitude_bad
                else 0.82
            )

            return [
                SecurityAlert(

                    timestamp=(
                        telemetry.timestamp
                    ),

                    attack_type=(
                        "TELEMETRY_MANIPULATION"
                    ),

                    category="telemetry",

                    severity="HIGH",

                    confidence=confidence,

                    source=(
                        "TelemetryConsistencyDetector"
                    ),

                    evidence={
                        "speed_disagreement":
                            features.sensor_speed_disagreement,

                        "altitude_disagreement":
                            features.sensor_altitude_disagreement,
                    },
                )
            ]

        return []


class DOSDetector:

    def detect(
        self,
        telemetry: DroneTelemetry,
        features: FeatureVector,
    ) -> List[SecurityAlert]:

        threshold = THRESHOLDS[
            "message_rate_high"
        ]

        if (
            features.message_rate
            > threshold * 1.5
        ):

            confidence = min(
                0.99,
                0.80
                + (
                    features.message_rate
                    - threshold * 1.5
                )
                / 300.0,
            )

            return [
                SecurityAlert(

                    timestamp=(
                        telemetry.timestamp
                    ),

                    attack_type=(
                        "DOS_ANOMALY"
                    ),

                    category="communication",

                    severity="HIGH",

                    confidence=confidence,

                    source="DOSDetector",

                    evidence={
                        "message_rate":
                            features.message_rate,

                        "configured_threshold":
                            threshold,
                    },
                )
            ]

        return []


DETECTORS = [
    GPSDetector(),
    MAVLinkDetector(),
    CommandDetector(),
    TelemetryConsistencyDetector(),
    DOSDetector(),
]


# ============================================================
# 8. DETECTION ENGINE
# ============================================================

class DetectionEngine:

    def __init__(self) -> None:

        self.feature_engine = (
            FeatureEngine()
        )

    def reset(self) -> None:

        self.feature_engine.reset()

    def process(
        self,
        telemetry: DroneTelemetry,
    ) -> tuple[
        List[SecurityAlert],
        float,
    ]:

        start = time.perf_counter()

        features = (
            self.feature_engine.extract(
                telemetry
            )
        )

        alerts: List[
            SecurityAlert
        ] = []

        for detector in DETECTORS:

            alerts.extend(
                detector.detect(
                    telemetry,
                    features,
                )
            )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        for alert in alerts:

            alert.processing_latency_ms = (
                latency_ms
            )

        return (
            alerts,
            latency_ms,
        )


# ============================================================
# 9. FIRMWARE INTEGRITY
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def verify_firmware(
    firmware_path: Path,
    expected_hash: str,
) -> Dict:

    actual_hash = (
        sha256_file(
            firmware_path
        )
    )

    return {
        "firmware_path":
            str(firmware_path),

        "expected_sha256":
            expected_hash.lower(),

        "actual_sha256":
            actual_hash,

        "integrity_ok":
            actual_hash
            == expected_hash.lower(),
    }


# ============================================================
# 10. HASH-CHAINED EVENT LOGGER
# ============================================================

class HashChainedEventLogger:

    def __init__(
        self,
        path: Path,
    ) -> None:

        self.path = path

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.previous_hash = (
            self._load_previous_hash()
        )

    def _load_previous_hash(
        self,
    ) -> str:

        if not self.path.exists():
            return "0" * 64

        last_record = None

        try:

            with self.path.open(
                "r",
                encoding="utf-8",
            ) as handle:

                for line in handle:

                    if line.strip():

                        last_record = (
                            json.loads(
                                line
                            )
                        )

        except (
            OSError,
            json.JSONDecodeError,
        ):

            return "0" * 64

        if not last_record:
            return "0" * 64

        return str(
            last_record.get(
                "record_hash",
                "0" * 64,
            )
        )

    @staticmethod
    def canonical(
        record: Dict,
    ) -> bytes:

        return json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(
            "utf-8"
        )

    def write(
        self,
        event: Dict,
    ) -> Dict:

        record = dict(event)

        record[
            "previous_hash"
        ] = self.previous_hash

        record_hash = (
            hashlib.sha256(
                self.canonical(
                    record
                )
            ).hexdigest()
        )

        record[
            "record_hash"
        ] = record_hash

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                )
                + "\n"
            )

        self.previous_hash = (
            record_hash
        )

        return record


# ============================================================
# 11. BENCHMARKING
# ============================================================

def pick_expected_detection(
    alerts: Iterable[
        SecurityAlert
    ],
    expected_attack: str,
) -> bool:

    return any(
        alert.attack_type
        == expected_attack
        for alert in alerts
    )


def run_scenario(
    name: str,
    scenario_factory: Callable[
        [int],
        List[DroneTelemetry],
    ],
    seed: int = 42,
) -> ScenarioResult:

    engine = DetectionEngine()

    stream = (
        scenario_factory(seed)
    )

    expected = (
        stream[-1].expected_attack
        if stream
        else "NONE"
    )

    detected_types: set[
        str
    ] = set()

    total_latency = 0.0

    samples = 0

    for telemetry in stream:

        alerts, latency = (
            engine.process(
                telemetry
            )
        )

        total_latency += latency
        samples += 1

        for alert in alerts:

            detected_types.add(
                alert.attack_type
            )

    average_latency = (
        total_latency
        / samples
        if samples
        else 0.0
    )

    if expected == "NONE":

        passed = (
            len(detected_types)
            == 0
        )

        detected_attack = (
            "NONE"
        )

    else:

        passed = (
            expected
            in detected_types
        )

        detected_attack = (
            expected
            if passed
            else (
                sorted(
                    detected_types
                )[0]
                if detected_types
                else "NONE"
            )
        )

    return ScenarioResult(
        scenario=name,
        expected_attack=expected,
        detected_attack=(
            detected_attack
        ),
        passed=passed,
        processing_latency_ms=(
            average_latency
        ),
    )


def benchmark_all() -> Dict:

    results: List[
        ScenarioResult
    ] = []

    for name, factory in SCENARIOS.items():

        results.append(
            run_scenario(
                name,
                factory,
                seed=42,
            )
        )

    attack_cases = [
        result
        for result in results
        if result.expected_attack
        != "NONE"
    ]

    normal_cases = [
        result
        for result in results
        if result.expected_attack
        == "NONE"
    ]

    detection_rate = (
        sum(
            result.passed
            for result in attack_cases
        )
        / len(attack_cases)
        if attack_cases
        else 1.0
    )

    false_positive_rate = (
        sum(
            not result.passed
            for result in normal_cases
        )
        / len(normal_cases)
        if normal_cases
        else 0.0
    )

    average_latency = (
        sum(
            result.processing_latency_ms
            for result in results
        )
        / len(results)
        if results
        else 0.0
    )

    coverage = sum(
        result.passed
        for result in attack_cases
    )

    return {
        "results": [
            asdict(result)
            for result in results
        ],

        "attack_detection_rate":
            detection_rate,

        "false_positive_rate":
            false_positive_rate,

        "average_processing_latency_ms":
            average_latency,

        "attack_vector_coverage":
            coverage,

        "attack_vector_total":
            len(attack_cases),
    }


# ============================================================
# 12. DATASET / REPORT OUTPUT
# ============================================================

def save_jsonl(
    records: Iterable,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for record in records:

            if hasattr(
                record,
                "__dataclass_fields__",
            ):

                data = asdict(
                    record
                )

            else:

                data = record

            handle.write(
                json.dumps(
                    data,
                    separators=(",", ":"),
                )
                + "\n"
            )


def save_report(
    report: Dict,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# 13. FIRMWARE DEMO
# ============================================================

def run_firmware_demo(
    output_dir: Path,
) -> Dict:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trusted = (
        output_dir
        / "firmware_trusted.bin"
    )

    modified = (
        output_dir
        / "firmware_modified.bin"
    )

    trusted.write_bytes(
        b"STAGE1-FIRMWARE\nVERSION=1.0\n"
    )

    modified.write_bytes(
        b"STAGE1-FIRMWARE\n"
        b"VERSION=1.0\n"
        b"CONTROLLED-MODIFICATION\n"
    )

    trusted_hash = (
        sha256_file(
            trusted
        )
    )

    trusted_check = (
        verify_firmware(
            trusted,
            trusted_hash,
        )
    )

    modified_check = (
        verify_firmware(
            modified,
            trusted_hash,
        )
    )

    return {
        "trusted":
            trusted_check,

        "modified":
            modified_check,

        "mismatch_detected":
            not modified_check[
                "integrity_ok"
            ],
    }


# ============================================================
# 14. SELF TESTS
# ============================================================

def self_test() -> None:

    # Feature extraction.
    engine = FeatureEngine()

    normal = normal_scenario()[0]

    first = engine.extract(
        normal
    )

    assert (
        first.position_change_m
        == 0.0
    )

    # GPS detector.
    engine = DetectionEngine()

    gps_data = (
        gps_spoofing_scenario()
    )

    found = False

    for item in gps_data:

        alerts, _ = (
            engine.process(
                item
            )
        )

        if any(
            alert.attack_type
            == "GPS_SPOOFING"
            for alert in alerts
        ):

            found = True
            break

    assert found, (
        "GPS detector failed."
    )

    # MAVLink.
    engine = DetectionEngine()

    mavlink_data = (
        mavlink_anomaly_scenario()
    )

    found = False

    for item in mavlink_data:

        alerts, _ = (
            engine.process(
                item
            )
        )

        if any(
            alert.attack_type
            == "MAVLINK_ANOMALY"
            for alert in alerts
        ):

            found = True
            break

    assert found, (
        "MAVLink detector failed."
    )

    # Command detector.
    engine = DetectionEngine()

    command_data = (
        command_anomaly_scenario()
    )

    found = False

    for item in command_data:

        alerts, _ = (
            engine.process(
                item
            )
        )

        if any(
            alert.attack_type
            == "COMMAND_ANOMALY"
            for alert in alerts
        ):

            found = True
            break

    assert found, (
        "Command detector failed."
    )

    # Telemetry detector.
    engine = DetectionEngine()

    telemetry_data = (
        telemetry_manipulation_scenario()
    )

    found = False

    for item in telemetry_data:

        alerts, _ = (
            engine.process(
                item
            )
        )

        if any(
            alert.attack_type
            == "TELEMETRY_MANIPULATION"
            for alert in alerts
        ):

            found = True
            break

    assert found, (
        "Telemetry detector failed."
    )

    # DoS detector.
    engine = DetectionEngine()

    dos_data = (
        dos_scenario()
    )

    found = False

    for item in dos_data:

        alerts, _ = (
            engine.process(
                item
            )
        )

        if any(
            alert.attack_type
            == "DOS_ANOMALY"
            for alert in alerts
        ):

            found = True
            break

    assert found, (
        "DoS detector failed."
    )

    # Firmware.
    with tempfile.TemporaryDirectory() as tmp:

        path = (
            Path(tmp)
            / "firmware.bin"
        )

        path.write_bytes(
            b"TEST-FIRMWARE"
        )

        good_hash = (
            sha256_file(
                path
            )
        )

        assert verify_firmware(
            path,
            good_hash,
        )["integrity_ok"]

        assert not verify_firmware(
            path,
            "0" * 64,
        )["integrity_ok"]

    print(
        "SELF-TEST: PASS"
    )


# ============================================================
# 15. COMPLETE STAGE 1 DEMO
# ============================================================

def run_stage1_demo(
    duration: float = 5.0,
) -> Dict:

    root = Path(
        "stage1_output"
    )

    dataset_dir = (
        root / "dataset"
    )

    log_dir = (
        root / "logs"
    )

    report_dir = (
        root / "reports"
    )

    scenario_dir = (
        dataset_dir
        / "scenarios"
    )

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Generate requested normal-flight dataset
    normal_data = NormalFlightSimulator(
        duration_s=duration,
        sample_period_s=0.1,
        seed=42,
    ).generate()

    save_jsonl(
        normal_data,
        dataset_dir
        / "normal_flight.jsonl",
    )

    # Generate the controlled attack/test datasets
    for name, factory in SCENARIOS.items():

        data = factory(42)

        save_jsonl(
            data,
            scenario_dir
            / f"{name.lower()}.jsonl",
        )

    # Benchmark
    benchmark = benchmark_all()

    # Firmware integrity
    firmware = run_firmware_demo(
        scenario_dir
    )

    # Event logging
    logger = (
        HashChainedEventLogger(
            log_dir
            / "security_events.jsonl"
        )
    )

    for result in benchmark[
        "results"
    ]:

        logger.write(
            {
                "event_type":
                    "stage1_benchmark_result",

                "scenario":
                    result["scenario"],

                "expected_attack":
                    result["expected_attack"],

                "detected_attack":
                    result["detected_attack"],

                "passed":
                    result["passed"],

                "processing_latency_ms":
                    result[
                        "processing_latency_ms"
                    ],
            }
        )

    logger.write(
        {
            "event_type":
                "firmware_integrity_test",

            "trusted_file_ok":
                firmware[
                    "trusted"
                ][
                    "integrity_ok"
                ],

            "modified_file_mismatch_detected":
                firmware[
                    "mismatch_detected"
                ],
        }
    )

    report = {
        "project":
            "Drone IDS Stage 1 Proof of Concept",

        "scope":
            "Controlled Python simulation",

        "benchmark":
            benchmark,

        "firmware_integrity":
            firmware,

        "note":
            (
                "Numerical thresholds are proposed "
                "Stage 1 PoC engineering values, not "
                "official competition thresholds."
            ),
    }

    save_report(
        report,
        report_dir
        / "stage1_benchmark.json",
    )

    # Console output
    print()
    print("=" * 78)
    print(
        "                 DRONE IDS - STAGE 1 PoC"
    )
    print("=" * 78)

    print()
    print(
        "SCENARIO RESULTS"
    )
    print("-" * 78)

    for result in benchmark[
        "results"
    ]:

        print(
            f"{result['scenario']:30s}"
            f" | "
            f"{'PASS' if result['passed'] else 'FAIL':4s}"
            f" | expected="
            f"{result['expected_attack']:24s}"
            f" | detected="
            f"{result['detected_attack']:24s}"
            f" | "
            f"{result['processing_latency_ms']:.4f} ms"
        )

    print("-" * 78)

    print(
        "Attack detection rate : "
        f"{benchmark['attack_detection_rate'] * 100:.2f}%"
    )

    print(
        "False positive rate   : "
        f"{benchmark['false_positive_rate'] * 100:.2f}%"
    )

    print(
        "Average processing    : "
        f"{benchmark['average_processing_latency_ms']:.4f} ms"
    )

    print(
        "Attack vectors        : "
        f"{benchmark['attack_vector_coverage']}/"
        f"{benchmark['attack_vector_total']}"
    )

    print()
    print(
        "FIRMWARE INTEGRITY"
    )
    print("-" * 78)

    print(
        "Trusted firmware:"
        f" {'PASS' if firmware['trusted']['integrity_ok'] else 'FAIL'}"
    )

    print(
        "Modified firmware:"
        f" {'MISMATCH DETECTED' if firmware['mismatch_detected'] else 'NOT DETECTED'}"
    )

    print()
    print(
        "OUTPUT FILES"
    )
    print("-" * 78)

    print(
        f"Dataset : {dataset_dir}"
    )

    print(
        f"Logs    : {log_dir}"
    )

    print(
        f"Reports : {report_dir}"
    )

    print("=" * 78)

    return report


# ============================================================
# 16. CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "PUSHPAK Objective 2 "
            "Stage 1 Drone IDS PoC"
        )
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help=(
            "Duration of the normal-flight "
            "dataset in seconds."
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal tests and exit.",
    )

    return parser


def main() -> int:

    parser = build_parser()

    args = parser.parse_args()

    if args.self_test:

        self_test()

        return 0

    self_test()

    run_stage1_demo(
        duration=args.duration
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
