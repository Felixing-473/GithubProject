import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import gym
import numpy as np
from gym import spaces


@dataclass
class Point:
    id: str
    x: float
    y: float
    zone: str
    init_normal: int
    init_medium: int
    init_severe: int


@dataclass
class City:
    id: str
    severe_beds_max: int
    medium_beds_max: int
    normal_capacity_max: int
    current_severe: int = 0
    current_medium: int = 0
    current_normal: int = 0

    @property
    def severe_remaining(self):
        return max(0, self.severe_beds_max - self.current_severe)

    @property
    def medium_remaining(self):
        return max(0, self.medium_beds_max - self.current_medium)

    @property
    def normal_remaining(self):
        return max(0, self.normal_capacity_max - self.current_normal)


@dataclass
class Hub:
    id: str
    x: float
    y: float
    current_normal: int = 0
    current_medium: int = 0
    current_severe: int = 0
    food: float = 0.0
    med: float = 0.0
    vehicle_level: int = 1
    capacity_persons: int = 5000
    food_storage_max: float = 8000.0
    med_storage_max: float = 2000.0
    is_available: bool = True
    stage_state: str = "NORMAL"  # NORMAL, WARNING, FLOODED
    base_travel_time_to_cities: Dict[str, int] = field(default_factory=dict)

    @property
    def total_people(self):
        return self.current_normal + self.current_medium + self.current_severe


@dataclass
class TransportJob:
    from_hub: str
    to_city: str
    alloc_severe: int
    alloc_medium: int
    alloc_normal: int
    depart_step: int
    arrive_step: int


class FloodEvacuationEnv(gym.Env):
    """A simplified implementation of the Flood Evacuation environment.

    Observation vector (fixed length):
      - For each hub B1,B2,B3: [severe, medium, normal, food, med, is_available]
      - current_step, time_to_flood_trigger
      - For each city S1,S2,S3: [severe_beds_remain, medium_beds_remain, normal_capacity_remain]
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config_path: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.seed(seed)
        if config_path is None:
            config_path = "configs/default.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # constants
        self.STEP_HOURS = 1
        self.MAX_STEPS = 72

        # consumption
        self.food_per_person_per_hour = self.config.get("food_per_person_per_hour", 1.0)
        self.med_per_patient_per_hour = self.config.get("med_per_patient_per_hour", 0.2)

        # vehicle depart rates per level
        self.vehicle_depart_rate = {1: 50, 2: 100, 3: 200}

        # map containers
        self.hubs: Dict[str, Hub] = {}
        self.cities: Dict[str, City] = {}
        self.points: List[Point] = []

        self._load_static()

        # transport queue
        self.outgoing_transports: List[TransportJob] = []

        # flood target and timing
        self.flood_target = random.choice(list(self.hubs.keys()))
        self.flood_trigger_step = 30

        # observation/action spaces
        # action: pre_action (Discrete 19683) and step_action (3 hubs -> 3 choices each)
        self.action_space = spaces.Dict({
            "pre_action": spaces.Discrete(19683),
            "step_action": spaces.MultiDiscrete([3, 3, 3]),
        })

        # observation: fixed-length vector
        obs_len = 29
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_len,), dtype=np.float32)

        # runtime state
        self.step_count = 0
        self.seed_val = seed
        self.last_pre_disaster_action = None
        # tracking transfers for reward calculation
        self.last_transferred_severe = 0
        self.last_transferred_medium = 0
        self.last_transferred_normal = 0

    def _load_static(self):
        # load hubs
        for h in self.config["hubs"]:
            hub = Hub(
                id=h["id"],
                x=h.get("x", 0.0),
                y=h.get("y", 0.0),
                current_normal=h.get("init_normal", 0),
                current_medium=h.get("init_medium", 0),
                current_severe=h.get("init_severe", 0),
                food=h.get("food", 0.0),
                med=h.get("med", 0.0),
                vehicle_level=h.get("vehicle_level", 1),
                capacity_persons=h.get("capacity_persons", 5000),
                food_storage_max=h.get("food_storage_max", 8000.0),
                med_storage_max=h.get("med_storage_max", 2000.0),
            )
            hub.base_travel_time_to_cities = h.get("base_travel_time_to_cities", {"S1": 4, "S2": 6, "S3": 8})
            self.hubs[hub.id] = hub

        for s in self.config["cities"]:
            city = City(
                id=s["id"],
                severe_beds_max=s.get("severe_beds_max", 100),
                medium_beds_max=s.get("medium_beds_max", 200),
                normal_capacity_max=s.get("normal_capacity_max", 1000),
            )
            self.cities[city.id] = city

        for p in self.config.get("points", []):
            pt = Point(
                id=p["id"],
                x=p.get("x", 0.0),
                y=p.get("y", 0.0),
                zone=p.get("zone", "B1"),
                init_normal=p.get("init_normal", 0),
                init_medium=p.get("init_medium", 0),
                init_severe=p.get("init_severe", 0),
            )
            self.points.append(pt)

    def seed(self, seed: Optional[int] = None):
        self.seed_val = seed
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, pre_disaster_action: Optional[int] = None):
        # reset runtime state
        # reload static to clear runtime mutations
        self.hubs.clear()
        self.cities.clear()
        self.points.clear()
        self._load_static()
        self.outgoing_transports = []
        self.step_count = 0
        self.flood_target = random.choice(list(self.hubs.keys()))
        self.last_pre_disaster_action = pre_disaster_action

        # reset transfer trackers
        self.last_transferred_severe = 0
        self.last_transferred_medium = 0
        self.last_transferred_normal = 0

        # apply pre-disaster action if provided
        if pre_disaster_action is not None:
            self._apply_pre_disaster_action(pre_disaster_action)

        obs = self._pack_observation()
        return obs

    def _apply_pre_disaster_action(self, action_int: int):
        # decode base-3 for each hub and 3 resources per hub
        # action_int in [0, 19682]
        x = action_int
        for hub_id in ["B1", "B2", "B3"]:
            # each hub has 3 resources: food, med, vehicle each in 1..3
            res = []
            for _ in range(3):
                res.append((x % 3) + 1)
                x //= 3
            food_level, med_level, vehicle_level = res
            hub = self.hubs[hub_id]
            hub.food = {1: 2000.0, 2: 4000.0, 3: 8000.0}[food_level]
            hub.med = {1: 500.0, 2: 1000.0, 3: 2000.0}[med_level]
            hub.vehicle_level = vehicle_level

    def step(self, action):
        # action is a dict with 'step_action'
        step_action = action.get("step_action") if isinstance(action, dict) else action
        self.step_count += 1

        # 1. material consumption
        self._material_consumption()

        # 2. advance transports arrival
        self._progress_transports()

        # 3. flood state advance
        self._advance_flood_state()

        # 4. pack observation before applying action (agent sees current)
        obs = self._pack_observation()

        # 5. apply step_action: for each hub if available, dispatch to chosen city
        if step_action is None:
            step_action = [0, 0, 0]
        # ensure length 3
        if len(step_action) != 3:
            step_action = list(step_action)[:3] + [0] * max(0, 3 - len(step_action))

        hub_ids = ["B1", "B2", "B3"]
        for i, hub_id in enumerate(hub_ids):
            hub = self.hubs[hub_id]
            if not hub.is_available or hub.total_people == 0:
                continue
            target_city_idx = int(step_action[i])
            target_city_id = ["S1", "S2", "S3"][target_city_idx]
            self._dispatch_from_hub_to_city(hub, self.cities[target_city_id])

        # compute reward
        reward = self._compute_reward()

        done = self.step_count >= self.MAX_STEPS or self._all_people_evacuated()

        info = {
            "step": self.step_count,
            "flood_target": self.flood_target,
        }

        return self._pack_observation(), float(reward), bool(done), info

    def _material_consumption(self):
        for hub in self.hubs.values():
            food_delta = hub.total_people * self.food_per_person_per_hour
            med_delta = (hub.current_medium + hub.current_severe) * self.med_per_patient_per_hour
            hub.food = max(0.0, hub.food - food_delta)
            hub.med = max(0.0, hub.med - med_delta)

    def _progress_transports(self):
        # check arrivals and record deltas
        arrived = [job for job in self.outgoing_transports if job.arrive_step <= self.step_count]
        severe_added = 0
        medium_added = 0
        normal_added = 0
        for job in arrived:
            city = self.cities[job.to_city]
            # allocate to beds with simple truncation
            severe_accept = min(job.alloc_severe, city.severe_remaining)
            city.current_severe += severe_accept
            medium_accept = min(job.alloc_medium, city.medium_remaining)
            city.current_medium += medium_accept
            normal_accept = min(job.alloc_normal, city.normal_remaining)
            city.current_normal += normal_accept
            severe_added += severe_accept
            medium_added += medium_accept
            normal_added += normal_accept
        # record last step transferred counts for reward computation
        self.last_transferred_severe = severe_added
        self.last_transferred_medium = medium_added
        self.last_transferred_normal = normal_added
        # remove arrived
        self.outgoing_transports = [job for job in self.outgoing_transports if job.arrive_step > self.step_count]

    def _advance_flood_state(self):
        # apply warning and flooded effects
        if self.step_count in (27, 28, 29):
            # warning: halve processing efficiency -> modeled implicitly by increasing travel time
            if self.flood_target in self.hubs:
                self.hubs[self.flood_target].stage_state = "WARNING"
                self.hubs[self.flood_target].is_available = True
        if self.step_count >= 30:
            if self.flood_target in self.hubs:
                hub = self.hubs[self.flood_target]
                hub.stage_state = "FLOODED"
                hub.is_available = False
                # auto-distribute existing people to other hubs immediately
                others = [h for k, h in self.hubs.items() if k != self.flood_target and h.is_available]
                if others:
                    for cls in ("current_severe", "current_medium", "current_normal"):
                        total = getattr(hub, cls)
                        # naive split by distance (equal split for this simplified impl)
                        per = total // len(others)
                        remainder = total - per * len(others)
                        for j, other in enumerate(others):
                            add = per + (1 if j < remainder else 0)
                            setattr(other, cls, getattr(other, cls) + add)
                        setattr(hub, cls, 0)

    def _dispatch_from_hub_to_city(self, hub: Hub, city: City):
        # Determine how many can depart based on vehicle rate and available people and city capacity
        depart_rate = self.vehicle_depart_rate.get(hub.vehicle_level, 50)
        # prioritize severe > medium > normal
        to_send_severe = min(hub.current_severe, depart_rate)
        remaining_rate = depart_rate - to_send_severe
        to_send_medium = min(hub.current_medium, remaining_rate)
        remaining_rate -= to_send_medium
        to_send_normal = min(hub.current_normal, remaining_rate)

        # consider city remaining capacity; we still create job and arrival will be truncated on arrival
        travel_base = hub.base_travel_time_to_cities.get(city.id, 4)
        # basic congestion factor
        congestion = min(2.0, hub.total_people / max(1, hub.capacity_persons))
        travel_time = math.ceil(travel_base * (1.0 + congestion))

        job = TransportJob(
            from_hub=hub.id,
            to_city=city.id,
            alloc_severe=to_send_severe,
            alloc_medium=to_send_medium,
            alloc_normal=to_send_normal,
            depart_step=self.step_count,
            arrive_step=self.step_count + travel_time,
        )
        # subtract people from hub immediately (they are in transit)
        hub.current_severe -= to_send_severe
        hub.current_medium -= to_send_medium
        hub.current_normal -= to_send_normal
        self.outgoing_transports.append(job)

    def _compute_reward(self):
        # simplified reward per spec weights
        w_severe = 100.0
        w_medium = 20.0
        p_severe_unplaced = 200.0
        p_overload = 500.0
        p_resource_empty = 50.0

        reward = 0.0

        # reward for newly settled severe/medium in cities (use recorded deltas from arrivals)
        reward += w_severe * getattr(self, "last_transferred_severe", 0)
        reward += w_medium * getattr(self, "last_transferred_medium", 0)

        # penalty for severe still in hubs
        severe_unplaced = sum(h.current_severe for h in self.hubs.values())
        reward -= p_severe_unplaced * severe_unplaced

        # penalty for resource shortages
        shortage_count = sum(1 for h in self.hubs.values() if h.food <= 0.0 or h.med <= 0.0)
        reward -= p_resource_empty * shortage_count

        # overload penalty if any city has negative remaining (shouldn't happen since we truncate on arrival)
        overload = 0
        for c in self.cities.values():
            over = max(0, (c.current_severe - c.severe_beds_max) + (c.current_medium - c.medium_beds_max) + (c.current_normal - c.normal_capacity_max))
            overload += over
        reward -= p_overload * overload

        # small time penalty
        total_waiting = sum(h.total_people for h in self.hubs.values())
        reward -= 0.01 * total_waiting

        return reward

    def _pack_observation(self):
        vec: List[float] = []
        for hub_id in ["B1", "B2", "B3"]:
            h = self.hubs[hub_id]
            vec.extend([
                float(h.current_severe),
                float(h.current_medium),
                float(h.current_normal),
                float(h.food),
                float(h.med),
                1.0 if h.is_available else 0.0,
            ])

        vec.append(float(self.step_count))
        time_to_flood = max(0, self.flood_trigger_step - self.step_count)
        vec.append(float(time_to_flood))

        for city_id in ["S1", "S2", "S3"]:
            c = self.cities[city_id]
            vec.extend([
                float(c.severe_remaining),
                float(c.medium_remaining),
                float(c.normal_remaining),
            ])

        return np.array(vec, dtype=np.float32)

    def _all_people_evacuated(self):
        return all(h.total_people == 0 for h in self.hubs.values())

    def render(self, mode="human"):
        print(f"Step {self.step_count}")
        for hid, h in self.hubs.items():
            print(f"{hid}: severe={h.current_severe} medium={h.current_medium} normal={h.current_normal} food={h.food:.1f} med={h.med:.1f} avail={h.is_available}")
        for cid, c in self.cities.items():
            print(f"{cid}: severe={c.current_severe} medium={c.current_medium} normal={c.current_normal}")

