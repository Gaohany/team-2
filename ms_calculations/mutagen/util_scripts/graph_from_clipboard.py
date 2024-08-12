import json
from enum import IntEnum
from pathlib import Path

class GTSRBLabelEnum(IntEnum):
    @classmethod
    def _missing_(cls, key):
        return cls.UNKNOWN

    MAX_SPEED_20 = 0
    MAX_SPEED_30 = 1
    MAX_SPEED_50 = 2
    MAX_SPEED_60 = 3
    MAX_SPEED_70 = 4
    MAX_SPEED_80 = 5
    NO_MAX_SPEED_80 = 6
    MAX_SPEED_100 = 7
    MAX_SPEED_120 = 8
    NO_OVERTAKING = 9
    NO_OVERTAKING_FOR_TRUCKS = 10
    PRIORITY_IN_TRAFFIC_NEXT_CROSSING = 11
    PRIORITY_IN_TRAFFIC_ROAD = 12
    YIELD = 13
    STOP = 14
    NO_VEHICLES_PERMITTED_CIRCLE = 15
    TRUCK_DRIVING_LEFT = 16
    NO_VEHICLES_PERMITTED_WHITE_BOX = 17
    WARNING = 18
    DANGEROUS_CURVE_TO_LEFT = 19
    DANGEROUS_CURVE_TO_RIGHT = 20
    DANGEROUS_DOUBLE_CURVE = 21
    BUMPY_ROAD = 22
    SLIPPERY_ROAD = 23
    ROAD_NARROWS_FROM_RIGHT = 24
    CONSTRUCTION_SITE = 25
    TRAFFIC_LIGHT_AHEAD = 26
    PEDESTRIAN = 27
    CHILDREN_CROSSING = 28
    BICYCLE_AHEAD = 29
    SNOWY_ROAD = 30
    DEER_CROSSING = 31
    NO_MAX_SPEED = 32
    TURN_RIGHT = 33
    TURN_LEFT = 34
    GO_STRAIGHT = 35
    GO_STRAIGHT_OR_RIGHT = 36
    GO_STRAIGHT_OR_LEFT = 37
    DRIVE_RIGHT = 38
    DRIVE_LEFT = 39
    ROUNDABOUT = 40
    PASSING_ALLOWED = 41
    PASSING_ALLOWED_FOR_TRUCKS = 42
    UNKNOWN = -1


red_circle_keys = {
    GTSRBLabelEnum.MAX_SPEED_20,
    GTSRBLabelEnum.MAX_SPEED_30,
    GTSRBLabelEnum.MAX_SPEED_50,
    GTSRBLabelEnum.MAX_SPEED_60,
    GTSRBLabelEnum.MAX_SPEED_70,
    GTSRBLabelEnum.MAX_SPEED_80,
    GTSRBLabelEnum.MAX_SPEED_100,
    GTSRBLabelEnum.MAX_SPEED_120,
    GTSRBLabelEnum.NO_VEHICLES_PERMITTED_CIRCLE,
    GTSRBLabelEnum.NO_OVERTAKING,
    GTSRBLabelEnum.NO_OVERTAKING_FOR_TRUCKS
}

blue = {
    GTSRBLabelEnum.TURN_RIGHT,
    GTSRBLabelEnum.TURN_LEFT,
    GTSRBLabelEnum.GO_STRAIGHT,
    GTSRBLabelEnum.GO_STRAIGHT_OR_RIGHT,
    GTSRBLabelEnum.GO_STRAIGHT_OR_LEFT,
    GTSRBLabelEnum.DRIVE_RIGHT,
    GTSRBLabelEnum.DRIVE_LEFT,
    GTSRBLabelEnum.ROUNDABOUT,
}

lines = [json.loads(li) for li in Path('output.json').read_text().splitlines()]

output = {
  "width": 1920,
  "height": 1280,
  "config": {
      "legend": {
          "symbolDirection": "horizontal",
          "labelFontSize": 16
      },
      "axis": {
          "labelFontSize": 16,
          "titleFontSize": 16
      },
      "title": {
          "fontSize": 18
      },
      "text": {
          "fontSize": 18
      }
  },
     "params": [
        {
          "name": "hover",
          "select": {"type": "point", "on": "pointerover", "clear": "pointerout"}
        }
      ],
    "data": {"values": []},
    "mark": {
    "type": "bar",
    "cursor": "pointer",
    "tooltip": True
  },
  "encoding": {
    "x": {
      "field": "class"
    },
    "opacity": {
          "condition": {"test": {"param": "hover", "empty": False}, "value": 0.75},
          "value": 1.0
        },
    "y": {
      "field": "d",
      "type": "quantitative",
      "axis": {"title": "distance"}
    },
    "xOffset": {
      "field": "group"
    },
    "color": {
      "field": "group",
      "legend": {
        "orient": "none",
        "legendY": 630,
        "title": "",
        "labelFontSize": 18
      },
      "scale": {
        "scheme": "category10"
      }
    }
  }
}

# k = 'max_classes_distance'
k = 'avg_classes_distance'

output["title"] = "Average distance per class (blue circle signs)"

for dataset in lines:
    if dataset['dataset'] not in {'test', 'nc', 'nbc', 'kmnc', 'meta_blue_circle'}:
        continue
    if dataset['dataset'] in {'nc', 'nbc', 'kmnc'}:
        dataset['dataset'] = f"fz_{dataset['dataset']}"

    # for i, e in enumerate(dataset['total_obj_counter']):
    for i, e in enumerate(dataset[k]):
        if i not in blue:
            continue
        output['data']['values'].append({"class": i, "group": dataset['dataset'], "d": e})

with Path("graph.json").open("w") as fp:
    json.dump(output, fp, indent=2)
