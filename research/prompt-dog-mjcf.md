Write a MuJoCo (3.x) MJCF model of a medium adult dog, a Labrador Retriever (withers height 0.57 m, about 30 kg), for training physics-based controllers with MJX and Brax PPO on one RTX 3090.

Requirements:
- MJX-compatible geometry only (capsules, spheres, boxes, a plane floor); keep contacts few: feet spheres plus trunk, head and elbows/hocks for lying down.
- Anatomy: trunk in 2 segments joined by a spine joint (pitch and yaw), neck (pitch, yaw), head (pitch), tail as 2 passive spring-damper hinges.
- Forelimbs: scapula-shoulder (pitch, small abduction), elbow (pitch), carpus (pitch). Hindlimbs: hip (pitch, small abduction), stifle (pitch), hock (pitch). Dog proportions and joint ranges that allow standing, walking, trotting, galloping, sitting, lying in a sphinx down, lying on the side, rolling, and getting up.
- Position actuators with kp and kv chosen for a 30 kg dog; realistic masses per segment.
- A "stand" keyframe, and if you can, "sit" and "down" keyframes.

Use these literature values where they help (from an earlier query):
[
  {
    "param": "withers_height",
    "unit": "m",
    "lab_typical": 0.57,
    "lab_range": [0.53, 0.62],
    "mal_typical": 0.61,
    "mal_range": [0.56, 0.66],
    "source": "Fischer 2011; Voss 2010",
    "confidence": "high",
    "note": "Pools adult males and females; working-line Labs sit near upper range"
  },
  {
    "param": "body_length_to_height_ratio",
    "unit": "ratio",
    "lab_typical": 1.08,
    "lab_range": [1.02, 1.15],
    "mal_typical": 1.01,
    "mal_range": [0.96, 1.06],
    "source": "Fischer 2011",
    "confidence": "high",
    "note": "Scapulohumeral joint to ischial tuberosity divided by withers height; Malinois is square, Lab slightly rectangular"
  },
  {
    "param": "body_mass",
    "unit": "kg",
    "lab_typical": 31.0,
    "lab_range": [25.0, 38.0],
    "mal_typical": 27.5,
    "mal_range": [20.0, 34.0],
    "source": "Clements 2005; Fischer 2011",
    "confidence": "high",
    "note": "Lean-to-moderate pet/working condition; excludes obese pets"
  },
  {
    "param": "preferred_trot_speed",
    "unit": "m/s",
    "lab_typical": 2.15,
    "lab_range": [1.8, 2.6],
    "mal_typical": 2.5,
    "mal_range": [2.0, 3.1],
    "source": "Bertram 2000; Agostinho 2011; Fischer 2011",
    "confidence": "high",
    "note": "Energetically optimal steady overground trot; equals stride_frequency * stride_length"
  },
  {
    "param": "trot_stride_frequency",
    "unit": "Hz",
    "lab_typical": 1.95,
    "lab_range": [1.75, 2.2],
    "mal_typical": 2.02,
    "mal_range": [1.8, 2.3],
    "source": "Bertram 2000; Maes 2008; Fischer 2011",
    "confidence": "high",
    "note": "Full cycles of one limb per second; varies weakly with speed compared to stride length"
  },
  {
    "param": "trot_stride_length",
    "unit": "m",
    "lab_typical": 1.1,
    "lab_range": [0.95, 1.3],
    "mal_typical": 1.24,
    "mal_range": [1.05, 1.5],
    "source": "Bertram 2000; Fischer 2011",
    "confidence": "high",
    "note": "At preferred trot speed; scales strongly with speed and leg length"
  },
  {
    "param": "trot_duty_factor",
    "unit": "fraction",
    "lab_typical": 0.44,
    "lab_range": [0.38, 0.5],
    "mal_typical": 0.41,
    "mal_range": [0.35, 0.47],
    "source": "Bertram 2000; Maes 2008",
    "confidence": "high",
    "note": "Forelimb duty factor is typically 0.02-0.04 higher than hindlimb; <0.5 implies aerial suspension phases"
  },
  {
    "param": "walk_duty_factor",
    "unit": "fraction",
    "lab_typical": 0.64,
    "lab_range": [0.58, 0.7],
    "mal_typical": 0.62,
    "mal_range": [0.56, 0.68],
    "source": "Jayes 1978; Bertram 2000",
    "confidence": "high",
    "note": "Measured at moderate symmetrical lateral-sequence walk"
  },
  {
    "param": "walk_to_trot_transition_speed",
    "unit": "m/s",
    "lab_typical": 1.55,
    "lab_range": [1.35, 1.75],
    "mal_typical": 1.65,
    "mal_range": [1.45, 1.85],
    "source": "Maes 2008; Abdelhadi 2013",
    "confidence": "medium",
    "note": "Occurs near Froude number ~0.45-0.55; Malinois studied directly in Maes 2008"
  },
  {
    "param": "trot_diagonal_dissociation",
    "unit": "fraction",
    "lab_typical": -0.01,
    "lab_range": [-0.04, 0.03],
    "mal_typical": -0.01,
    "mal_range": [-0.04, 0.03],
    "source": "Bertram 2000; Fischer 2011",
    "confidence": "medium",
    "note": "Near-simultaneous; diagonal forefoot commonly touches down 0-3% of stride before hindfoot"
  },
  {
    "param": "trot_vertical_trunk_oscillation",
    "unit": "m",
    "lab_typical": 0.038,
    "lab_range": [0.025, 0.055],
    "mal_typical": 0.042,
    "mal_range": [0.028, 0.06],
    "source": "Wachs 2016; Fischer 2011",
    "confidence": "medium",
    "note": "Peak-to-peak vertical excursion at withers, oscillating at 2x stride frequency"
  },
  {
    "param": "trot_neck_angle_above_horizontal",
    "unit": "deg",
    "lab_typical": 28,
    "lab_range": [12, 45],
    "mal_typical": 42,
    "mal_range": [22, 58],
    "source": "estimate",
    "confidence": "low",
    "note": "Cervical spine chord above horizontal; highly context-dependent (scenting vs alert)"
  },
  {
    "param": "scapula_rotation_range_trot",
    "unit": "deg",
    "lab_typical": 29,
    "lab_range": [23, 35],
    "mal_typical": 32,
    "mal_range": [26, 38],
    "source": "Fischer 1999; Fischer 2011",
    "confidence": "high",
    "note": "Sagittal plane angular excursion of scapular spine; primary driver of forelimb stride length"
  },
  {
    "param": "carpal_flexion_max_in_swing",
    "unit": "deg",
    "lab_typical": 75,
    "lab_range": [60, 92],
    "mal_typical": 78,
    "mal_range": [62, 95],
    "source": "DeCamp 1993; Agostinho 2011",
    "confidence": "medium",
    "note": "Degrees flexed palmarly from a 180-deg straight antebrachium-metacarpus axis during mid-swing"
  },
  {
    "param": "tarsal_hock_angle_min_trot",
    "unit": "deg",
    "lab_typical": 108,
    "lab_range": [96, 120],
    "mal_typical": 103,
    "mal_range": [92, 115],
    "source": "Clements 2005; Agostinho 2011; Fischer 2011",
    "confidence": "high",
    "note": "Minimum cranial included angle during swing-phase flexion"
  },
  {
    "param": "tarsal_hock_angle_max_trot",
    "unit": "deg",
    "lab_typical": 160,
    "lab_range": [148, 170],
    "mal_typical": 164,
    "mal_range": [152, 173],
    "source": "Clements 2005; Agostinho 2011; Fischer 2011",
    "confidence": "high",
    "note": "Maximum cranial included angle at late-stance push-off"
  },
  {
    "param": "standing_shoulder_angle",
    "unit": "deg",
    "lab_typical": 112,
    "lab_range": [102, 122],
    "mal_typical": 110,
    "mal_range": [100, 120],
    "source": "Jarvis 2013; Fischer 2011",
    "confidence": "high",
    "note": "Caudal scapulohumeral angle; radiographic/kinematic values are ~110 deg despite 90-deg show standards"
  },
  {
    "param": "standing_elbow_angle",
    "unit": "deg",
    "lab_typical": 138,
    "lab_range": [128, 148],
    "mal_typical": 140,
    "mal_range": [130, 150],
    "source": "Jaegger 2002; Jarvis 2013",
    "confidence": "high",
    "note": "Cranial humeroradioulnar included angle"
  },
  {
    "param": "standing_carpus_angle",
    "unit": "deg",
    "lab_typical": 195,
    "lab_range": [188, 204],
    "mal_typical": 194,
    "mal_range": [188, 202],
    "source": "Jaegger 2002; Jarvis 2013",
    "confidence": "high",
    "note": "Palmar aspect angle where 180 is straight and >180 indicates normal passive dorsiflexion/hyperextension"
  },
  {
    "param": "standing_hip_angle",
    "unit": "deg",
    "lab_typical": 108,
    "lab_range": [98, 118],
    "mal_typical": 106,
    "mal_range": [96, 116],
    "source": "Jaegger 2002; Agostinho 2011",
    "confidence": "medium",
    "note": "Ventral pelvis-to-femur included angle"
  },
  {
    "param": "standing_stifle_angle",
    "unit": "deg",
    "lab_typical": 138,
    "lab_range": [128, 148],
    "mal_typical": 136,
    "mal_range": [126, 146],
    "source": "Jaegger 2002; Agostinho 2011",
    "confidence": "high",
    "note": "Caudal femorotibial included angle; Jaegger 2002 measured Labradors specifically"
  },
  {
    "param": "standing_hock_angle",
    "unit": "deg",
    "lab_typical": 142,
    "lab_range": [132, 152],
    "mal_typical": 140,
    "mal_range": [130, 150],
    "source": "Jaegger 2002; Jarvis 2013",
    "confidence": "high",
    "note": "Cranial tibiotarsal included angle"
  },
  {
    "param": "segment_length_scapula",
    "unit": "withers_height",
    "lab_typical": 0.28,
    "lab_range": [0.25, 0.31],
    "mal_typical": 0.28,
    "mal_range": [0.25, 0.31],
    "source": "Fischer 2011",
    "confidence": "high",
    "note": "Dorsal scapular border to shoulder joint center; remarkably conserved across breeds"
  },
  {
    "param": "segment_length_humerus",
    "unit": "withers_height",
    "lab_typical": 0.29,
    "lab_range": [0.26, 0.32],
    "mal_typical": 0.29,
    "mal_range": [0.26, 0.32],
    "source": "Fischer 2011",
    "confidence": "high",
    "note": "Shoulder joint center to elbow joint center"
  },
  {
    "param": "segment_length_radius_ulna",
    "unit": "withers_height",
    "lab_typical": 0.3,
    "lab_range": [0.27, 0.33],
    "mal_typical": 0.32,
    "mal_range": [0.29, 0.35],
    "source": "Fischer 2011",
    "confidence": "high",
    "note": "Elbow to antebrachiocarpal joint; slightly longer in cursorial/herding Malinois"
  },
  {
    "param": "segment_length_metacarpus_plus_paw",
    "unit": "withers_height",
    "lab_typical": 0.19,
    "lab_range": [0.16, 0.22],
    "mal_typical": 0.19,
    "mal_range": [0.16, 0.22],
    "source": "Fischer 2011; estimate",
    "confidence": "medium",
    "note": "Carpus to distal digit tip along segment axis (~0.12 metacarpus + ~0.07 digit); vertical standing height is ~0.16"
  },
  {
    "param": "segment_length_femur",
    "unit": "withers_height",
    "lab_typical": 0.31,
    "lab_range": [0.28, 0.34],
    "mal_typical": 0.31,
    "mal_range": [0.28, 0.34],
    "source": "F

Then give the training recipe, concretely:
1. Observations and actions (action = joint position targets around the stand pose).
2. Rewards for velocity tracking with a chosen gait (walk, trot, gallop) using footfall-phase targets, plus posture rewards for sit/down/stand-up skills, plus fall recovery.
3. Terminations, domain randomization, and the PPO settings and step budget you expect on one RTX 3090 with MJX.
4. What usually goes wrong with quadruped RL that looks unnatural, and how others fixed it (motion priors such as AMP, reference-motion imitation, gait shaping).

Output: the full MJCF in one ```xml block, then the recipe. Be precise; this will be compiled and trained, not read for ideas.