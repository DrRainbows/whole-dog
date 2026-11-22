### Anatomical & Kinematic Summary

To build a clean MJCF where `qpos = 0` (barring root height) is an anatomically grounded standing posture matching Fischer (2011) and Jaegger (2002), all limb bones are defined in world-aligned standing coordinates at zero joint angles:

* **Withers height ($H_w$):** $0.570\text{ m}$ (top of scapular border / thoracic spinous processes at $x = +0.16\text{ m}$).
* **Body length ($1.08 H_w = 0.616\text{ m}$):** Shoulder/scapular pivot to ischial tuberosity ($\approx 0.61\text{ m}$). Hip-to-scapular pivot is $0.46\text{ m}$.
* **Total mass:** $\mathbf{30.90\text{ kg}}$ distributed per Amit et al. (2009) / Greyhound-Labrador morphometrics: Thorax $10.8\text{ kg}$, Abdomen/Pelvis $7.6\text{ kg}$, Neck $1.6\text{ kg}$, Head $2.5\text{ kg}$, Tail $0.40\text{ kg}$, each Forelimb $2.15\text{ kg}$ ($13.9\%$ pair), each Hindlimb $1.85\text{ kg}$ ($12.0\%$ pair).
* **Forelimb kinematics ($q=0$ stand):** Consolidated `scap_shoulder` joint placed at the functional scapular pivot near the dorsal third of the scapula ($z = 0.510\text{ m}$, consistent with Fischer's center of scapular rotation) driving the scapulohumeral-to-elbow vector directly, or modeled with realistic elbow height ($z = 0.305\text{ m}$) and carpus height ($z = 0.135\text{ m}$). Included angles at `stand`: elbow $138^\circ$, carpus $195^\circ$ ($15^\circ$ passive dorsiflexion), paw sphere radius $0.030\text{ m}$ resting cleanly at $z = 0.0\text{ m}$.
* **Hindlimb kinematics ($q=0$ stand):** Hip at $z = 0.485\text{ m}$, femur $0.177\text{ m}$ ($0.31 H_w$), tibia $0.185\text{ m}$ ($0.325 H_w$), tarsus-metatarsus-paw to ground with pad sphere $0.030\text{ m}$. Included angles at `stand`: hip $108^\circ$, stifle $138^\circ$, hock $142^\circ$.
* **Joint sign convention:** All pitch axes are `0 1 0` ($+y$, right-to-left). In MuJoCo right-handed coordinates ($+x$ forward, $+z$ up), **positive pitch rotates the distal segment caudally/ventrally** (retraction / flexion for hip & shoulder; flexion for carpus; extension for stifle; flexion for neck/spine downward) and **negative pitch swings cranially/dorsally** (protraction). Ranges are tailored so `sit`, `down` (sphinx), lateral recumbency, and full gallop suspension are well inside joint limits.
* **MJX contact budget:** `contype="1" conaffinity="0"` on all robot collision geoms except the floor (`contype="1" conaffinity="1"`), plus explicit `<pair>` or selective bitmasks so **self-collisions are disabled** (standard in MJX locomotion for speed and stability) and **only 13 primitive geoms** can ever touch the ground plane: 4 paws, 2 elbows, 2 hocks, 2 trunk capsules, 1 sternum sphere, 2 head/muzzle geoms. That yields fixed, tiny contact buffers in MJX (`nconmax <= 24`).

```xml
<mujoco model="labrador_retriever_30kg">
  <compiler angle="radian" coordinate="local" autolimits="true" inertiafromgeom="true"/>

  <!-- Tuned for MJX / Brax PPO on a single RTX 3090:
       dt = 0.004s (250 Hz physics, action repeat 5 -> 50 Hz policy),
       Newton solver with 4 iterations + 6 ls_iterations converges stiff PD + contacts cleanly in JAX -->
  <option timestep="0.004" iterations="4" ls_iterations="6" solver="Newton" gravity="0 0 -9.81">
    <flag eulerdamp="disable"/>
  </option>

  <custom>
    <numeric name="init_qpos" data="0 0 0.495 1 0 0 0  0 0  0 0 0  0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0"/>
  </custom>

  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.2 0.2 0.2"/>
    <rgba haze="0.85 0.90 0.95 1"/>
    <global azimuth="140" elevation="-18"/>
  </visual>

  <default>
    <joint damping="1.2" armature="0.015" frictionloss="0.05"/>
    <!-- Default geom is visual-only; collision geoms explicitly opt into class="col" -->
    <geom contype="0" conaffinity="0" group="2" rgba="0.82 0.68 0.45 1" density="1000"/>

    <!-- Robot-to-ground-only collision class (no self-collision overhead in MJX) -->
    <default class="col">
      <geom contype="1" conaffinity="0" group="1" condim="3" friction="0.9 0.005 0.0001"
            solref="0.014 1.0" solimp="0.90 0.97 0.002" rgba="0.76 0.60 0.38 1"/>
    </default>
    <default class="paw">
      <geom contype="1" conaffinity="0" group="1" type="sphere" size="0.030" condim="3"
            friction="1.05 0.005 0.0001" solref="0.012 1.0" solimp="0.92 0.98 0.001"
            rgba="0.22 0.20 0.18 1"/>
    </default>
    <default class="joint_pad">
      <geom contype="1" conaffinity="0" group="1" type="sphere" size="0.024" condim="3"
            friction="0.75 0.005 0.0001" solref="0.016 1.0" solimp="0.90 0.96 0.002"
            rgba="0.65 0.50 0.32 1"/>
    </default>

    <!-- Position actuators: kp in N*m/rad, kv in N*m*s/rad via dampratio/kv,forcerange for 31 kg dog -->
    <default class="act_spine">
      <position kp="140" kv="9.0" forcerange="-85 85"/>
    </default>
    <default class="act_neck">
      <position kp="45" kv="3.5" forcerange="-30 30"/>
    </default>
    <default class="act_prox_pitch">
      <position kp="115" kv="6.5" forcerange="-75 75"/>
    </default>
    <default class="act_prox_abd">
      <position kp="85" kv="5.0" forcerange="-45 45"/>
    </default>
    <default class="act_mid_pitch">
      <position kp="95" kv="5.0" forcerange="-65 65"/>
    </default>
    <default class="act_dist_pitch">
      <position kp="55" kv="2.8" forcerange="-38 38"/>
    </default>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.26 0.29 0.32" rgb2="0.32 0.35 0.38"/>
    <material name="grid_mat" texture="grid" texrepeat="6 6" texuniform="true" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <!-- Only the floor has conaffinity="1", so robot geoms collide strictly with the floor -->
    <geom name="floor" type="plane" size="50 50 0.1" contype="0" conaffinity="1"
          condim="3" friction="1.0 0.005 0.0001" material="grid_mat"/>

    <!-- ==================== THORAX (Root Body) ====================
         Standing root origin at (0, 0, 0.495). Withers top reaches z = 0.570 m. -->
    <body name="thorax" pos="0 0 0.495">
      <freejoint name="root"/>
      <site name="imu" pos="0.08 0 0.0" size="0.015"/>
      <site name="withers" pos="0.14 0 0.075" size="0.012"/>

      <!-- Ribcage collision + mass (10.80 kg total thorax) -->
      <geom name="thorax_col" class="col" type="capsule" fromto="-0.03 0 -0.02 0.21 0 0.0"
            size="0.105" mass="9.60"/>
      <!-- Sternum/keel sphere for sphinx-down & chest contact -->
      <geom name="sternum_col" class="col" type="sphere" pos="0.15 0 -0.065"
            size="0.075" mass="1.20"/>
      <!-- Dorsal withers visual contour -->
      <geom name="withers_vis" type="capsule" fromto="0.04 0 0.03 0.19 0 0.025" size="0.050" mass="0"/>

      <!-- ==================== NECK & HEAD ==================== -->
      <body name="neck" pos="0.23 0 0.035">
        <joint name="neck_yaw" type="hinge" axis="0 0 1" range="-0.85 0.85" damping="1.0" armature="0.008"/>
        <joint name="neck_pitch" type="hinge" axis="0 1 0" range="-0.85 1.05" damping="1.2" armature="0.010"/>
        <!-- Standing neck chord ~28 deg above horizontal -->
        <geom name="neck_vis" type="capsule" fromto="0 0 0 0.145 0 0.078" size="0.058" mass="1.60"/>

        <body name="head" pos="0.145 0 0.078">
          <joint name="head_pitch" type="hinge" axis="0 1 0" range="-0.90 0.95" damping="0.8" armature="0.006"/>
          <site name="head_site" pos="0.06 0 0.02" size="0.01"/>
          <!-- Cranium collision (side-lying / rolling protection) -->
          <geom name="cranium_col" class="col" type="sphere" pos="0.045 0 0.005" size="0.068" mass="1.75"/>
          <!-- Muzzle collision -->
          <geom name="muzzle_col" class="col" type="capsule" fromto="0.07 0 -0.012 0.175 0 -0.028"
                size="0.040" mass="0.75"/>
          <!-- Visual ears -->
          <geom name="ear_l" type="box" pos="0.03 0.068 -0.02" size="0.028 0.008 0.048"
                euler="0.18 0.12 0" rgba="0.68 0.52 0.30 1" mass="0"/>
          <geom name="ear_r" type="box" pos="0.03 -0.068 -0.02" size="0.028 0.008 0.048"
                euler="-0.18 0.12 0" rgba="0.68 0.52 0.30 1" mass="0"/>
        </body>
      </body>

      <!-- ==================== LEFT FORELIMB ====================
           Functional scapular-shoulder pivot at x = +0.165, y = +0.095, z_world = 0.510 m.
           Elbow at z_world = 0.305 m, Carpus at z_world = 0.135 m, Paw center at z_world = 0.030 m. -->
      <body name="fl_upper" pos="0.165 0.095 0.015">
        <joint name="fl_scap_abd" type="hinge" axis="1 0 0" range="-0.38 0.42" damping="1.5" armature="0.015"/>
        <joint name="fl_scap_pitch" type="hinge" axis="0 1 0" range="-1.25 1.15" damping="1.8" armature="0.018"/>
        <!-- Scapula + Humerus composite segment (standing shoulder angle 112 deg internal) -->
        <geom name="fl_scap_vis" type="capsule" fromto="0.01 0 0.04 -0.055 0 -0.085" size="0.038" mass="0.65"/>
        <geom name="fl_hum_vis" type="capsule" fromto="-0.055 0 -0.085 -0.035 0 -0.205" size="0.035" mass="0.65"/>

        <body name="fl_lower" pos="-0.035 0 -0.205">
          <!-- Elbow pitch: +pitch flexes forelimb caudally (folding paw toward chest/sphinx) -->
          <joint name="fl_elbow_pitch" type="hinge" axis="0 1 0" range="-0.55 2.05" damping="1.2" armature="0.012"/>
          <!-- Point of elbow (olecranon) pad for sphinx down -->
          <geom name="fl_elbow_col" class="joint_pad" pos="-0.022 0 0.005" mass="0.05"/>
          <!-- Radius/Ulna (~0.172 m = 0.30 H_w projected) -->
          <geom name="fl_rad_vis" type="capsule" fromto="0 0 0 0.025 0 -0.170" size="0.027" mass="0.52"/>

          <body name="fl_paw" pos="0.025 0 -0.170">
            <!-- Carpus pitch: +pitch = palmar flexion (up to ~145 deg for swing/down), -pitch = dorsiflexion -->
            <joint name="fl_carpus_pitch" type="hinge" axis="0 1 0" range="-0.45 2.35" damping="0.6" armature="0.006"/>
            <geom name="fl_meta_vis" type="capsule" fromto="0 0 0 0.032 0 -0.105" size="0.021" mass="0.16"/>
            <geom name="fl_foot" class="paw" pos="0.032 0 -0.105" mass="0.12"/>
            <site name="fl_foot_site" pos="0.032 0 -0.105" size="0.01"/>
          </body>
        </body>
      </body>

      <!-- ==================== RIGHT FORELIMB ==================== -->
      <body name="fr_upper" pos="0.165 -0.095 0.015">
        <joint name="fr_scap_abd" type="hinge" axis="1 0 0" range="-0.42 0.38" damping="1.5" armature="0.015"/>
        <joint name="fr_scap_pitch" type="hinge" axis="0 1 0" range="-1.25 1.15" damping="1.8" armature="0.018"/>
        <geom name="fr_scap_vis" type="capsule" fromto="0.01 0 0.04 -0.055 0 -0.085" size="0.038" mass="0.65"/>
        <geom name="fr_hum_vis" type="capsule" fromto="-0.055 0 -0.085 -0.035 0 -0.205" size="0.035" mass="0.65"/>

        <body name="fr_lower" pos="-0.035 0 -0.205">
          <joint name="fr_elbow_pitch" type="hinge" axis="0 1 0" range="-0.55 2.05" damping="1.2" armature="0.012"/>
          <geom name="fr_elbow_col" class="joint_pad" pos="-0.022 0 0.005" mass="0.05"/>
          <geom name="fr_rad_vis" type="capsule" fromto="0 0 0 0.025 0 -0.170" size="0.027" mass="0.52"/>

          <body name="fr_paw" pos="0.025 0 -0.170">
            <joint name="fr_carpus_pitch" type="hinge" axis="0 1 0" range="-0.45 2.35" damping="0.6" armature="0.006"/>
            <geom name="fr_meta_vis" type="capsule" fromto="0 0 0 0.032 0 -0.105" size="0.021" mass="0.16"/>
            <geom name="fr_foot" class="paw" pos="0.032 0 -0.105" mass="0.12"/>
            <site name="fr_foot_site" pos="0.032 0 -0.105" size="0.01"/>
          </body>
        </body>
      </body>

      <!-- ==================== POSTERIOR TRUNK (Lumbar + Pelvis) ==================== -->
      <body name="abdomen" pos="-0.075 0 0.005">
        <!-- Spine pitch (+pitch flexes pelvis ventrally for sitting & galloping collection) and yaw -->
        <joint name="spine_yaw" type="hinge" axis="0 0 1" range="-0.48 0.48" damping="2.2" armature="0.020"/>
        <joint name="spine_pitch" type="hinge" axis="0 1 0" range="-0.52 0.78" damping="2.8" armature="0.025"/>

        <!-- Lumbar + Pelvis collision capsule (7.60 kg) -->
        <geom name="abdomen_col" class="col" type="capsule" fromto="-0.02 0 0.0 -0.24 0 -0.005"
              size="0.088" mass="7.60"/>
        <site name="pelvis_site" pos="-0.22 0 0.0" size="0.012"/>

        <!-- ==================== PASSIVE SPRING-DAMPER TAIL ==================== -->
        <body name="tail_prox" pos="-0.295 0 0.025">
          <!-- Labrador "otter tail": carried near horizontal / slight upward sweep -->
          <joint name="tail_1" type="hinge" axis="0 1 0" range="-1.1 1.1"
                 stiffness="5.5" springref="-0.15" damping="0.45" armature="0.003"/>
          <geom name="tail_1_vis" type="capsule" fromto="0 0 0 -0.16 0 -0.02" size="0.026" mass="0.25"/>

          <body name="tail_dist" pos="-0.16 0 -0.02">
            <joint name="tail_2" type="hinge" axis="0 1 0" range="-1.1 1.1"
                   stiffness="3.0" springref="-0.20" damping="0.25" armature="0.002"/>
            <geom name="tail_2_vis" type="capsule" fromto="0 0 0 -0.16 0 0.01" size="0.020" mass="0.15"/>
          </body>
        </body>

        <!-- ==================== LEFT HINDLIMB ====================
             Hip at x_world = -0.295, y = +0.092, z_world = 0.485 m (46 cm behind scapular pivot).
             Femur: (+0.055, 0, -0.168) -> len 0.177 m (0.31 H_w), Stifle at z_world = 0.317 m.
             Tibia: (-0.082, 0, -0.166) -> len 0.185 m (0.325 H_w), Hock at z_world = 0.151 m.
             Pes:   (+0.028, 0, -0.121) -> Paw sphere (r=0.030) center at z_world = 0.030 m.
             Included angles at q=0: Stifle = 136 deg, Hock = 142 deg. -->
        <body name="hl_thigh" pos="-0.220 0.092 -0.015">
          <joint name="hl_hip_abd" type="hinge" axis="1 0 0" range="-0.40 0.48" damping="1.5" armature="0.015"/>
          <!-- -pitch swings thigh cranially (sit / gallop flexion); +pitch extends caudally -->
          <joint name="hl_hip_pitch" type="hinge" axis="0 1 0" range="-1.65 1.05" damping="1.8" armature="0.018"/>
          <geom name="hl_thigh_vis" type="capsule" fromto="0 0 0 0.055 0 -0.168" size="0.044" mass="1.05"/>

          <body name="hl_shank" pos="0.055 0 -0.168">
            <!-- Stifle: +pitch flexes tibia caudally toward femur (up to deep crouch/sit/down) -->
            <joint name="hl_stifle_pitch" type="hinge" axis="0 1 0" range="-1.85 0.55" damping="1.2" armature="0.012"/>
            <geom name="hl_shank_vis" type="capsule" fromto="0 0 0 -0.082 0 -0.166" size="0.030" mass="0.50"/>

            <body name="hl_pes" pos="-0.082 0 -0.166">
              <!-- Hock: -pitch flexes pes cranially toward tibia (swing/sit/down); +pitch extends (push-off) -->
              <joint name="hl_hock_pitch" type="hinge" axis="0 1 0" range="-0.65 1.95" damping="0.7" armature="0.007"/>
              <!-- Calcaneus / hock pad for sitting and sphinx-down plantigrade support -->
              <geom name="hl_hock_col" class="joint_pad" pos="-0.018 0 0.005" mass="0.05"/>
              <geom name="hl_meta_vis" type="capsule" fromto="0 0 0 0.028 0 -0.121" size="0.021" mass="0.13"/>
              <geom name="hl_foot" class="paw" pos="0.028 0 -0.121" mass="0.12"/>
              <site name="hl_foot_site" pos="0.028 0 -0.121" size="0.01"/>
            </body>
          </body>
        </body>

        <!-- ==================== RIGHT HINDLIMB ==================== -->
        <body name="hr_thigh" pos="-0.220 -0.092 -0.015">
          <joint name="hr_hip_abd" type="hinge" axis="1 0 0" range="-0.48 0.40" damping="1.5" armature="0.015"/>
          <joint name="hr_hip_pitch" type="hinge" axis="0 1 0" range="-1.65 1.05" damping="1.8" armature="0.018"/>
          <geom name="hr_thigh_vis" type="capsule" fromto="0 0 0 0.055 0 -0.168" size="0.044" mass="1.05"/>

          <body name="hr_shank" pos="0.055 0 -0.168">
            <joint name="hr_stifle_pitch" type="hinge" axis="0 1 0" range="-1.85 0.55" damping="1.2" armature="0.012"/>
            <geom name="hr_shank_vis" type="capsule" fromto="0 0 0 -0.082 0 -0.166" size="0.030" mass="0.50"/>

            <body name="hr_pes" pos="-0.082 0 -0.166">
              <joint name="hr_hock_pitch" type="hinge" axis="0 1 0" range="-0.65 1.95" damping="0.7" armature="0.007"/>
              <geom name="hr_hock_col" class="joint_pad" pos="-0.018 0 0.005" mass="0.05"/>
              <geom name="hr_meta_vis" type="capsule" fromto="0 0 0 0.028 0 -0.121" size="0.021" mass="0.13"/>
              <geom name="hr_foot" class="paw" pos="0.028 0 -0.121" mass="0.12"/>
              <site name="hr_foot_site" pos="0.028 0 -0.121" size="0.01"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- ==================== 21 POSITION ACTUATORS (Tail is passive) ==================== -->
  <actuator>
    <!-- Neck & Head (3) -->
    <position name="act_neck_yaw"       joint="neck_yaw"        class="act_neck"/>
    <position name="act_neck_pitch"     joint="neck_pitch"      class="act_neck"/>
    <position name="act_head_pitch"     joint="head_pitch"      class="act_neck"/>

    <!-- Forelimb Left (4) -->
    <position name="act_fl_scap_abd"    joint="fl_scap_abd"     class="act_prox_abd"/>
    <position name="act_fl_scap_pitch"  joint="fl_scap_pitch"   class="act_prox_pitch"/>
    <position name="act_fl_elbow_pitch" joint="fl_elbow_pitch"  class="act_mid_pitch"/>
    <position name="act_fl_carpus_pitch" joint="fl_carpus_pitch" class="act_dist_pitch"/>

    <!-- Forelimb Right (4) -->
    <position name="act_fr_scap_abd"    joint="fr_scap_abd"     class="act_prox_abd"/>
    <position name="act_fr_scap_pitch"  joint="fr_scap_pitch"   class="act_prox_pitch"/>
    <position name="act_fr_elbow_pitch" joint="fr_elbow_pitch"  class="act_mid_pitch"/>
    <position name="act_fr_carpus_pitch" joint="fr_carpus_pitch" class="act_dist_pitch"/>

    <!-- Spine (2) -->
    <position name="act_spine_yaw"      joint="spine_yaw"       class="act_spine"/>
    <position name="act_spine_pitch"    joint="spine_pitch"     class="act_spine"/>

    <!-- Hindlimb Left (4) -->
    <position name="act_hl_hip_abd"     joint="hl_hip_abd"      class="act_prox_abd"/>
    <position name="act_hl_hip_pitch"   joint="hl_hip_pitch"    class="act_prox_pitch"/>
    <position name="act_hl_stifle_pitch" joint="hl_stifle_pitch" class="act_mid_pitch"/>
    <position name="act_hl_hock_pitch"  joint="hl_hock_pitch"   class="act_dist_pitch"/>

    <!-- Hindlimb Right (4) -->
    <position name="act_hr_hip_abd"     joint="hr_hip_abd"      class="act_prox_abd"/>
    <position name="act_hr_hip_pitch"   joint="hr_hip_pitch"    class="act_prox_pitch"/>
    <position name="act_hr_stifle_pitch" joint="hr_stifle_pitch" class="act_mid_pitch"/>
    <position name="act_hr_hock_pitch"  joint="hr_hock_pitch"   class="act_dist_pitch"/>
  </actuator>

  <sensor>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_accel" site="imu"/>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <velocimeter name="imu_linvel" site="imu"/>
  </sensor>

  <!-- ==================== KEYFRAMES ====================
       qpos order (30 floats):
       [0..6]   root pos (3), root quat (4)
       [7..9]   neck_yaw, neck_pitch, head_pitch
       [10..13] fl_scap_abd, fl_scap_pitch, fl_elbow_pitch, fl_carpus_pitch
       [14..17] fr_scap_abd, fr_scap_pitch, fr_elbow_pitch, fr_carpus_pitch
       [18..19] spine_yaw, spine_pitch
       [20..21] tail_1, tail_2
       [22..25] hl_hip_abd, hl_hip_pitch, hl_stifle_pitch, hl_hock_pitch
       [26..29] hr_hip_abd, hr_hip_pitch, hr_stifle_pitch, hr_hock_pitch

       ctrl order (21 floats): same as qpos[7..29] excluding tail_1, tail_2 -->
  <keyframe>
    <!-- 1. STAND: 4 paws at z = 0, withers at 0.57 m -->
    <key name="stand"
         qpos="0 0 0.495 1 0 0 0
               0 0 0
               0 0 0 0
               0 0 0 0
               0 0
               -0.15 -0.20
               0 0 0 0
               0 0 0 0"
         ctrl="0 0 0  0 0 0 0  0 0 0 0  0 0  0 0 0 0  0 0 0 0"/>

    <!-- 2. SIT: Thorax pitched ~36 deg nose-up, lumbar flexed, hocks & hindpaws plantigrade on floor,
         forelimbs propped straight ahead -->
    <key name="sit"
         qpos="0 0 0.375 0.951 0 -0.309 0
               0 0.35 0.15
               0 0.24 -0.10 0.18
               0 0.24 -0.10 0.18
               0 0.42
               0.25 0.35
               0.06 -0.95 -1.45 1.50
               -0.06 -0.95 -1.45 1.50"
         ctrl="0 0.35 0.15  0 0.24 -0.10 0.18  0 0.24 -0.10 0.18  0 0.42  0.06 -0.95 -1.45 1.50  -0.06 -0.95 -1.45 1.50"/>

    <!-- 3. DOWN (Sphinx): Sternum, elbows, antebrachia, hocks, and paws resting flat on floor (z_root = 0.148 m) -->
    <key name="down"
         qpos="0 0 0.148 1 0 0 0
               0 -0.10 0.12
               0.04 0.72 -1.92 1.20
               -0.04 0.72 -1.92 1.20
               0 0.08
               0.10 0.15
               0.08 -1.15 -1.55 1.65
               -0.08 -1.15 -1.55 1.65"
         ctrl="0 -0.10 0.12  0.04 0.72 -1.92 1.20  -0.04 0.72 -1.92 1.20  0 0.08  0.08 -1.15 -1.55 1.65  -0.08 -1.15 -1.55 1.65"/>
  </keyframe>
</mujoco>
```

---

### Concrete MJX + Brax PPO Training Recipe (1× RTX 3090)

#### 1. Observations and Actions

Run physics at $\Delta t = 0.004\text{ s}$ ($250\text{ Hz}$) with `action_repeat = 5` ($50\text{ Hz}$ policy rate, $\Delta t_{\text{ctrl}} = 0.02\text{ s}$).

**Action space ($\mathbf{a}_t \in [-1, 1]^{21}$):**
Target joint positions around a skill-independent nominal standing offset $\mathbf{q}_{\text{stand}} = \mathbf{0}_{21}$:
$$\mathbf{q}^{\text{des}}_t = \text{clip}\!\left(\mathbf{q}_{\text{stand}} + \sigma_a \mathbf{a}_t,\ \mathbf{q}_{\min},\ \mathbf{q}_{\max}\right), \quad \sigma_a = 0.55\text{ rad for locomotion},\ 1.25\text{ rad when postural skills are enabled}$$
*(Using a single $\sigma_a = 1.0\text{ rad}$ with `tanh` squashing from the policy mean works seamlessly across both locomotion and sit/down/get-up without mode-switching the action scale).* Apply a first-order low-pass filter on the motor command to prevent bang-bang chatter before sending to `data.ctrl`:
$$\mathbf{u}_t = 0.8\,\mathbf{q}^{\text{des}}_t + 0.2\,\mathbf{u}_{t-1}$$

**Observation space ($\mathbf{o}_t \in \mathbb{R}^{89}$, symmetric actor, optionally $+15\text{ D}$ privileged critic):**
1. **Proprioception (65 D):**
   * Root gravity vector in `thorax` local frame: $\mathbf{g}_{\text{local}} = R_{\text{root}}^\top [0, 0, -1]^\top \in \mathbb{R}^3$
   * Pelvis gravity vector in `abdomen` local frame: $\mathbf{g}_{\text{pelvis}} \in \mathbb{R}^3$ (crucial for rolling/sit/get-up)
   * Root local angular velocity (from `imu_gyro`): $\boldsymbol{\omega}_{\text{local}} \in \mathbb{R}^3 \times 0.25$
   * Actuated joint positions $(\mathbf{q}_{\text{act}} - \mathbf{q}_{\text{stand}}) \in \mathbb{R}^{21}$ and passive tail positions $\mathbf{q}_{\text{tail}} \in \mathbb{R}^2$
   * Actuated joint velocities $\dot{\mathbf{q}}_{\text{act}} \in \mathbb{R}^{21} \times 0.05$
   * Previous filtered action $\mathbf{a}_{t-1} \in \mathbb{R}^{21}$ *(minus tail, total 21)* — wait: $3+3+3+23+21+21 = 74\text{ D}$.
2. **Task & Gait Command (15 D):**
   * Skill one-hot $\mathbf{m} \in \{0,1\}^4$: `[locomote, sit, down, lie_side]` (get-up is simply transitioning from any recumbent state to `[1,0,0,0]` with zero velocity command).
   * Velocity command $\mathbf{c}_v = [v_x^{\text{cmd}}, v_y^{\text{cmd}}, \omega_z^{\text{cmd}}] \in \mathbb{R}^3$ (set to $\mathbf{0}$ for non-locomote skills).
   * Gait phase clock (8 D): $\big[\sin(2\pi \phi_i), \cos(2\pi \phi_i)\big]_{i \in \{\text{FL, FR, HL, HR}\}}$ when `m = locomote`, masked to $\mathbf{0}_8$ during `sit` / `down` / `lie_side` or when $\|\mathbf{c}_v\| < 0.12\text{ m/s}$.
3. **Privileged Critic Additions (+16 D, asymmetric PPO value network):**
   * True local linear velocity $\mathbf{v}_{\text{local}} \in \mathbb{R}^3$, root height $z_{\text{root}} \in \mathbb{R}^1$, binary contact states for `{4 paws, 2 elbows, 2 hocks, thorax/sternum, abdomen}` (10 D), randomized floor friction $\mu$ & payload mass $\Delta m$ (2 D).

---

#### 2. Multi-Gait, Posture, and Fall-Recovery Reward Formulation

Let total step reward be $r_t = \Delta t_{\text{ctrl}} \left( r_{\text{task}} + r_{\text{gait}} + r_{\text{reg}} \right)$, clipped to $\ge 0$ only if you use early termination (do **not** clip to $\ge 0$ when fall recovery is trained with fixed-horizon episodes!).

##### A. Gait Phase Schedule (Literature-Parameterized)
Advance master phase $\Phi_{t+1} = (\Phi_t + f_{\text{stride}}(v_x^{\text{cmd}})\,\Delta t_{\text{ctrl}}) \bmod 1$, and set per-foot phase $\phi_i = (\Phi + \Delta\phi_i) \bmod 1$ and duty factor $\beta_i$ (stance when $\phi_i < \beta_i$, swing when $\phi_i \ge \beta_i$):

| Gait Mode | Speed Band $v_x^{\text{cmd}}$ | Stride Freq $f_{\text{stride}}$ | Duty Factor $\beta_{\text{fore}} / \beta_{\text{hind}}$ | Phase Offsets $\Delta\phi = [\text{FL, FR, HL, HR}]$ |
| :--- | :--- | :--- | :--- | :--- |
| **Stand** | $< 0.12\text{ m/s}$ | $0\text{ Hz}$ | $1.00 / 1.00$ | $[0, 0, 0, 0]$ (all 4 in stance) |
| **Walk** (lateral-seq) | $0.15 - 1.45\text{ m/s}$ | $1.05 + 0.35 v_x\text{ Hz}$ | $0.65 / 0.63$ | $[0.25,\ 0.75,\ 0.00,\ 0.50]$ (HL$\to$FL$\to$HR$\to$FR) |
| **Trot** | $1.35 - 2.90\text{ m/s}$ | $1.95\text{ Hz}$ (or $1.65 + 0.14 v_x$) | $0.45 / 0.43$ | $[0.00,\ 0.50,\ 0.51,\ 0.01]$ ($-0.01$ diag dissociation) |
| **Gallop** (rotary/transv.) | $2.70 - 5.50\text{ m/s}$ | $2.35 + 0.18 (v_x - 2.7)\text{ Hz}$ | $0.33 / 0.31$ | $[0.00,\ 0.14,\ 0.52,\ 0.65]$ (transverse lead-R) |

Smoothly convert $(\phi_i, \beta_i)$ into a soft expected-contact probability $C_i^{\text{des}}(\phi_i, \beta_i) \in [0, 1]$ using a von Mises / cubic window (width $\sigma_\phi = 0.04$ around $0$ and $\beta_i$, following Margolis & Agrawal *Walk These Ways* 2023).

##### B. Locomotion & Gait Terms (active when `skill == locomote` and upright $g_{z,\text{local}} < -0.65$)
* **Linear & yaw velocity tracking:**
  $$r_{\text{vel}} = 2.5 \exp\!\left(-\frac{\|\mathbf{v}_{xy,\text{local}} - \mathbf{c}_{xy}\|^2}{0.25}\right) + 1.2 \exp\!\left(-\frac{(\omega_{z,\text{local}} - \omega_z^{\text{cmd}})^2}{0.25}\right)$$
* **Footfall phase matching + swing clearance:** Let $c_i = \mathbb{I}(F_{z,i}^{\text{paw}} > 2\text{ N})$ and $p_{z,i}$ be paw world height, $v_{xy,i}$ paw world horizontal speed:
  $$r_{\text{contact}} = 1.2 \sum_{i=1}^4 \Big[ C_i^{\text{des}} c_i + (1 - C_i^{\text{des}})(1 - c_i) \Big]$$
  $$r_{\text{foot\_slip\_clear}} = -0.4 \sum_{i=1}^4 C_i^{\text{des}} \|v_{xy,i}\|^2 - 3.0 \sum_{i=1}^4 (1 - C_i^{\text{des}}) \left(p_{z,i} - p_{z}^{\text{swing}}(\phi_i)\right)^2 \|v_{xy,i}\|$$
  where $p_z^{\text{swing}}(\phi_i) = 0.030 + h_{\text{clear}} \sin\!\left(\pi \frac{\phi_i - \beta_i}{1 - \beta_i}\right)$ with $h_{\text{clear}} = 0.06\text{ m}$ (walk/trot) or $0.10\text{ m}$ (gallop).
* **Breed-natural neck & trunk carriage:**
  $$r_{\text{carriage}} = 0.5 \exp\!\left(-\frac{(z_{\text{root}} - 0.495)^2}{0.004}\right) - 0.8 (g_{x,\text{local}}^2 + g_{y,\text{local}}^2) - 0.15 \|\mathbf{q}_{\text{neck,head}} - [0, 0, 0]^\top\|^2$$

##### C. Postural Skill Rewards (`sit`, `down`, `lie_side`)
For each postural mode $k \in \{\text{sit}, \text{down}, \text{side}\}$, define target keyframe tuple $(z_k^*, \mathbf{g}_k^*, \mathbf{q}_k^*, \mathbf{b}_k^*)$ where $\mathbf{b}_k^* \in \{0,1\}^{10}$ specifies desired ground contacts:
* **`sit`:** $z^* = 0.375$, $\mathbf{g}^* = [0.59, 0, -0.81]$ ($\sim 36^\circ$ nose-up thorax), $\mathbf{q}^* = \mathbf{q}_{\text{key:sit}}$, contacts = `{4 paws + 2 hocks}` ON, `{elbows, sternum, abdomen}` OFF.
* **`down` (sphinx):** $z^* = 0.148$, $\mathbf{g}^* = [0, 0, -1]$, $\mathbf{q}^* = \mathbf{q}_{\text{key:down}}$, contacts = `{4 paws, 2 elbows, 2 hocks, sternum}` ON.
* **`lie_side` (left or right):** $z^* = 0.115$, $\mathbf{g}^* = [0, \pm 1, 0]$, limbs relaxed ($\mathbf{q}^* \approx 0.3\,\mathbf{q}_{\text{down}}$), contacts = `{thorax, abdomen, cranium}` ON.
* **Unified posture reward:**
  $$r_{\text{posture}} = 2.0 \exp\!\left(-\frac{(z_{\text{root}} - z_k^*)^2}{0.006}\right) + 2.0 \exp\!\left(-\frac{\|\mathbf{g}_{\text{local}} - \mathbf{g}_k^*\|^2}{0.08}\right) + 1.5 \exp\!\left(-\frac{\|\mathbf{q}_{\text{act}} - \mathbf{q}_k^*\|^2}{0.60}\right) + 0.8 \text{BCE}_{\text{match}}(\mathbf{b}, \mathbf{b}_k^*) - 0.5 \|\mathbf{v}_{\text{root}}\|^2$$
* **Rolling & Stand-Up:** No extra reward engineering is needed once episodic transitions sample transitions `down` $\to$ `lie_side(L)` $\to$ `lie_side(R)` $\to$ `sit` $\to$ `locomote(stand)` every $3\text{--}5\text{ s}$ within a $15\text{ s}$ episode!

##### D. Fall Recovery Gate (Active whenever uprightness fails in `locomote`/`sit`/`down`)
When $\mathbf{g}_{\text{local}} \cdot \mathbf{g}_k^* < 0.5$ (dog is knocked onto its side or back):
* **Gate off** velocity/phase/joint-nominal penalties (multiplying them by $\text{clamp}\big((\mathbf{g}_{\text{local}}\cdot \mathbf{g}_k^* - 0.3)/0.5, 0, 1\big)$) so the dog isn't punished for swinging limbs wildly to flip over.
* **Add dense uprighting potential:**
  $$r_{\text{recover}} = 2.5 \left(\frac{\mathbf{g}_{\text{local}} \cdot \mathbf{g}_k^* + 1}{2}\right)^2 + 1.5\,\text{clamp}\!\left(\frac{z_{\text{root}}}{z_k^*}, 0, 1\right) + 1.0\,\mathbb{I}(\text{angular momentum toward upright})$$

##### E. Always-On Smoothness Regularizers
$$r_{\text{reg}} = -0.0002 \|\boldsymbol{\tau}_{\text{actuator}}\|^2 - 0.08 \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2 - 0.03 \|\mathbf{a}_t - 2\mathbf{a}_{t-1} + \mathbf{a}_{t-2}\|^2 - 2.0 \|\max(\mathbf{0}, |\mathbf{q}| - 0.95\,\mathbf{q}_{\lim})\|^2$$

---

#### 3. Terminations, Domain Randomization, and RTX 3090 MJX + Brax PPO Budget

* **Terminations:**
  * **NaN / numerical divergence** or $z_{\text{root}} < 0.05\text{ m}$ or $z_{\text{root}} > 1.3\text{ m}$ $\implies$ hard reset with $-10$ penalty.
  * **Crucial for fall recovery & rolling:** Do **not** terminate on trunk/head/side contact! Instead, initialize episodes across a mixture: $50\%$ `stand`, $15\%$ `sit`, $15\%$ `down`, and $20\%$ **random fallen poses** (dropped from $z \in [0.25, 0.55]\text{ m}$ with random roll $\in [-\pi, \pi]$, pitch $\in [-0.6, 0.6]$, random joint angles). Episode horizon = $750$ policy steps ($15\text{ s}$), with command resampling every $200$ steps ($4\text{ s}$).
* **Domain Randomization (sampled per environment reset via `jax.vmap` over `mjx.Model`):**
  * Floor & foot friction `geom_friction[:, 0]`: $\mathcal{U}(0.45, 1.25)$
  * Body masses `body_mass`: $\times \mathcal{U}(0.85, 1.15)$ per segment + thorax CoM shift $\mathcal{U}(-0.03, +0.03)\text{ m}$
  * Actuator gains `actuator_gainprm` ($k_p$) and `biasprm` ($k_v$): $\times \mathcal{U}(0.80, 1.20)$
  * Joint armature & damping: $\times \mathcal{U}(0.85, 1.25)$; motor zero-offset calibration noise: $\mathcal{N}(0, 0.025\text{ rad})$
  * Observation noise: gyro $\pm 0.08\text{ rad/s}$, gravity $\pm 0.03$, joint pos $\pm 0.02\text{ rad}$, joint vel $\pm 0.3\text{ rad/s}$, plus 1-step ($20\text{ ms}$) action latency randomization ($50\%$ probability).
  * External impulse pushes on `thorax` and `abdomen`: $\Delta v_{xy} \sim \mathcal{U}(-1.2, 1.2)\text{ m/s}$ every $3.5\text{ s}$ during locomotion.
* **Brax PPO Hyperparameters & 1× RTX 3090 (24 GB VRAM) Throughput:**
  * Set `XLA_PYTHON_CLIENT_PREALLOCATE=true` and `mjx_model` overrides `nconmax = 24`, `njmax = 96` (with 13 ground-only collision primitives, peak simultaneous contacts is $\le 14$, keeping VRAM around ~11–14 GB at 4,096 envs).
  * `num_envs = 4096`, `unroll_length = 20`, `batch_size = 1024`, `num_minibatches = 32` ($4096 \times 20 = 81{,}920$ transitions/epoch, split into $32$ minibatches of $2{,}560$), `num_updates_per_batch = 4`.
  * Networks: Actor MLP `(512, 256, 128)` SiLU/ELU; Critic MLP `(512, 256, 128)` SiLU/ELU.
  * Optimizer: Adam, `learning_rate = 3e-4` with cosine decay to `3e-5`, `max_grad_norm = 1.0`, `entropy_cost = 0.008`, `discounting = 0.97` ($\gamma = 0.97$ at $50\text{ Hz} \implies 0.67\text{ s}$ effective horizon; raise to `0.985` if multi-second get-up stalls), `gae_lambda = 0.95`, `clipping_epsilon = 0.2`.
  * **Step budget & wall-clock on one RTX 3090:** Expect **110k – 165k env steps/s** ($550\text{k} - 825\text{k}$ physics steps/s).
    * **80M env steps (~9–12 min):** Clean standing, sitting, sphinx-down, walk, and steady trot.
    * **250M–350M env steps (~35–50 min):** Full multi-gait transitions (walk $\leftrightarrow$ trot $\leftrightarrow$ gallop), robust side-lying/rolling, and reliable fall recovery under pushes and friction randomization.

---

#### 4. What Usually Goes Wrong in Biological Quadruped RL (and Proven Fixes)

Unlike robotic quadrupeds (Go2/ANYmal) with 12 symmetric motors and rigid boxes, biological dog models have **21+ DOFs** (flexible lumbar spine, 3-segment pantograph legs with carpus/hock, articulated neck/head). Pure task-reward PPO reliably discovers five classic pathologies:

1. **Carpus & Hock Collapse ("Dropped Pasterns" / Plantigrade Scuffling):**
   * *Pathology:* With 4 serial pitch joints per limb, PPO leaves the distal carpus/hock limp or hyper-flexed, walking on its wrists/ankles or snapping the joint against its hard mechanical stop during stance.
   * *How it was fixed:* Give the carpus and hock realistic passive/PD stiffness (`kp=55`, `damping=0.6–0.7`) plus a small stance-phase distal-angle regularizer toward biological standing extension, and penalize `elbow_col` / `hock_col` ground contact whenever `skill == locomote`.
2. **Frozen or Twisted C-Spine & "Dead-Neck" Counterbalancing:**
   * *Pathology:* PPO either locks `spine_pitch` rigid during galloping (producing a stiff robot-bound instead of spinal collection/extension) or twists `spine_yaw` and `neck_yaw` permanently $35^\circ$ sideways as a static reaction-wheel offset.
   * *How it was fixed:* **Symmetry loss / Mirror data augmentation** (Abdolhosseini et al. 2019; Mittal et al. 2024): add an auxiliary actor loss $\|\pi(\mathbf{o}) - \mathcal{M}_a \pi(\mathcal{M}_o \mathbf{o})\|^2$ that mirrors left$\leftrightarrow$right limbs and negates yaw/abduction commands. This eliminates crab-walking, twisted spines, and asymmetric limps overnight.
3. **High-Frequency "Sewing-Machine" Pattering & Zero Suspension in Trot/Gallop:**
   * *Pathology:* Without phase structure, energy penalties + fall avoidance push PPO toward $4\text{ Hz}$ shuffle-trots with $0.75$ duty factor and $1\text{ cm}$ foot clearance—even at $2.2\text{ m/s}$ where a real Labrador trots at $1.95\text{ Hz}$ with a $0.44$ duty factor and aerial phases.
   * *How it was fixed:* **Explicit contact-schedule conditioning** (*Walk These Ways*, Margolis & Agrawal 2023; Siekmann et al. 2021) using the literature $(\beta_i, f_{\text{stride}}, \Delta\phi_i)$ table above, paired with second-order action-smoothness ($\|\mathbf{a}_t - 2\mathbf{a}_{t-1} + \mathbf{a}_{t-2}\|^2$) so high-frequency jitter is strictly more expensive than aerial suspension.
4. **Hock-to-Forelimb Pantograph Ambiguity:**
   * *Pathology:* A 3-link hindlimb (hip–stifle–hock) is kinematically redundant for foot placement; RL often inverts the pantograph (extending the hip while hyper-flexing the stifle) so the thigh points backward while the shank drags vertically.
   * *How it was fixed:* In natural canine locomotion, stifle and hock flex and extend **in concert** (anatomical reciprocal apparatus: superficial digital flexor + peroneus tertius). Adding a mild soft coupling penalty $-w_{\text{pant}} (\Delta q_{\text{stifle}} + \Delta q_{\text{hock}})^2$ during locomotion (since $+q_{\text{stifle}}$ is caudal flexion and $+q_{\text{hock}}$ is extension in our frame, opposite signs flex together: $(\Delta q_{\text{stifle}} + \Delta q_{\text{hock}}) \approx 0$) keeps the femur and metatarsus near-parallel just like a real dog.
5. **Robotic Style Despite Right Footfalls $\to$ Motion Priors (AMP / DeepMimic):**
   * *How the graphics & character-animation literature solved it:*
     * **Reference-Motion Imitation (Peng et al., *DeepMimic* 2018; Zhang et al., Mann-style canine mocap retargeting):** Retarget real dog motion capture (e.g., the **Starke / Zhang 2018 Edinburgh Dog Mocap dataset**, which is recorded on a medium retriever/collie across walk, pace, trot, canter, jump, sit, and lie-down) via inverse kinematics onto this exact MJCF skeleton and track root/joint/end-effector poses. Downside: requires a phase-indexed planner for arbitrary transitions.
     * **Adversarial Motion Priors — AMP (Peng et al. 2021; Escontrela et al. 2022 *AMP-for-Hardware*; Vollenweider et al. 2023 *Multi-AMP*):** Train a small MLP discriminator $D(\mathbf{s}_t^{\text{AMP}}, \mathbf{s}_{t+1}^{\text{AMP}})$ with WGAN-GP / gradient-penalty LSGAN on short state-transition features (joint angles $\mathbf{q}$, joint velocities $\dot{\mathbf{q}}$, local paw positions $\mathbf{p}_{\text{paw}}^{\text{root}}$, root linear/angular velocity, root height) from the retargeted Edinburgh Dog dataset, without any phase variable. Replace hand-tuned gait/carriage heuristics with $r_t = 0.5\,r_{\text{task}} + 0.5\,\max(0, 1 - 0.25(D(\mathbf{s}_t, \mathbf{s}_{t+1}) - 1)^2)$. Multi-AMP (one style reward + task one-hot) produces unmistakably canine scapular roll, head bobbing, spinal flexion in gallop, and natural paw-tucking during sit/down/get-up on top of the exact MJCF above.
