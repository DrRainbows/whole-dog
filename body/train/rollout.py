"""Run a trained tracking policy and record what the physics dog does (run in WSL).

  python rollout.py RUN_DIR REF.npz [REF.npz ...] [--starts 0,200] [--steps 400]

Writes RUN_DIR/rollout.npz: for each start, the simulated qpos per control step, the capture's qpos
at the same moments, and the step at which tracking failed (the dog keeps being simulated after a
failure, so falls play out physically).
"""
import argparse
import glob
import os
import pickle

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from dog_track import DogTrack

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(os.path.dirname(HERE), "dog_mocap_mjx.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("ref", nargs="+")
    ap.add_argument("--starts", default="0")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--xml", default=XML, help="model file (v1 policies need dog_mocap_mjx_v1.xml)")
    ap.add_argument("--no-assist", action="store_true", help="run a trunk-assist policy with the assist switched off")
    ap.add_argument("--tag", default="", help="suffix for the output file (rollout{tag}.npz)")
    a = ap.parse_args()
    refs = sorted(p for r in a.ref for p in (glob.glob(r) if any(c in r for c in "*?[") else [r]))
    from dog_track import default_config

    cfg = default_config()
    run_cfg = os.path.join(a.run, "run.json")
    if os.path.exists(run_cfg):  # rebuild the run's own settings (e.g. action scale)
        import json as _json

        rc = _json.load(open(run_cfg))
        cfg.action_scale = rc.get("action_scale", cfg.action_scale)
        cfg.lookahead2, cfg.obs_paws, cfg.w_rootvel = rc.get("lookahead2", 0), rc.get("obs_paws", False), rc.get("w_rootvel", 0.0)
        cfg.rfc_force, cfg.rfc_torque, cfg.w_rfc = rc.get("rfc_force", 0.0), rc.get("rfc_torque", 0.0), rc.get("w_rfc", 0.0)
        if a.no_assist and (cfg.rfc_force or cfg.rfc_torque):  # same policy, assist off: walks without the hand?
            cfg.rfc_force, cfg.rfc_torque = 1e-9, 1e-9
    cfg.restart_info = False  # keep following the capture after a failure
    env = DogTrack(a.xml, refs, config=cfg, max_dogs=64)
    params = pickle.load(open(os.path.join(a.run, "params.pkl"), "rb"))
    import json

    cfg_path = os.path.join(a.run, "net.json")
    net_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {
        "policy_hidden_layer_sizes": (512, 256, 128), "value_hidden_layer_sizes": (512, 256, 128)}
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize, **net_cfg)
    policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    starts = [int(s) for s in a.starts.split(",")]

    def run_one(k):
        st0 = env.reset_at(jax.random.PRNGKey(0), k)

        def body(st, i):
            act, _ = policy(st.obs, jax.random.PRNGKey(i))
            # keep simulating after a failure: clear 'done' so the episode isn't treated as over
            nxt = env.step(st, act)
            m = nxt.metrics
            return nxt.replace(done=jp.zeros(())), (nxt.pipeline_state.qpos, nxt.done, nxt.info["k"],
                                                     jp.stack([m["assist_f"], m["assist_t"]]))

        _, (qs, dones, ks, assist) = jax.lax.scan(body, st0, jp.arange(a.steps))
        return qs, dones, ks, assist

    out = {}
    run = jax.jit(run_one)
    for k in starts:
        qs, dones, ks, assist = (np.array(x) for x in run(jp.int32(k)))
        end = int(np.array(env._end)[k])
        fail = int(np.argmax(dones > 0)) if (dones > 0).any() else -1
        ref_q = np.array(env._ref_q)[np.minimum(ks, end)]
        out[f"sim_{k}"], out[f"ref_{k}"], out[f"fail_{k}"], out[f"assist_{k}"] = qs, ref_q, fail, assist
        held = assist[: fail if fail > 0 else len(assist)]
        print(f"start {k}: {len(qs)} steps; first failure or clip end at step {fail}; clip ends at frame {end}; "
              f"assist force {held[:, 0].mean():.3f} x weight, torque {held[:, 1].mean():.3f}", flush=True)
    np.savez(os.path.join(a.run, f"rollout{a.tag}.npz"), starts=np.array(starts), **out)


if __name__ == "__main__":
    main()
