"""Export a trained Brax PPO policy to plain numpy (run in WSL, where JAX is), and check it against JAX.

  python export_policy.py RUN_DIR [--params params.pkl]   -> RUN_DIR/policy_np.npz

The policy network is the observation normalizer (running mean and std), then dense layers with swish, then a last
dense layer whose first half is the action's location; the deterministic action is tanh(location). numpy_policy()
in dog_cpu.py runs it with no JAX, for the real-time loop on the CPU.
"""
import argparse
import json
import os
import pickle

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np


def export(run, params_name="params.pkl"):
    params = pickle.load(open(os.path.join(run, params_name), "rb"))
    norm, pol = params[0], params[1]["params"]
    layers = sorted(pol, key=lambda k: int(k.split("_")[1]))
    out = {"mean": np.asarray(norm.mean, np.float32), "std": np.asarray(norm.std, np.float32),
           "n_layers": np.int32(len(layers))}
    for i, name in enumerate(layers):
        out[f"W{i}"] = np.asarray(pol[name]["kernel"], np.float32)
        out[f"b{i}"] = np.asarray(pol[name]["bias"], np.float32)
    path = os.path.join(run, "policy_np.npz")
    np.savez(path, **out)
    return path


def check(run, path, n=64):
    """The numpy policy against Brax's own inference on random observations near the normalizer's statistics."""
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dog_cpu import NumpyPolicy
    params = pickle.load(open(os.path.join(run, "params.pkl"), "rb"))
    net_cfg = json.load(open(os.path.join(run, "net.json")))
    obs_size = int(params[0].mean.shape[0])
    act_size = int(params[1]["params"][sorted(params[1]["params"])[-1]]["bias"].shape[0]) // 2
    nets = ppo_networks.make_ppo_networks(obs_size, act_size, preprocess_observations_fn=running_statistics.normalize,
                                          **net_cfg)
    jpol = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
    npol = NumpyPolicy(path)
    rng = np.random.default_rng(0)
    obs = np.asarray(params[0].mean) + rng.normal(size=(n, obs_size)) * np.asarray(params[0].std)
    a_j = np.array(jax.vmap(lambda o: jpol(o, jax.random.PRNGKey(0))[0])(obs.astype(np.float32)))
    a_n = np.stack([npol(o) for o in obs])
    return float(np.abs(a_j - a_n).max())


def export_body(run):
    """RUN_DIR/cpu_body.json: the body as the online loop builds it for this run (dog_cpu.consts_from_env)."""
    import glob
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dog_cpu import consts_from_env
    from dog_track import DogTrack, default_config
    rc = json.load(open(os.path.join(run, "run.json")))
    cfg = default_config()
    cfg.muscles, cfg.obs_drive = rc.get("muscles", ""), rc.get("obs_drive", False)
    cfg.action_scale = rc.get("action_scale", cfg.action_scale)
    cfg.lookahead2, cfg.obs_paws, cfg.w_rootvel = rc.get("lookahead2", 0), rc.get("obs_paws", False), rc.get("w_rootvel", 0.0)
    anat = "dog_anat" in rc.get("xml", "")
    flat = os.environ.get("DOG_WORK", "runs") + "/refs_anat/flat" if anat else os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "refs", "walk")
    dummy = sorted(glob.glob(os.path.join(flat, "*.npz")))[:1]  # as online_dog.load() builds it
    env = DogTrack(rc["xml"], dummy, config=cfg, max_dogs=8)
    path = os.path.join(run, "cpu_body.json")
    json.dump(consts_from_env(env), open(path, "w"), indent=0)
    return path


def export_reflex(run):
    """RUN_DIR/cpu_reflex.json for a scramble_up run (train_track.py --task right), built as online_dog.load_reflex
    builds it: its standing references and its scale."""
    import glob
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dog_cpu import consts_from_reflex
    from dog_right import DogRight, right_config
    from dog_track import default_config
    rc = json.load(open(os.path.join(run, "run.json")))
    cfg = right_config(default_config())
    cfg.muscles = rc.get("muscles", "")
    refs = sorted(glob.glob(os.environ.get("DOG_WORK", "runs") + "/refs_anat/flat/*.npz"))
    er = DogRight(rc["xml"], refs, config=cfg, max_dogs=8)
    path = os.path.join(run, "cpu_reflex.json")
    json.dump(consts_from_reflex(er), open(path, "w"), indent=0)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--reflex", action="store_true", help="a scramble_up run: also writes cpu_reflex.json")
    a = ap.parse_args()
    path = export(a.run, a.params)
    err = check(a.run, path)
    extra = export_reflex(a.run) if a.reflex else export_body(a.run)
    print(f"EXPORTED {path} and {extra}; numpy vs JAX action, largest difference {err:.2e}", flush=True)


if __name__ == "__main__":
    main()
