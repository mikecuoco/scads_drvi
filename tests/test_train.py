"""factorize.train: the distributed decisions, made where they can be checked.

None of this needs a GPU, torch or scvi. That is the point of the split -- how many
processes a run should have, which rank may write, and what the record says are all
decided from the environment and the config, so they are testable on a laptop, and the
only thing left for a GPU to find out is whether the fit is any good.

The one test that does need torch is the round trip of the written record back through
``fit_meta``, and even that needs only numpy.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from scads_drvi.config import Project
from scads_drvi.factorize.train import (
    DDP_STRATEGY,
    DDP_STRICT_STRATEGY,
    Launch,
    TrainConfig,
    check_config,
    fit_record,
    is_global_zero,
    resolve_launch,
    train_kwargs,
    unwrap_compiled,
)


def config(**changes) -> TrainConfig:
    base = {"adata": "/nowhere/input.h5ad", "fit": "a_fit"}
    return TrainConfig(**{**base, **changes})


# -- resolve_launch ---------------------------------------------------------------------


def test_one_device_gets_no_strategy():
    """A DDP strategy on one device builds a sampler and a group for a run with neither."""
    launch = resolve_launch(1, env={})
    assert launch.strategy is None
    assert launch.world_size == 1
    assert not launch.distributed


def test_several_devices_default_to_the_tolerant_strategy():
    launch = resolve_launch(4, env={})
    assert launch.strategy == DDP_STRATEGY
    assert launch.world_size == 4
    assert launch.num_nodes == 1
    assert not launch.external


def test_find_unused_parameters_false_takes_the_faster_strategy():
    """The Pareto knob: same run, no autograd-graph walk before each reduction."""
    cfg = config(devices=4, find_unused_parameters=False)
    launch = resolve_launch(cfg.devices, default_strategy=cfg.default_strategy(), env={})
    assert launch.strategy == DDP_STRICT_STRATEGY
    assert "find_unused_parameters" not in launch.strategy


def test_a_device_list_is_counted_not_passed_through_as_one():
    launch = resolve_launch([0, 2, 3], env={})
    assert launch.world_size == 3
    assert launch.devices == [0, 2, 3]


def test_all_devices_needs_the_accelerators_count_to_be_known():
    """`-1` and "auto" are answerable only by the accelerator, so world_size stays None."""
    assert resolve_launch(-1, env={}).world_size is None
    assert resolve_launch("auto", env={}).world_size is None
    assert resolve_launch(-1, env={}, available=8).world_size == 8


def test_explicit_strategy_on_one_device_is_refused():
    with pytest.raises(SystemExit, match="single device"):
        resolve_launch(1, strategy=DDP_STRATEGY, env={})


def test_zero_devices_is_refused():
    with pytest.raises(SystemExit, match="0 devices"):
        resolve_launch(0, env={})


@pytest.mark.parametrize(
    ("env", "world", "per_node"),
    [
        ({"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"}, 4, 4),
        ({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "4"}, 8, 4),
        ({"WORLD_SIZE": "2"}, 2, 2),
        ({"SLURM_NTASKS": "6", "SLURM_NTASKS_PER_NODE": "3"}, 6, 3),
    ],
)
def test_an_external_launcher_decides_the_device_count(env, world, per_node):
    """torchrun and srun have already started the processes; Lightning must not spawn."""
    launch = resolve_launch(1, env=env)
    assert launch.external
    assert launch.devices == per_node
    assert launch.world_size == world
    assert launch.num_nodes == world // per_node
    assert launch.strategy == DDP_STRATEGY


def test_asking_for_a_device_count_under_an_external_launcher_is_refused():
    """Two authorities on the same number is how a job ends up with the wrong one."""
    with pytest.raises(SystemExit, match="launched externally"):
        resolve_launch(2, env={"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"})


def test_the_default_device_count_does_not_count_as_asking():
    launch = resolve_launch(1, env={"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"})
    assert launch.devices == 4


def test_a_ragged_world_is_refused():
    with pytest.raises(SystemExit, match="whole number of nodes"):
        resolve_launch(1, env={"WORLD_SIZE": "7", "LOCAL_WORLD_SIZE": "4"})


# -- which process may write ------------------------------------------------------------


class FakeTrainer:
    def __init__(self, global_rank=0, world_size=1):
        self.global_rank = global_rank
        self.world_size = world_size


class FakeModel:
    def __init__(self, trainer=None, module=None):
        self.trainer = trainer
        self.module = module


def test_the_trainer_outranks_the_environment():
    """Once Lightning exists it is authoritative; the env is only how children find out."""
    model = FakeModel(trainer=FakeTrainer(global_rank=3))
    assert not is_global_zero(model, env={})
    assert is_global_zero(FakeModel(trainer=FakeTrainer(global_rank=0)), env={"RANK": "2"})


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, True),
        ({"RANK": "0"}, True),
        ({"RANK": "1"}, False),
        ({"SLURM_PROCID": "2"}, False),
        ({"LOCAL_RANK": "0", "NODE_RANK": "0"}, True),
        ({"LOCAL_RANK": "1"}, False),
        ({"NODE_RANK": "1"}, False),
    ],
)
def test_every_launcher_reports_the_rank_somewhere_else(env, expected):
    assert is_global_zero(env=env) is expected


# -- the config -------------------------------------------------------------------------


def test_a_fit_must_be_a_name_not_a_path():
    with pytest.raises(ValueError, match="directory name"):
        config(fit="nested/fit")


def test_a_strategy_without_ddp_is_refused():
    """scvi decides on the distributed sampler by that substring and nothing else."""
    with pytest.raises(ValueError, match="no 'ddp' in it"):
        config(devices=4, strategy="fsdp")


def test_a_strategy_that_contradicts_the_flag_is_refused():
    with pytest.raises(ValueError, match="disagree"):
        config(devices=4, strategy=DDP_STRATEGY, find_unused_parameters=False)
    with pytest.raises(ValueError, match="disagree"):
        config(devices=4, strategy=DDP_STRICT_STRATEGY, find_unused_parameters=True)


def test_a_strategy_that_agrees_with_the_flag_is_kept():
    assert config(devices=4, strategy=DDP_STRATEGY).strategy == DDP_STRATEGY
    assert (
        config(devices=4, strategy=DDP_STRICT_STRATEGY, find_unused_parameters=False).strategy
        == DDP_STRICT_STRATEGY
    )


def test_the_effective_batch_is_per_device_times_ranks():
    cfg = config(batch_size=128)
    assert cfg.effective_batch_size(4) == 512
    assert cfg.effective_batch_size(1) == 128
    assert cfg.effective_batch_size(None) is None


def test_early_stopping_under_ddp_is_refused():
    """scvi turns it off and warns; a warning in an hours-long log is not a decision."""
    launch = resolve_launch(4, env={})
    with pytest.raises(SystemExit, match="early_stopping"):
        check_config(config(devices=4, early_stopping=True), launch)
    check_config(config(early_stopping=True), resolve_launch(1, env={}))


def test_an_unset_seed_under_ddp_is_refused():
    """Each rank splits train/validation itself; without a seed they disagree."""
    launch = resolve_launch(4, env={})
    with pytest.raises(SystemExit, match="seed=None"):
        check_config(config(devices=4, seed=None), launch)
    check_config(config(seed=None), resolve_launch(1, env={}))


# -- what reaches model.train -----------------------------------------------------------


def test_the_strategy_reaches_train_because_that_is_where_scvi_looks():
    kwargs = train_kwargs(config(devices=4), resolve_launch(4, env={}))
    assert kwargs["strategy"] == DDP_STRATEGY
    assert kwargs["devices"] == 4
    assert kwargs["accelerator"] == "gpu"
    assert "num_nodes" not in kwargs


def test_one_device_passes_no_strategy_at_all():
    kwargs = train_kwargs(config(), resolve_launch(1, env={}))
    assert "strategy" not in kwargs


def test_num_nodes_travels_only_when_there_is_more_than_one():
    launch = resolve_launch(1, env={"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "4"})
    assert train_kwargs(config(), launch)["num_nodes"] == 2


def test_a_smoke_test_shortens_the_run():
    kwargs = train_kwargs(config(max_epochs=400, smoke_test=True, smoke_epochs=2), Launch(1))
    assert kwargs["max_epochs"] == 2


def test_trainer_kwargs_are_the_last_word():
    """The escape hatch has to be able to override, or it is not an escape hatch."""
    kwargs = train_kwargs(
        config(trainer_kwargs={"enable_checkpointing": True, "batch_size": 64}), Launch(1)
    )
    assert kwargs["enable_checkpointing"] is True
    assert kwargs["batch_size"] == 64


def test_unset_optional_arguments_are_left_to_scvi():
    kwargs = train_kwargs(config(), Launch(1))
    assert "train_size" not in kwargs
    assert "precision" not in kwargs
    assert "plan_kwargs" not in kwargs


# -- the compile wrapper ------------------------------------------------------------------


class Wrapper:
    def __init__(self, inner):
        self._orig_mod = inner


def test_a_compiled_module_is_unwrapped_before_it_can_be_saved():
    inner = object()
    model = FakeModel(module=Wrapper(inner))
    assert unwrap_compiled(model) is True
    assert model.module is inner


def test_an_uncompiled_module_is_left_alone():
    inner = object()
    model = FakeModel(module=inner)
    assert unwrap_compiled(model) is False
    assert model.module is inner


# -- the record ---------------------------------------------------------------------------


def test_the_record_reads_back_through_the_consumer(tmp_path):
    """A key renamed here is otherwise found by the stage that needed it, hours later."""
    from scads_drvi.factorize.model import fit_meta

    cfg = config(
        fit="a_fit",
        n_latent=48,
        n_split_latent=4,
        batch_key="a_covariate",
        layer="counts",
        min_fragment=1000,
        gene_likelihood="pnb",
        dispersion="gene",
        seed=7,
        devices=4,
        batch_size=128,
    )
    launch = resolve_launch(4, env={})
    record = fit_record(
        cfg,
        launch,
        n_cells=1000,
        n_features=2000,
        gate={"n_cells_kept": 1000, "n_cells_excluded": 24},
        epochs_run=399,
        elapsed_s=61.25,
        compiled=True,
        world_size=4,
    )

    project = Project(root=tmp_path)
    out = project.fit_dir("a_fit")
    out.mkdir(parents=True)
    (out / "fit.meta.json").write_text(json.dumps(record))

    meta = fit_meta(project, "a_fit")
    assert meta.n_latent == 48
    assert meta.n_split_latent == 4
    assert meta.batch_key == "a_covariate"
    assert meta.layer == "counts"
    assert meta.min_fragment == 1000
    assert meta.is_gated
    assert meta.seed == 7
    assert meta.n_cells == 1000
    assert meta.n_features == 2000
    # Everything nested stays reachable rather than being dropped on the way through.
    assert meta.raw["train"]["effective_batch_size"] == 512
    assert meta.raw["train"]["world_size"] == 4
    assert meta.raw["gate"]["n_cells_excluded"] == 24


def test_the_record_says_which_point_on_the_trade_off_was_taken():
    cfg = config(devices=4, find_unused_parameters=False)
    launch = resolve_launch(4, default_strategy=cfg.default_strategy(), env={})
    record = fit_record(
        cfg,
        launch,
        n_cells=10,
        n_features=10,
        gate=None,
        epochs_run=1,
        elapsed_s=1.0,
        compiled=False,
        world_size=4,
    )
    assert record["train"]["find_unused_parameters"] is False
    assert record["train"]["strategy"] == DDP_STRICT_STRATEGY


def test_an_ungated_run_records_no_gate_and_no_depth_column():
    record = fit_record(
        config(),
        Launch(1, world_size=1),
        n_cells=10,
        n_features=10,
        gate=None,
        epochs_run=1,
        elapsed_s=1.0,
        compiled=False,
        world_size=1,
    )
    assert "gate" not in record
    assert record["min_fragment"] is None
    assert record["depth_col"] is None


def test_the_record_is_json_serialisable_including_a_device_list():
    record = fit_record(
        config(devices=[0, 1]),
        resolve_launch([0, 1], env={}),
        n_cells=10,
        n_features=10,
        gate=None,
        epochs_run=1,
        elapsed_s=1.0,
        compiled=False,
        world_size=2,
    )
    assert json.loads(json.dumps(record))["train"]["devices"] == [0, 1]


# -- the command ---------------------------------------------------------------------------

PRESET_MODULE = textwrap.dedent(
    """
    from scads_drvi.factorize.train import TrainConfig

    PRESETS = {"train": {
        "production": TrainConfig(adata="/nowhere/input.h5ad", fit="a_fit", devices=1),
        "wide": TrainConfig(adata="/nowhere/input.h5ad", fit="b_fit", n_latent=96,
                            devices=4, find_unused_parameters=False, seed=0),
    }}
    """
)


@pytest.fixture
def preset_module(tmp_path):
    path = tmp_path / "an_arm_config.py"
    path.write_text(PRESET_MODULE)
    return path


def test_the_command_hands_train_the_preset_and_nothing_else(monkeypatch, tmp_path, preset_module):
    """No parameter is reachable from the command line -- see the preset machinery."""
    from scads_drvi.factorize import train as module

    seen = {}
    monkeypatch.setattr(module, "train", lambda project, config, **kw: seen.update(
        project=project, config=config, **kw) or tmp_path)

    module.main(["wide", "--config", str(preset_module), "--root", str(tmp_path)])

    assert seen["config"].n_latent == 96
    assert seen["config"].find_unused_parameters is False
    assert seen["project"].root == tmp_path
    # The preset name alone does not pin the values -- the module is editable -- so the
    # record carries the hash of the module the values actually came from.
    assert seen["provenance"]["preset"] == "wide"
    assert len(seen["provenance"]["config_sha256"]) == 64
    assert seen["provenance"]["resolved"]["n_latent"] == 96


def test_the_command_refuses_a_preset_that_is_not_a_train_config(monkeypatch, tmp_path):
    path = tmp_path / "an_arm_config.py"
    path.write_text('PRESETS = {"train": {"production": object()}}')
    with pytest.raises(SystemExit, match="not a TrainConfig"):
        from scads_drvi.factorize.train import main

        main(["--config", str(path), "--root", str(tmp_path)])


def _run_module(args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "scads_drvi.factorize.train", *args],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=cwd,
    )


def test_the_module_runs_as_a_command(preset_module):
    proc = _run_module(["--list", "--config", str(preset_module)])
    assert proc.returncode == 0, proc.stderr
    assert "production" in proc.stdout
    assert "n_latent = 96" in proc.stdout


def test_running_under_dash_m_does_not_split_the_config_class_in_two(preset_module, tmp_path):
    """`python -m` executes the module as `__main__`.

    Its `TrainConfig` is then a different class object from the one the preset module
    imports from `scads_drvi.factorize.train`, and `isinstance` rejects every valid
    preset. The entry point re-enters through the canonical module to avoid it; this is
    what notices if that guard is removed.

    The run is expected to fail -- there is no such input file, and no GPU is promised --
    but it must fail *past* the preset, not on it.
    """
    proc = _run_module(["production", "--config", str(preset_module), "--root", str(tmp_path)])
    assert proc.returncode != 0
    assert "not a TrainConfig" not in proc.stdout + proc.stderr
