"""Run non-fine-tuned Marshall Fire inference.
By default, all rows and all configured experiments are processed.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from model import LLMModel


ID_COLUMNS = {
    "persona_id": "string",
    "geoid": "string",
    "tract_id": "string",
}

REQUIRED_COLUMNS = {
    "persona_id",
    "geoid",
    "tract_id",
    "persona_output",
    "distance_to_fire_km",
    "direction_from_fire",
    "total_population_category",
    "median_income_category",
    "white_population_category",
    "alert_level",
}

GROUND_TRUTH_COLUMNS = ("evacuated", "evacuation_label")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="inference_config.yaml",
        help="Path to the YAML inference configuration.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Run one experiment only (e.g., E1). Default: all configured experiments.",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit to process the complete CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print one prompt without loading the vLLM engine.",
    )
    return parser.parse_args()


def get_path_from_env_or_config(env_name, config_value):
    value = os.environ.get(env_name)
    if value:
        return Path(value).expanduser().resolve()
    if not config_value:
        raise ValueError(
            f"Set {env_name} or provide its value in inference_config.yaml."
        )
    return Path(config_value).expanduser().resolve()


def load_dataset(data_path, max_agents=None):
    if not data_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {data_path}")

    df = pd.read_csv(data_path, dtype=ID_COLUMNS)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "synthetic_data.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df["persona_id"].isna().any():
        raise ValueError("persona_id contains missing values.")
    if df["persona_id"].duplicated().any():
        raise ValueError("persona_id must be unique in synthetic_data.csv.")

    if max_agents is not None:
        if max_agents <= 0:
            raise ValueError("--max-agents must be a positive integer.")
        df = df.head(max_agents).copy()

    return df.reset_index(drop=True)


def build_sampling_params(config):
    generation = config["generation"]
    return SamplingParams(
        max_tokens=int(generation["max_tokens"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
        n=1,
    )


def load_engine(config, model_path, seed):
    model_cfg = config["model"]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )

    llm = LLM(
        model=str(model_path),
        dtype=str(model_cfg["dtype"]),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        enforce_eager=bool(model_cfg.get("enforce_eager", False)),
        max_model_len=int(model_cfg["max_model_len"]),
        gpu_memory_utilization=float(model_cfg["gpu_memory_utilization"]),
        max_num_seqs=int(model_cfg["max_num_seqs"]),
        seed=int(seed),
    )

    return llm, tokenizer, build_sampling_params(config)


def ground_truth_value(row, key):
    if key not in row.index:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    return value


def make_prediction_record(agent, iteration, experiment, ground_truth_by_id):
    predicted = (
        True if agent.final_decision == "evacuate"
        else False if agent.final_decision == "not evacuate"
        else None
    )
    truth = ground_truth_by_id.get(agent.uid, {})

    return {
        "iteration": iteration,
        "experiment": experiment,
        "agent_id": agent.uid,
        "geoid": agent.geoid,
        "tract_id": agent.tract_id,
        "fire_distance_km": agent.home_fire_distance,
        "direction_from_fire": agent.fire_direction,
        "alert_level": agent.alert,
        "predicted_decision": agent.final_decision,
        "predicted_evacuated": predicted,
        "ground_truth_evacuated": truth.get("evacuated"),
        "ground_truth_label": truth.get("evacuation_label"),
        "inference_seconds": agent.inference_seconds,
        "generated_response": agent.generated_response,
    }


def main():
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    paths_cfg = config["paths"]
    model_path = get_path_from_env_or_config(
        "MODEL_PATH",
        paths_cfg.get("model_path"),
    )
    data_path = get_path_from_env_or_config(
        "DATA_PATH",
        paths_cfg.get("data_csv"),
    )
    output_dir = get_path_from_env_or_config(
        "OUTPUT_DIR",
        paths_cfg.get("output_dir"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = config["run"]
    iteration = int(os.environ.get("ITERATION", run_cfg.get("iteration", 1)))
    base_seed = int(os.environ.get("BASE_SEED", run_cfg.get("base_seed", 5045)))
    seed = int(os.environ.get("VLLM_SEED", base_seed + iteration))

    max_agents = args.max_agents
    if max_agents is None and os.environ.get("MAX_AGENTS"):
        max_agents = int(os.environ["MAX_AGENTS"])
    if max_agents is None:
        max_agents = config["data"].get("max_agents")
    if max_agents in (0, "0"):
        max_agents = None

    df = load_dataset(data_path, max_agents=max_agents)

    ground_truth_by_id = {}
    for _, row in df.iterrows():
        truth = {
            key: ground_truth_value(row, key)
            for key in GROUND_TRUTH_COLUMNS
        }
        ground_truth_by_id[str(row["persona_id"])] = {
            key: value
            for key, value in truth.items()
            if value is not None
        }

    experiments = config["experiments"]
    if args.experiment:
        if args.experiment not in experiments:
            raise ValueError(
                f"Unknown experiment {args.experiment}. "
                f"Available: {', '.join(experiments)}"
            )
        experiments = {args.experiment: experiments[args.experiment]}

    print("=" * 78)
    print("NON-FINE-TUNED MARSHALL FIRE INFERENCE")
    print(f"Base model: {model_path}")
    print(f"Input data: {data_path}")
    print(f"Agents: {len(df):,}")
    print(f"Iteration: {iteration} | Seed: {seed}")
    print(f"Experiments: {list(experiments)}")
    print("LoRA adapter: NONE")
    print("=" * 78)

    prompt_cfg = config["prompt"]

    if args.dry_run:
        from agent import LLMAgent

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
            local_files_only=bool(config["model"].get("local_files_only", True)),
        )
        first_experiment = next(iter(experiments))
        first_row = df.iloc[0].to_dict()

        class _DummyModel:
            current_step = 1
            evacuated_agents = []
            schedule = None

        dummy = LLMAgent(
            unique_id=0,
            model_mesa=_DummyModel(),
            agent_data=first_row,
            llm=None,
            tokenizer=tokenizer,
            sampling_params=None,
            prompt_config=prompt_cfg,
            included_keys=experiments[first_experiment],
        )
        print(dummy.generate_prompt())
        print("\nDry run completed. No GPU inference was performed.")
        return

    llm, tokenizer, sampling_params = load_engine(config, model_path, seed)

    print_prompts = bool(config.get("logging", {}).get("print_prompts", False))
    for experiment, included_keys in experiments.items():
        print(f"\nStarting {experiment}: included features={included_keys}")

        model = LLMModel(
            df=df,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            prompt_config=prompt_cfg,
            included_keys=included_keys,
            seed=seed,
            print_prompts=print_prompts,
        )
        model.step()

        records = [
            make_prediction_record(
                agent,
                iteration,
                experiment,
                ground_truth_by_id,
            )
            for agent in model.agent_list
        ]
        result_df = pd.DataFrame(records)

        experiment_dir = (
            output_dir / f"iteration_{iteration}" / experiment
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = experiment_dir / "agent_predictions.csv"
        result_df.to_csv(predictions_path, index=False)

        evacuated = int((result_df["predicted_decision"] == "evacuate").sum())
        not_evacuated = int(
            (result_df["predicted_decision"] == "not evacuate").sum()
        )
        unknown = int((result_df["predicted_decision"] == "unknown").sum())

        summary = pd.DataFrame([{
            "iteration": iteration,
            "experiment": experiment,
            "n_agents": len(result_df),
            "n_evacuated": evacuated,
            "n_not_evacuated": not_evacuated,
            "n_unparsed": unknown,
            "predicted_evacuation_rate": (
                evacuated / (evacuated + not_evacuated)
                if evacuated + not_evacuated > 0 else float("nan")
            ),
            "seed": seed,
        }])
        summary.to_csv(experiment_dir / "summary.csv", index=False)

        print(
            f"{experiment} complete | evacuated={evacuated:,} | "
            f"not evacuated={not_evacuated:,} | unparsed={unknown:,}"
        )
        print(f"Saved: {predictions_path}")

    print("\nAll requested non-fine-tuned inference experiments completed.")


if __name__ == "__main__":
    main()
