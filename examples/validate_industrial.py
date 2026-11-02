import argparse
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import subprocess


class Change(Enum):
    BASELINE = "baseline"
    EMPTY = "empty-buffers"
    TRANSPORT = "slow-transport"
    BASIC = "concentrated-no-research"
    HIGH = "mortality-double"
    LOW = "mortality-half"


@dataclass(frozen=True)
class Case:
    change: Change
    seed: int
    households: int

    @property
    def name(self):
        return f"{self.change.value}-{self.households}-{self.seed}"


def execute(command, log):
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=stream, check=True)


def mortality(profile, numerator, denominator):
    for key in ["background_annual_ppm", "occupational_annual_ppm"]:
        profile[key] = [rate * numerator // denominator for rate in profile[key]]
    profile["famine_daily_ppm_at_thirty_days"] = profile["famine_daily_ppm_at_thirty_days"] * numerator // denominator


def run_case(arguments, case):
    base = arguments.output / case.name
    scenario = base.with_suffix(".scenario.json")
    trajectory = base.with_suffix(".jsonl")
    result = base.with_suffix(".json")
    experiment = "concentrated" if case.change == Change.BASIC else "baseline"
    execute([str(arguments.binary), "scenario", "--households", str(case.households),
             "--seed", str(case.seed), "--experiment", experiment, "--output", str(scenario)],
            base.with_suffix(".scenario.log"))
    config = json.loads(scenario.read_text())
    changes = {
        Change.BASELINE: lambda: None,
        Change.EMPTY: lambda: config.update(buffer_days=0),
        Change.TRANSPORT: lambda: config.update(shipment_speed=max(1, config["shipment_speed"] // 4)),
        Change.BASIC: lambda: config.update(research=None),
        Change.HIGH: lambda: mortality(config["demography"]["mortality"], 2, 1),
        Change.LOW: lambda: mortality(config["demography"]["mortality"], 1, 2),
    }
    changes[case.change]()
    scenario.write_text(json.dumps(config, sort_keys=True))
    command = [str(arguments.binary), "run", "--scenario", str(scenario), "--backend", "gpu",
               "--workgroup", str(arguments.workgroup), "--days", str(arguments.days),
               "--trajectory", str(trajectory), "--output", str(result)]
    if case.households >= 1_000_000:
        command.extend(["--checkpoint", str(base.with_suffix(".checkpoint.json")),
                        "--checkpoint-every", "365"])
    execute(command, base.with_suffix(".log"))
    report = json.loads(result.read_text())
    opening = report["opening"]
    initial_population = case.households * config["demography"]["initial_household_members"]
    days = 0
    peak_deficit = 0
    milestones = []
    with trajectory.open() as stream:
        for line in stream:
            snapshot = json.loads(line)
            days += 1
            assert snapshot["day"] == days
            assert snapshot["cash"] == opening["cash"], (case.name, days, "cash")
            people = snapshot["population"]
            assert people["living"] + sum(people["deaths"]) == initial_population + people["births"], (case.name, days, "population")
            for good in range(16):
                resources = opening["goods"][good] + snapshot["produced"][good]
                accounts = sum(snapshot[field][good] for field in ["inventories", "transit", "used", "consumption"])
                assert resources == accounts, (case.name, days, good, resources, accounts)
            peak_deficit = max(peak_deficit, snapshot["shortages"])
            if days in [1, 30, 60, 90, 180, 270, 365, 730, 1095, arguments.days]:
                milestones.append(snapshot)
    assert days == arguments.days
    return {"name": case.name, "scenario_sha256": hashlib.sha256(scenario.read_bytes()).hexdigest(),
            "households_initial": case.households, "population_initial": initial_population,
            "days": days, "all_daily_accounts_exact": True, "peak_food_deficit": peak_deficit,
            "opening": opening, "seconds": report["seconds"], "backend": report["backend"],
            "final": report["snapshot"], "milestones": milestones}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/release/bad_econ"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["small", "million"], default="small")
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--workgroup", type=int, default=128)
    parser.add_argument("--case", action="append", choices=[change.value for change in Change])
    arguments = parser.parse_args()
    arguments.binary = arguments.binary.resolve()
    arguments.output.mkdir(parents=True, exist_ok=True)
    cases = [Case(change, seed, 1000) for change, seed in [
        (Change.BASELINE, 42), (Change.BASELINE, 73), (Change.BASIC, 42), (Change.BASIC, 73),
        (Change.EMPTY, 42), (Change.TRANSPORT, 73), (Change.HIGH, 42), (Change.LOW, 42),
    ]] if arguments.suite == "small" else [Case(Change.BASELINE, 42, 1_000_000), Case(Change.TRANSPORT, 73, 1_000_000)]
    if arguments.case:
        cases = [case for case in cases if case.change.value in arguments.case]
    if not cases:
        parser.error("the selected cases are unavailable in this suite")
    report = {"binary_sha256": hashlib.sha256(arguments.binary.read_bytes()).hexdigest(),
              "workgroup": arguments.workgroup, "cases": []}
    for case in cases:
        (arguments.output / "status.json").write_text(json.dumps({"case": case.name, "state": "running"}))
        try:
            result = run_case(arguments, case)
        except (subprocess.CalledProcessError, AssertionError) as error:
            (arguments.output / "status.json").write_text(json.dumps({"case": case.name, "state": "failed", "error": str(error)}))
            raise
        report["cases"].append(result)
        (arguments.output / "report.json").write_text(json.dumps(report, indent=2))
    (arguments.output / "status.json").write_text(json.dumps({"state": "complete"}))


if __name__ == "__main__":
    main()
