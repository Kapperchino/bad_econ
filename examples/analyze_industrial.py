import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class AnnualAccounts:
    first_day: int
    living_at_start: int
    last_day: int = 0
    living_at_end: int = 0
    living_person_days: int = 0
    births: int = 0
    deaths: list[int] = field(default_factory=lambda: [0, 0, 0])
    paid_household_days: int = 0
    food_required: int = 0
    bread_consumed: int = 0
    food_deficit: int = 0
    bread_produced: int = 0
    household_bread_produced: int = 0
    firm_entries: int = 0
    firm_closures: int = 0

    def observe(self, previous, current):
        population = current["population"]
        self.last_day = current["day"]
        self.living_at_end = population["living"]
        self.living_person_days += population["living"]
        self.births += population["births"] - previous["population"]["births"]
        self.deaths = [total + after - before for total, before, after in zip(
            self.deaths, previous["population"]["deaths"], population["deaths"])]
        self.paid_household_days += current["employed"]
        consumed = current["consumed"] - previous["consumed"]
        self.food_required += consumed + current["shortages"]
        self.bread_consumed += consumed
        self.food_deficit += current["shortages"]
        self.bread_produced += current["produced"][2] - previous["produced"][2]
        self.household_bread_produced += current["household_production"][2] - previous["household_production"][2]
        self.firm_entries += current["firm_entries"] - previous["firm_entries"]
        self.firm_closures += current["firm_closures"] - previous["firm_closures"]

    def report(self, initial_population):
        days = self.last_day - self.first_day + 1
        assert self.living_at_start + self.births == self.living_at_end + sum(self.deaths)
        return {
            **asdict(self),
            "days": days,
            "mean_paid_households": self.paid_household_days / days,
            "mean_daily_food_requirement": self.food_required / days,
            "mean_daily_food_deficit": self.food_deficit / days,
            "food_deficit_per_thousand_initial_person_days": self.food_deficit * 1000 / (initial_population * days),
            "births_per_thousand_person_years": self.births * 365000 / max(self.living_person_days, 1),
        }


def analyze(root, case):
    previous = {
        "population": {"living": case["population_initial"], "births": 0, "deaths": [0, 0, 0]},
        "consumed": 0, "produced": [0] * 16, "household_production": [0] * 16,
        "firm_entries": 0, "firm_closures": 0,
    }
    years = []
    count = 0
    with (root / (case["name"] + ".jsonl")).open() as stream:
        for line in stream:
            current = json.loads(line)
            count += 1
            assert current["day"] == count
            if (count - 1) % 365 == 0:
                years.append(AnnualAccounts(count, previous["population"]["living"]))
            years[-1].observe(previous, current)
            previous = current
    assert count == case["days"]
    assert previous["population"] == case["final"]["population"]
    assert sum(year.bread_consumed for year in years) == case["final"]["consumed"]
    return {"name": case["name"], "population_initial": case["population_initial"],
            "scenario_sha256": case["scenario_sha256"],
            "years": [year.report(case["population_initial"]) for year in years]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    reports = []
    for root in arguments.runs:
        source = json.loads((root / "report.json").read_text())
        reports.append({"binary_sha256": source["binary_sha256"], "workgroup": source["workgroup"],
                        "cases": [analyze(root, case) for case in source["cases"]]})
    result = {
        "units": "Food is in bread ticks; deaths are background, occupational, famine. Person-days use the living population after each audited day. Food required equals consumed bread plus recorded unmet requirements.",
        "interpretation": "Annual accounts describe replenishment, employment, and population change without assigning a recovery verdict. Falling food requirements can accompany deaths. Partial final years retain their actual day count.",
        "runs": reports,
    }
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
