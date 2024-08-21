import json

import asyncio
from pathlib import Path

from documents import FileSystem
from lexis import extract_number


async def eval(
    name_of_file_to_save: str | Path, name_of_dir_to_read: str | Path
):
    base_path = Path("data") / "eval"

    eval_dict = {
        "all_docs": 0,
        "self_top1": 0,
        "soft": 0,
        "hard": 0,
        "avg": 0,
        "no_relevant": 0,
    }

    with open(base_path / "performance.json", encoding="utf-8") as file:
        methods = list(json.load(file).keys())

    metrics = {method: eval_dict.copy() for method in methods}

    list_of_files = list((base_path / name_of_dir_to_read).iterdir())

    for file_path in list_of_files:
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)

        citations = list(map(extract_number, data["56"]))
        cluster = list(map(extract_number, data["cluster"]))

        evaluate_data = set(citations)
        # evaluate_data = set(citations + cluster)

        for extractor_name, relevant in data["relevant"].items():
            if extractor_name not in metrics.keys():
                continue

            relevant = list(set(map(extract_number, relevant)))[:20]

            if len(evaluate_data) > 0:
                num_of_hits = 0
                for doc in evaluate_data:
                    if doc in relevant:
                        num_of_hits += 1

                metrics[extractor_name]["soft"] += num_of_hits > 0
                if len(evaluate_data) > 1:
                    metrics[extractor_name]["avg"] += num_of_hits / len(
                        evaluate_data
                    )
                    metrics[extractor_name]["hard"] += num_of_hits == len(
                        evaluate_data
                    )
                metrics[extractor_name]["all_docs"] += 1

            if relevant:
                metrics[extractor_name]["self_top1"] += (
                    relevant[0] == file_path.stem
                )
            else:
                metrics[extractor_name]["no_relevant"] += 1

    for extractor_name, eval in metrics.items():
        print(extractor_name, "metrics:")
        for metric in eval.keys():
            if metric in ["self_top1", "no_relevant"]:
                eval[metric] = round(eval[metric] / len(list_of_files))
            elif metric != "all_docs":
                eval[metric] = round(eval[metric] / eval["all_docs"], 4)

            print(metric, "-", round(eval[metric], 4))
        print()

    with open(base_path / name_of_file_to_save, "w+") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    coro = eval("eval.json", "ex_from_cluster")
    asyncio.run(coro)
