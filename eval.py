from documents import FileSystem
import asyncio
from pathlib import Path
import json


async def main(name_of_file: str | Path):
    eval_dict = {
        "all_docs": 0,
        "self_top1": 0,
        "soft": 0,
        "hard": 0,
        "avg": 0,
        "no_relevant": 0,
    }
    metrics = {
        "RAKE": eval_dict.copy(),
        "YAKE": eval_dict.copy(),
        "TXTR": eval_dict.copy(),
        "KBRT": eval_dict.copy(),
        "MPTE": eval_dict.copy(),
    }

    base_path = Path("data") / "eval"

    list_of_files = list((base_path / "kwe").iterdir())

    for file_path in list_of_files:
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)

        citations = data["56"]
        for extractor_name, relevant in data["relevant"].items():
            if len(citations) > 0:
                num_of_hits = 0
                for cite in citations:
                    if cite in relevant:
                        num_of_hits += 1

                metrics[extractor_name]["soft"] += num_of_hits > 0
                if len(citations) > 1:
                    metrics[extractor_name]["avg"] += num_of_hits / len(citations)
                    metrics[extractor_name]["hard"] += num_of_hits == len(citations)
                metrics[extractor_name]["all_docs"] += 1

            if relevant:
                metrics[extractor_name]["self_top1"] += relevant[0] == file_path.stem
            else:
                metrics[extractor_name]["no_relevant"] += 1

    for extractor_name, eval in metrics.items():
        print(extractor_name, "metrics:")
        for metric in eval.keys():
            if metric in ["self_top1", "no_relevant"]:
                eval[metric] = round(eval[metric] / len(list_of_files))
            elif metric != "all_docs":
                eval[metric] = round(eval[metric] / eval["all_docs"], 2)

            print(metric, "-", round(eval[metric], 2))
        print()

    with open(base_path / name_of_file, "w+") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main("eval.json"))
