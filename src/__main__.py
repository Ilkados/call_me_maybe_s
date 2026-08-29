import argparse
import json
import sys
from pathlib import Path
from src.generator import LLMGenerator
from src.parser import FunctionDefinition, InputPrompt
from typing import Any


def write_output(entries: list[Any], path: str) -> None:
    """Write results list to a JSON file, creating directories as needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=4)
    except Exception as error:
        print(f"Error writing output: {error}")


def main() -> None:
    """Entry point: parse args, load inputs, run generation, write output."""
    cli = argparse.ArgumentParser(
        description="Function calling with constrained decoding."
    )
    cli.add_argument(
        '--functions_definition',
        default='data/input/functions_definition.json',
        help='Path to function definitions JSON file'
    )
    cli.add_argument(
        '--input',
        default='data/input/function_calling_tests.json',
        help='Path to input prompts JSON file'
    )
    cli.add_argument(
        '--output',
        default='data/output/function_calling_results.json',
        help='Path to output JSON file'
    )

    args = cli.parse_args()

    unknowen_fun = {
        "name": "No function available for that prompt",
        "description": (
            "Select this function only if no available function"
            " matches the user's prompt after considering all "
            "provided function definitions."
        ),
        "parameters": {},
        "returns": {}
    }

    try:
        with open(args.functions_definition, 'r') as f:
            functions = json.load(f)
            if not isinstance(functions, list):
                raise ValueError(
                    "Functions definition file must contain a JSON array.")
            if not functions:
                raise ValueError(
                    "Functions definition file cannot be empty.")
        with open(args.input, 'r') as f:
            prompts = json.load(f)
            if not isinstance(prompts, list):
                raise ValueError(
                    "Functions calling test file must contain a JSON array.")
            if not prompts:
                raise ValueError(
                    "Functions calling test file cannot be empty.")
        for fn in functions:
            _ = FunctionDefinition.model_validate(fn)
        functions.append(unknowen_fun)

    except (KeyboardInterrupt, Exception) as e:
        print(f"Error reading input files: {e}", file=sys.stderr)
        sys.exit(1)

    results: list[Any] = []
    generator = LLMGenerator(functions)
    try:
        for i, item in enumerate(prompts):
            user_prompt = ""
            try:
                _ = InputPrompt.model_validate(item)
                print(f"\n========... start {i + 1} generation ...========")
                user_prompt = item["prompt"]
                if user_prompt == "":
                    raise ValueError('No prompt founded')
                result = generator.generate(user_prompt)
                results.append(result)
                print(f"==> [{i + 1}] Generated succesfully : ", result)
            except Exception as e:
                print(
                    f"Error generating for prompt '{user_prompt}': {e}",
                    file=sys.stderr
                )
                results.append({
                    "prompt": user_prompt,
                    "name": "",
                    "parameters": {}
                })

        write_output(results, args.output)
        print(f"\n========[Output written to {args.output}]========")
    except (Exception, KeyboardInterrupt):
        print("Error!", file=sys.stderr)


if __name__ == "__main__":
    main()
